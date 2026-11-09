#!/usr/bin/env python3
"""
Generate Chakra execution traces (.et) for ASTRA-sim:  LLM (dense transformer)
and MoE (mixture-of-experts transformer) training on a 3D torus.

Every NPU runs the same program (SPMD), so we write the same graph N times
(workload.0.et ... workload.{N-1}.et).

Parallelism -> torus dimension mapping (via the `involved_dim` attribute):
    LLM :  TP on dim0 | DP on dim1+dim2
    MoE :  TP on dim0 | EP (all-to-all) on dim1 | dense-param DP on dim1+dim2
           | expert-param DP on dim2 only (each EP rank owns different experts)

Usage:
    python3 gen_workload.py --model llm --out ../workload/llm_torus/llm
    python3 gen_workload.py --model moe --out ../workload/moe_torus/moe
Run with -h to see every knob (hidden size, layers, seq len, experts, ...).
"""
import argparse, os, sys

CHAKRA = os.environ.get(
    "CHAKRA_DIR",
    os.path.expanduser("~/astra-sim/extern/graph_frontend/chakra"))
sys.path.insert(0, os.path.join(CHAKRA, "schema", "protobuf"))
sys.path.insert(0, os.path.join(CHAKRA, "src", "third_party", "utils"))
from et_def_pb2 import (Node, GlobalMetadata, AttributeProto as Attr, BoolList,  # noqa: E402
                        COMP_NODE, COMM_COLL_NODE, ALL_REDUCE, ALL_TO_ALL)
from protolib import encodeMessage  # noqa: E402

BYTES = 2  # fp16/bf16


class Graph:
    def __init__(self, ndims):
        self.nodes, self.nid, self.ndims = [], 0, ndims

    def _new(self, name, ntype, deps):
        n = Node(id=self.nid, name=name, type=ntype)
        self.nid += 1
        n.data_deps.extend([d.id for d in deps if d is not None])
        n.attr.append(Attr(name="is_cpu_op", bool_val=False))
        self.nodes.append(n)
        return n

    def comp(self, name, us, deps):
        n = self._new(name, COMP_NODE, deps)
        n.duration_micros = max(1, int(us))
        return n

    def coll(self, name, ctype, size_bytes, dims, deps, priority=0):
        n = self._new(name, COMM_COLL_NODE, deps)
        n.attr.append(Attr(name="comm_type", int64_val=ctype))
        n.attr.append(Attr(name="comm_size", int64_val=int(size_bytes)))
        mask = [d in dims for d in range(self.ndims)]
        n.attr.append(Attr(name="involved_dim", bool_list=BoolList(values=mask)))
        if priority:
            n.attr.append(Attr(name="comm_priority", int32_val=priority))
        return n


def build(a):
    g = Graph(ndims=3)
    TP, DP_DIMS = [0], [1, 2]
    EP_DIM, EXP_DP_DIMS = [1], [2]

    tokens = a.micro_batch * a.seq
    act = tokens * a.hidden * BYTES                 # one activation tensor
    tflops = a.tflops * 1e12

    def t(flops):                                   # flops -> microseconds
        return flops / a.tp / tflops * 1e6

    # per-layer FLOPs (fwd); bwd ~= 2x fwd, split equally into IG and WG
    attn_flops = 8 * tokens * a.hidden ** 2 + 4 * tokens * a.seq * a.hidden
    if a.model == "llm":
        mlp_flops = 16 * tokens * a.hidden ** 2      # 4h FFN, 2 GEMMs
        dense_params = 12 * a.hidden ** 2 / a.tp
        expert_params = 0
    else:
        # top-k routed tokens through a SwiGLU expert (3 GEMMs, ffn_hidden)
        mlp_flops = a.topk * 6 * tokens * a.hidden * a.ffn
        dense_params = 4 * a.hidden ** 2 / a.tp       # attention only
        local_experts = a.experts // a.ep
        expert_params = local_experts * 3 * a.hidden * a.ffn / a.tp
        a2a = tokens * a.topk * a.hidden * BYTES / a.tp  # dispatch/combine buffer

    last_grad = [None] * a.layers   # grad all-reduce of previous iteration
    prev = None
    for it in range(a.iters):
        # ---------------- forward ----------------
        fwd_out = []
        for l in range(a.layers):
            p = f"it{it}_L{l}"
            deps = [prev] + ([last_grad[l]] if last_grad[l] else [])
            c = g.comp(f"{p}_fwd_attn", t(attn_flops), deps)
            prev = g.coll(f"{p}_fwd_attn_TP_AR", ALL_REDUCE, act, TP, [c])
            if a.model == "llm":
                c = g.comp(f"{p}_fwd_mlp", t(mlp_flops), [prev])
                prev = g.coll(f"{p}_fwd_mlp_TP_AR", ALL_REDUCE, act, TP, [c])
            else:
                r = g.comp(f"{p}_fwd_router", 20, [prev])
                d = g.coll(f"{p}_fwd_dispatch_A2A", ALL_TO_ALL, a2a, EP_DIM, [r])
                c = g.comp(f"{p}_fwd_experts", t(mlp_flops), [d])
                prev = g.coll(f"{p}_fwd_combine_A2A", ALL_TO_ALL, a2a, EP_DIM, [c])
            fwd_out.append(prev)

        # ---------------- backward ----------------
        grads = [None] * a.layers
        for l in reversed(range(a.layers)):
            p = f"it{it}_L{l}"
            if a.model == "llm":
                ig = g.comp(f"{p}_bwd_mlp_ig", t(mlp_flops), [prev])
                wg = g.comp(f"{p}_bwd_mlp_wg", t(mlp_flops), [ig])
                x = g.coll(f"{p}_bwd_mlp_TP_AR", ALL_REDUCE, act, TP, [ig])
            else:
                d = g.coll(f"{p}_bwd_combine_A2A", ALL_TO_ALL, a2a, EP_DIM, [prev])
                ig = g.comp(f"{p}_bwd_experts_ig", t(mlp_flops), [d])
                wg = g.comp(f"{p}_bwd_experts_wg", t(mlp_flops), [ig])
                x = g.coll(f"{p}_bwd_dispatch_A2A", ALL_TO_ALL, a2a, EP_DIM, [ig])
                # expert grads: sync only among replicas of the same experts
                g.coll(f"{p}_bwd_expert_DP_AR", ALL_REDUCE,
                       expert_params * BYTES, EXP_DP_DIMS, [wg])
            aig = g.comp(f"{p}_bwd_attn_ig", t(attn_flops), [x])
            awg = g.comp(f"{p}_bwd_attn_wg", t(attn_flops), [aig, wg])
            prev = g.coll(f"{p}_bwd_attn_TP_AR", ALL_REDUCE, act, TP, [aig])
            # dense-weight gradient sync across data-parallel replicas
            grads[l] = g.coll(f"{p}_bwd_grad_DP_AR", ALL_REDUCE,
                              dense_params * BYTES, DP_DIMS, [awg])
        last_grad = grads
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["llm", "moe"], required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. ../workload/llm_torus/llm")
    ap.add_argument("--npus", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--tp", type=int, default=4, help="must equal torus dim0 size")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--tflops", type=float, default=400.0, help="effective TFLOP/s per NPU")
    ap.add_argument("--experts", type=int, default=8, help="MoE only")
    ap.add_argument("--topk", type=int, default=2, help="MoE only")
    ap.add_argument("--ep", type=int, default=4, help="MoE only; must equal torus dim1 size")
    ap.add_argument("--ffn", type=int, default=14336, help="MoE expert FFN hidden size")
    a = ap.parse_args()

    g = build(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    meta = GlobalMetadata(attr=[Attr(name="schema", string_val="1.0.2-chakra.0.0.4")])
    for npu in range(a.npus):
        with open(f"{a.out}.{npu}.et", "wb") as f:
            encodeMessage(f, meta)
            for n in g.nodes:
                encodeMessage(f, n)
    print(f"[{a.model}] wrote {a.npus} traces, {len(g.nodes)} nodes each -> {a.out}.*.et")


if __name__ == "__main__":
    main()
