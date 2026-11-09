#!/usr/bin/env python3
"""Run a single-job Dense LLM workload with TP=4 and DP=4 on a 16-GPU Ring."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


PINNED_ASTRA_COMMIT = "4ce9ecfe746a3ee5fe47559a11d91644d641fa7f"
FINISH_PATTERN = re.compile(r"sys\[(\d+)\] finished, (\d+) cycles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Dense LLM TP=4, DP=4 Ring ASTRA-sim experiment."
    )

    parser.add_argument(
        "--astra",
        type=Path,
        required=True,
        help="Path to the ASTRA-sim checkout",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/llm-ring"),
        help="Directory containing LLM Ring configuration files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/llm-ring-tpdp"),
        help="Directory for experiment results",
    )

    return parser.parse_args()


def validate_inputs(config: dict, config_dir: Path) -> None:
    """Validate the expected 16-GPU TP=4, DP=4 Ring configuration."""

    if config["topology_npus"] != 16:
        raise ValueError("This experiment currently expects exactly 16 GPUs.")

    if config["tensor_parallel_size"] != 4:
        raise ValueError("tensor_parallel_size must be 4.")

    if config["data_parallel_size"] != 4:
        raise ValueError("data_parallel_size must be 4.")

    if (
        config["tensor_parallel_size"] * config["data_parallel_size"]
        != config["topology_npus"]
    ):
        raise ValueError(
            "tensor_parallel_size * data_parallel_size must equal topology_npus"
        )

    network_path = config_dir / "network_16gpu_ring.yml"
    network_text = network_path.read_text()

    expected_network_values = {
        "npus_count": config["topology_npus"],
        "bandwidth": config["bandwidth_GBps"],
        "latency": config["latency_ns"],
    }

    for key, expected in expected_network_values.items():
        match = re.search(
            rf"^{key}:\s*\[\s*([0-9.]+)\s*\]",
            network_text,
            re.MULTILINE,
        )

        if not match or float(match.group(1)) != float(expected):
            raise ValueError(
                f"{key} in network_16gpu_ring.yml must equal {expected}"
            )

    if not re.search(
        r"^topology:\s*\[\s*Ring\s*\]",
        network_text,
        re.MULTILINE,
    ):
        raise ValueError("network_16gpu_ring.yml must use Ring topology")

    system = json.loads(
        (config_dir / "system_ring.json").read_text()
    )

    if system.get("all-reduce-implementation") != ["ring"]:
        raise ValueError(
            "system_ring.json must use Ring AllReduce"
        )


def load_chakra(astra: Path) -> dict:
    """Import Chakra definitions from the ASTRA-sim checkout."""

    sys.path.insert(0, str(astra))

    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        ALL_REDUCE,
        COMM_COLL_NODE,
        COMP_NODE,
        AttributeProto,
        GlobalMetadata,
        Node,
    )

    from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
        encodeMessage,
    )

    return {
        "ALL_REDUCE": ALL_REDUCE,
        "COMM_COLL_NODE": COMM_COLL_NODE,
        "COMP_NODE": COMP_NODE,
        "AttributeProto": AttributeProto,
        "GlobalMetadata": GlobalMetadata,
        "Node": Node,
        "encodeMessage": encodeMessage,
    }


def build_groups() -> tuple[
    dict[str, list[int]],
    dict[int, str],
    dict[int, str],
]:
    """Create TP and DP communicator groups for 16 GPUs."""

    tp_groups = {
        "1": [0, 1, 2, 3],
        "2": [4, 5, 6, 7],
        "3": [8, 9, 10, 11],
        "4": [12, 13, 14, 15],
    }

    dp_groups = {
        "5": [0, 4, 8, 12],
        "6": [1, 5, 9, 13],
        "7": [2, 6, 10, 14],
        "8": [3, 7, 11, 15],
    }

    groups = {
        **tp_groups,
        **dp_groups,
    }

    rank_to_tp_group = {
        rank: group_id
        for group_id, members in tp_groups.items()
        for rank in members
    }

    rank_to_dp_group = {
        rank: group_id
        for group_id, members in dp_groups.items()
        for rank in members
    }

    return groups, rank_to_tp_group, rank_to_dp_group


def write_traces(
    case_dir: Path,
    topology_npus: int,
    iterations: int,
    compute_us: int,
    tp_message_bytes: int,
    dp_message_bytes: int,
    rank_to_tp_group: dict[int, str],
    rank_to_dp_group: dict[int, str],
    chakra: dict,
) -> None:
    """Generate one Chakra trace per GPU with TP and DP AllReduce phases."""

    for rank in range(topology_npus):
        path = case_dir / f"workload.{rank}.et"

        with path.open("wb") as stream:
            chakra["encodeMessage"](
                stream,
                chakra["GlobalMetadata"](version="0.0.4"),
            )

            previous_id: int | None = None
            node_id = 0

            def emit(
                node_type: int,
                name: str,
                duration_us: int = 0,
                collective: bool = False,
                message_bytes: int = 0,
                group_id: str | None = None,
            ) -> None:
                nonlocal previous_id, node_id

                node = chakra["Node"](
                    id=node_id,
                    name=name,
                    type=node_type,
                    duration_micros=duration_us,
                )

                if previous_id is not None:
                    node.data_deps.append(previous_id)

                node.attr.append(
                    chakra["AttributeProto"](
                        name="is_cpu_op",
                        bool_val=False,
                    )
                )

                if collective:
                    node.attr.extend(
                        [
                            chakra["AttributeProto"](
                                name="comm_type",
                                int64_val=chakra["ALL_REDUCE"],
                            ),
                            chakra["AttributeProto"](
                                name="comm_size",
                                int64_val=message_bytes,
                            ),
                            chakra["AttributeProto"](
                                name="pg_name",
                                string_val=group_id,
                            ),
                        ]
                    )

                chakra["encodeMessage"](stream, node)

                previous_id = node_id
                node_id += 1

            tp_group = rank_to_tp_group[rank]
            dp_group = rank_to_dp_group[rank]

            for iteration in range(iterations):
                emit(
                    chakra["COMP_NODE"],
                    f"llm_tp_compute_{iteration}",
                    duration_us=compute_us,
                )

                emit(
                    chakra["COMM_COLL_NODE"],
                    f"llm_tp_allreduce_{iteration}",
                    collective=True,
                    message_bytes=tp_message_bytes,
                    group_id=tp_group,
                )

                emit(
                    chakra["COMP_NODE"],
                    f"llm_dp_compute_{iteration}",
                    duration_us=compute_us,
                )

                emit(
                    chakra["COMM_COLL_NODE"],
                    f"llm_dp_allreduce_{iteration}",
                    collective=True,
                    message_bytes=dp_message_bytes,
                    group_id=dp_group,
                )

            emit(
                chakra["COMP_NODE"],
                "llm_final_compute",
                duration_us=compute_us,
            )


def write_csv(path: Path, row: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(row.keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()

    astra = args.astra.resolve()
    config_dir = args.config.resolve()
    output_dir = args.output.resolve()

    config = json.loads(
        (config_dir / "experiment.json").read_text()
    )

    validate_inputs(config, config_dir)

    astra_commit = subprocess.check_output(
        ["git", "-C", str(astra), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    if astra_commit != PINNED_ASTRA_COMMIT:
        raise RuntimeError(
            f"expected ASTRA-sim {PINNED_ASTRA_COMMIT}, "
            f"found {astra_commit}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "astra-sim-commit.txt").write_text(
        astra_commit + "\n"
    )

    astra_status = subprocess.check_output(
        ["git", "-C", str(astra), "status", "--short"],
        text=True,
    )

    (output_dir / "astra-sim-status.txt").write_text(
        astra_status
    )

    if astra_status:
        print(
            "WARNING: ASTRA-sim has local changes; "
            "provenance has been recorded."
        )

    case_dir = output_dir / "baseline"
    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups, rank_to_tp_group, rank_to_dp_group = build_groups()

    (case_dir / "groups.json").write_text(
        json.dumps(groups, indent=2) + "\n"
    )

    chakra = load_chakra(astra)

    write_traces(
        case_dir=case_dir,
        topology_npus=config["topology_npus"],
        iterations=config["measurement_iterations"],
        compute_us=config["compute_phase_us"],
        tp_message_bytes=config["tp_message_bytes"],
        dp_message_bytes=config["dp_message_bytes"],
        rank_to_tp_group=rank_to_tp_group,
        rank_to_dp_group=rank_to_dp_group,
        chakra=chakra,
    )

    shutil.copy2(
        config_dir / "network_16gpu_ring.yml",
        case_dir / "network.yml",
    )

    shutil.copy2(
        config_dir / "system_ring.json",
        case_dir / "system.json",
    )

    description = {
        "workload": "Dense LLM",
        "topology": "Ring",
        "job_count": 1,
        "gpu_count": config["topology_npus"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "data_parallel_size": config["data_parallel_size"],
        "collective": "AllReduce",
        "collective_implementation": "Ring",
        "iterations": config["measurement_iterations"],
        "compute_phase_us": config["compute_phase_us"],
        "tp_message_bytes": config["tp_message_bytes"],
        "dp_message_bytes": config["dp_message_bytes"],
        "bandwidth_GBps": config["bandwidth_GBps"],
        "latency_ns": config["latency_ns"],
        "trace_pattern": (
            "(compute -> TP AllReduce -> compute -> DP AllReduce) "
            "repeated N times -> final compute"
        ),
    }

    (case_dir / "description.json").write_text(
        json.dumps(description, indent=2) + "\n"
    )

    binary = (
        astra
        / "build/astra_analytical/build/bin/"
        "AstraSim_Analytical_Congestion_Aware"
    )

    if not binary.is_file():
        raise FileNotFoundError(
            f"ASTRA-sim binary not found: {binary}"
        )

    remote_memory = (
        Path(__file__).resolve().parents[1]
        / "configs/no_memory_expansion.json"
    )

    command = [
        str(binary),
        f"--workload-configuration={case_dir}/workload",
        f"--comm-group-configuration={case_dir}/groups.json",
        f"--system-configuration={case_dir}/system.json",
        f"--remote-memory-configuration={remote_memory}",
        f"--network-configuration={case_dir}/network.yml",
        f"--logging-folder={case_dir}/logging",
    ]

    (case_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )

    result = subprocess.run(
        command,
        cwd=astra,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "CS4251_POLICY": "immediate",
        },
    )

    (case_dir / "run.log").write_text(
        result.stdout
    )

    if result.returncode:
        raise RuntimeError(
            f"LLM Ring experiment failed with exit code "
            f"{result.returncode}. Inspect {case_dir / 'run.log'}"
        )

    completion_by_rank = {
        int(rank): int(cycles)
        for rank, cycles in FINISH_PATTERN.findall(
            result.stdout
        )
    }

    expected_ranks = set(
        range(config["topology_npus"])
    )

    if set(completion_by_rank) != expected_ranks:
        missing = sorted(
            expected_ranks
            - set(completion_by_rank)
        )

        raise RuntimeError(
            f"only {len(completion_by_rank)} of "
            f"{config['topology_npus']} ranks completed; "
            f"missing ranks: {missing}"
        )

    completion_ns = max(
        completion_by_rank.values()
    )

    completion_us = (
        completion_ns / 1000.0
    )

    compute_total_us = (
        2 * config["measurement_iterations"] + 1
    ) * config["compute_phase_us"]

    communication_and_wait_us = (
        completion_us - compute_total_us
    )

    result_row = {
        "workload": "Dense LLM",
        "topology": "Ring",
        "gpu_count": config["topology_npus"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "data_parallel_size": config["data_parallel_size"],
        "iterations": config["measurement_iterations"],
        "tp_message_bytes": config["tp_message_bytes"],
        "dp_message_bytes": config["dp_message_bytes"],
        "compute_total_us": compute_total_us,
        "completion_ns": completion_ns,
        "jct_us": round(
            completion_us,
            6,
        ),
        "communication_and_wait_us": round(
            communication_and_wait_us,
            6,
        ),
        "mean_iteration_communication_and_wait_us": round(
            communication_and_wait_us
            / config["measurement_iterations"],
            6,
        ),
    }

    write_csv(
        output_dir / "summary.csv",
        result_row,
    )

    print(
        f"LLM Ring TP/DP baseline: "
        f"{config['topology_npus']} of "
        f"{config['topology_npus']} ranks completed"
    )

    print(
        f"TP groups: 4 groups of 4 GPUs"
    )

    print(
        f"DP groups: 4 groups of 4 GPUs"
    )

    print(
        f"Total simulated time: "
        f"{completion_us:.3f} us"
    )

    print(
        f"Configured compute time: "
        f"{compute_total_us:.3f} us"
    )

    print(
        f"Communication + wait time: "
        f"{communication_and_wait_us:.3f} us"
    )

    print(
        f"Wrote results to {output_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())