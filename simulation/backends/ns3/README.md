# Network backend: ns-3 (matrix standard)

**All Progress Report 2 cells — including baseline — use the ns-3 backend.**

Analytical remains available only for quick local debug (`BACKEND=analytical`), not for reported matrix numbers.

| Topology | Physical file | Logical dims | GPUs |
|---|---|---|---|
| Ring | `ring/physical_topology.txt` | `["16"]` | 16 |
| Rail | `rail/physical_topology.txt` | `["8","2"]` | 16 |
| Torus | `torus/physical_topology.txt` | `["4","2","2"]` | 16 |

Schedulers (baseline, NCCL, later TACCL / …) only change `system.json`. The ns-3 fabric stays fixed per topology.
