# Scheduler: baseline (Astra-SIM defaults)

Default Astra-SIM `system.json` policies (typically LIFO scheduling + native collective implementations).

**Network backend:** Progress Report 2 matrix numbers — including this baseline column — use **ns-3** (`BACKEND=ns3`). Analytical (`BACKEND=analytical`) is debug-only and must not be mixed into reported comparisons against NCCL / TACCL / etc.

Only `system.json` differs across scheduler columns; Ring / Rail / Torus physical + logical ns-3 fabrics stay fixed under `backends/ns3/`.
