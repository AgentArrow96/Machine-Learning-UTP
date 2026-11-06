# Evals for the improvement study (defined before implementation; ECC eval-harness)

| ID | PASS criterion | Grader |
|---|---|---|
| M1 | `configs/improve.json` + `docs/improve.md` hashes recorded before the first improve run (timestamps) | audit |
| M2 | Frozen artifacts untouched: G1 hashes of the 140 original files equal `output/audit/g1_after.json`; the class-imbalance study's freeze file unchanged | audit |
| M3 | Every planned run validated, frozen, then tested once; freeze precedes every test evaluation | audit |
| M4 | Scoring parity: every stored improve prediction is a 256-px grid scored against the baseline masks; an independent sklearn re-scoring of the stored predictions equals the stored confusion counts (pass^2) | audit |
| M5 | Re-prediction: each U-Net/hybrid variant re-predicted from its checkpoint by code that does not import `floodnet` matches the stored predictions (0 differing pixels); RF variants re-predicted from cached features | audit |
| M6 | Reproduction: one retrained run per new training path (aux10 U-Net, recipe U-Net, rf_context RF) bitwise identical | repro |
| M7 | Combine rule applied mechanically from validation results only (recomputed independently, same components) | audit |
| M8 | Bootstrap and decisions re-implemented independently: 0 mismatches (pass^2) | audit |
| V | pytest green; ruff 0 on new files; node --check; verify_delivery + browser QA pass | verification loop |
