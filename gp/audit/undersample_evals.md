# Evals for the uniform and undersampling pipelines (defined before implementation)

| ID | PASS criterion | Grader |
|---|---|---|
| U1 | `configs/undersample.json`, `docs/undersample.md` and this file are hashed before the first run (timestamps) | audit |
| U2 | Frozen artifacts untouched: G1 (140 files) equal `output/audit/g1_after.json`; `web/data/results.json` and `web/data/after.json` hashes unchanged | audit |
| U3 | Every planned run validated, frozen, then tested once; only the chosen ratio's runs are tested; freeze precedes every test evaluation | audit |
| U4 | Rules hold in the saved artifacts: uniform RF uses every training region and the U-Net's class weights; ratio RF keeps every target region and round(r x #target) "Other" regions; ratio U-Net keep probability = min(1, r x target / other pixels); blackened training photos are black exactly on "Other" | audit |
| U5 | Independent re-scoring of every tested run's stored predictions with sklearn equals the stored confusion counts (pass^2) | audit |
| U6 | Ratio choice recomputed from the validation results equals the stored choice | audit |
| U7 | Bootstrap and decisions re-implemented independently: 0 mismatches (pass^2) | audit |
| V | pytest green; ruff 0 on new files; node --check; verify_delivery + browser QA pass | verification loop |
