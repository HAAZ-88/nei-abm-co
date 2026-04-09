# Social Program Governance ABM

This repository contains the agent-based model used to compare hierarchical,
delegated, and adaptive governance arrangements in social program access.

The model is designed as an analytical illustration for a paper on new
institutional economics, complexity, and governance. It focuses on how
alternative governance arrangements perform under heterogeneous territorial
conditions, learning, congestion, screening frictions, and capture risks.

## Main file

- `social_program_governance_abm.py`: main simulation script.

## Included outputs

- `summary.csv`: scenario summary for the illustrative run (seed 6).
- `timeseries.csv`: period-by-period results for the illustrative run.
- `comparative_trajectories.png`: comparative trajectories for seed 6.
- `governance_history.csv`: governance switches in the adaptive scenario.
- `offices_adaptive.csv`: office-level final outcomes in the adaptive scenario.
- `summary_master.csv`: merged summary for 30 seeds.

## Quick start

Run a single comparative simulation:

```bash
python social_program_governance_abm.py --scenario compare --seed 6 --out-dir outputs_seed6
```

Run 30 seeds on Windows Command Prompt:

```bat
run_30_seeds.bat
```

## Reproducibility

See `REPRODUCIBILITY.md` for exact commands and `MODEL_DESCRIPTION.md` for a
structured description of the model.

## Citation

See `CITATION.cff` for repository citation metadata.
