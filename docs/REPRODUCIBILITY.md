# Reproducibility

## Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib

## Single illustrative run (seed 6)

```bash
python social_program_governance_abm.py --scenario compare --seed 6 --out-dir outputs_seed6
```

## Thirty-seed run (Windows Command Prompt)

```bat
run_30_seeds.bat
```

## Manual merge command

```bash
python merge_seed_summaries.py --inputs-root outputs_governance_abm_30seeds --output outputs_governance_abm_30seeds/summary_master.csv
```
