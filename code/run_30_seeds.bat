@echo off
setlocal

set SCRIPT=social_program_governance_abm.py
set OUTROOT=outputs_governance_abm_30seeds

for /L %%S in (1,1,30) do (
    python %SCRIPT% --scenario compare --seed %%S --out-dir %OUTROOT%\seed_%%S
)

python merge_seed_summaries.py --inputs-root %OUTROOT% --output %OUTROOT%\summary_master.csv

endlocal
