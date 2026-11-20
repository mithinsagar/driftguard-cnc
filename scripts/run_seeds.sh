#!/bin/sh
# Extra training seeds for every learned scorer (B0 is deterministic and is run once).
PY=.venv/bin/python
for s in 1 2; do $PY scripts/run_eval.py --proto lopo lomo time --scorers B1 B2 --seed $s || exit 1; done
for s in 1 2; do $PY scripts/run_eval.py --proto time lomo lopo --scorers AE --seed $s || exit 1; done
echo done
