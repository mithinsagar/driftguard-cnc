#!/bin/sh
# Streaming label-delay study across protocols, scorers and stream orders (claim C2).
set -e
PY=.venv/bin/python
for proto in lomo time; do
  for scorer in B0 B1 B2 AE; do
    for scen in burst natural; do
      $PY scripts/run_stream.py --proto $proto --scorer $scorer --sweep --scenario $scen > /dev/null 2>&1
    done
  done
done
$PY scripts/run_stream.py --proto lomo --scorer B2 --sweep --scenario burst --label-rates 1.0 0.5 0.25 0.1 > /dev/null 2>&1
echo done
