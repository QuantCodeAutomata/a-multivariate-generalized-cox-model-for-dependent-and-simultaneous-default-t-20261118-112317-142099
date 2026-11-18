#!/usr/bin/env bash
# Runs the test suite and every experiment of this repository, in order.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Running tests ==="
python -m pytest -m "not market_data"

for n in 1 2 3 4 5 6 7 8 9; do
  echo "=== Running exp${n} ==="
  python -m "exp.exp${n}"
done

echo "=== Done. See results/RESULTS.md for the discussion of each result. ==="
