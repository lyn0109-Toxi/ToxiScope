#!/bin/bash
# Software regression checks; this does not validate a toxicology model.
set -euo pipefail

cd "$(dirname "$0")/.."
TOXISCOPE_PYTHON="${PYTHON:-python3}"

"$TOXISCOPE_PYTHON" -c "import rdkit; print('RDKit version:', rdkit.__version__)"
"$TOXISCOPE_PYTHON" -m pytest tests "$@"

echo "Software regression checks passed. Scientific and regulatory validation require separate evidence."
