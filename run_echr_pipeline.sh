#!/usr/bin/env bash
# =============================================================================
# run_echr_pipeline.sh
# Run the full ECHR pipeline (steps 00–06) from the computations/ directory.
# =============================================================================
# Usage:
#   chmod +x run_echr_pipeline.sh
#   ./run_echr_pipeline.sh
#
# Prerequisites:
#   - data/raw/echr/network_from_cache.json must exist
#   - The virtual environment in env/ must be set up (run run_all.sh once, or
#     manually: python3 -m venv env && source env/bin/activate && pip install -r src/requirements.txt)
# =============================================================================

set -e

# Keep Mac awake
caffeinate -dims &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null; echo 'caffeinate stopped.'" EXIT

# Activate virtual environment
source env/bin/activate

echo "==> Step 0: Converting network_from_cache.json to pipeline inputs ..."
python src/00_convert_cache_to_pipeline_inputs.py

echo ""
echo "==> Step 2: Building directed citation F-hypergraph G (ECHR) ..."
python src/02_build_citation_hypergraph.py --dataset echr

echo ""
echo "==> Step 3: Structural profiling on G (ECHR) ..."
python src/03_structural_profiling.py --dataset echr

echo ""
echo "==> Step 4: Legal Argument Space — embedding + clustering (ECHR) ..."
python src/04_legal_argument_space.py --dataset echr

echo ""
echo "==> Step 5: Empirical analysis (ECHR) ..."
python src/05_empirical_analysis.py --dataset echr

echo ""
echo "==> Step 6: Missing Precedent validation (ECHR) ..."
python src/06_validation.py --dataset echr

echo ""
echo "==> All done. Results are in results/echr/"
