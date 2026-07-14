#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Full pipeline for legal_hypergraphs_jcllt
# =============================================================================
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Run this from the root of the computations/ directory.
# It will:
#   1. Keep your Mac awake (caffeinate)
#   2. Create and activate a virtual environment
#   3. Install all dependencies
#   4. Run the full 6-step pipeline for both SCOTUS and ECHR
# =============================================================================

set -e  # exit immediately on any error

# ---------------------------------------------------------------------------
# 0. Keep Mac awake
# ---------------------------------------------------------------------------
echo "==> Keeping Mac awake with caffeinate ..."
caffeinate -dims &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null; echo 'caffeinate stopped.'" EXIT

# ---------------------------------------------------------------------------
# 1. Virtual environment
# ---------------------------------------------------------------------------
if [ ! -d "env" ]; then
    echo "==> Creating virtual environment ..."
    python3 -m venv env
fi

echo "==> Activating virtual environment ..."
source env/bin/activate

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo "==> Installing dependencies ..."
pip install --upgrade pip -q
# Install everything except umap-learn first
pip install numpy pandas networkx requests beautifulsoup4 scipy scikit-learn matplotlib seaborn tqdm -q
# Install umap-learn with NO deps to avoid numba/llvmlite build failures
pip install umap-learn --no-deps -q

# ---------------------------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------------------------
echo ""
echo "==> Step 1: Fetching data (SCOTUS bulk S3 + ECHR HUDOC) ..."
python src/01_fetch_data.py --dataset both

echo ""
echo "==> Step 2: Building directed citation F-hypergraph G ..."
python src/02_build_citation_hypergraph.py --dataset both

echo ""
echo "==> Step 3: Structural profiling on G ..."
python src/03_structural_profiling.py --dataset both

echo ""
echo "==> Step 4: Legal Argument Space (embedding + clustering) ..."
python src/04_legal_argument_space.py --dataset both

echo ""
echo "==> Step 5: Empirical analysis ..."
python src/05_empirical_analysis.py --dataset both

echo ""
echo "==> Step 6: Missing Precedent validation ..."
python src/06_validation.py --dataset both

echo ""
echo "==> All done. Results are in results/scotus/ and results/echr/"
