#!/bin/bash
# Korean FF3 v2 Full Pipeline
# Run all analysis steps sequentially

set -e

echo "=========================================="
echo "Korean FF3 v2 Analysis Pipeline"
echo "=========================================="

# Note: Data preparation steps (Tasks 0-3) are assumed done
# This script runs from Task 5 onwards

cd "$(dirname "$0")"

echo ""
echo "[1/8] Calculating stock returns + filtering..."
python scripts/calc_stock_returns.py

echo ""
echo "[2/8] BE carry-forward..."
python scripts/be_carry_forward.py

echo ""
echo "[3/8] Building panel data..."
python scripts/build_panel.py

echo ""
echo "[4/8] Calculating Mkt-RF..."
python scripts/calc_mkt_rf.py

echo ""
echo "[5/8] Constructing 6-portfolio factors..."
python scripts/construct_6_portfolios.py

echo ""
echo "[6/8] Constructing 25-portfolio returns..."
python scripts/construct_25_portfolios.py

echo ""
echo "[7/8] Running regressions + GRS test..."
python scripts/run_regression_korea.py

echo ""
echo "[8/8] Generating comparison tables and charts..."
python scripts/make_comparison_tables.py
python scripts/make_charts.py
python scripts/make_heatmap.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Outputs in: output/"
echo "Charts in: output/charts/"
echo "=========================================="
