"""
tests/test_section6_visualizations.py
RED -> GREEN tests for 06_section6_visualizations.py (Section 6: Visualization Suite)

Verifies that all 6 PNG figures are generated with correct dimensions and
non-zero file size, matching the FF(1993) replication visual insights.
"""

import os
import subprocess
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(BASE_DIR, "src", "06_section6_visualizations.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "appendix_output")

FIGURE_FILES = {
    "fig1": "fig1_average_returns_heatmap.png",
    "fig2": "fig2_factor_cumulative_returns.png",
    "fig3": "fig3_r2_comparison.png",
    "fig4": "fig4_alpha_distribution.png",
    "fig5": "fig5_factor_loadings_heatmap.png",
    "fig6": "fig6_smb_hml_scatter.png",
}

MIN_WIDTH = 800
MIN_HEIGHT = 600


@pytest.fixture(scope="module")
def run_script():
    """Run the Section 6 visualization script once per module."""
    assert os.path.exists(SCRIPT_PATH), f"Script not found: {SCRIPT_PATH}"
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )
    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result


# ---------------------------------------------------------------------------
# Output existence and file size
# ---------------------------------------------------------------------------


class TestFigureExistence:
    """Verify all 6 PNG files exist with non-zero size."""

    @pytest.mark.parametrize("fig_key", list(FIGURE_FILES.keys()))
    def test_figure_file_exists(self, run_script, fig_key):
        path = os.path.join(OUTPUT_DIR, FIGURE_FILES[fig_key])
        assert os.path.exists(path), f"Missing figure: {FIGURE_FILES[fig_key]}"

    @pytest.mark.parametrize("fig_key", list(FIGURE_FILES.keys()))
    def test_figure_nonzero_size(self, run_script, fig_key):
        path = os.path.join(OUTPUT_DIR, FIGURE_FILES[fig_key])
        assert os.path.getsize(path) > 0, f"Empty file: {FIGURE_FILES[fig_key]}"

    @pytest.mark.parametrize("fig_key", list(FIGURE_FILES.keys()))
    def test_figure_minimum_dimensions(self, run_script, fig_key):
        """Each PNG must be at least 800x600 pixels."""
        path = os.path.join(OUTPUT_DIR, FIGURE_FILES[fig_key])
        try:
            from PIL import Image
            img = Image.open(path)
            w, h = img.size
            assert w >= MIN_WIDTH, (
                f"{FIGURE_FILES[fig_key]} width {w} < {MIN_WIDTH}"
            )
            assert h >= MIN_HEIGHT, (
                f"{FIGURE_FILES[fig_key]} height {h} < {MIN_HEIGHT}"
            )
        except ImportError:
            pytest.skip("Pillow not installed; skipping dimension check")


# ---------------------------------------------------------------------------
# Script output verification
# ---------------------------------------------------------------------------


class TestScriptOutput:
    """Verify the script prints confirmation for each figure."""

    def test_script_prints_figure_generation(self, run_script):
        stdout = run_script.stdout
        for fig_key, fname in FIGURE_FILES.items():
            assert fname in stdout or fig_key in stdout.lower(), (
                f"Script output should mention {fname}"
            )
