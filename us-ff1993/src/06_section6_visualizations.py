"""
06_section6_visualizations.py
Section 6: Visualization Suite for FF(1993) Replication

Generates 6 publication-quality PNG figures matching the paper's key visual insights:
  Fig 1: Average Excess Returns Heatmap (Table 2)
  Fig 2: Factor Cumulative Returns Time Series (Section 2.1)
  Fig 3: R² Comparison Across Models (Tables 1, 3, 4, 5)
  Fig 4: Alpha (Intercept) Distribution (Table 5)
  Fig 5: Factor Loadings Heatmap (Table 5)
  Fig 6: SMB vs HML Factor Returns Scatter (Section 2.1)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_DPI = 150
FIG_SIZE = (10, 8)

# Publication-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Stock portfolio labels in 5x5 grid order (row=Size quintile, col=BE/ME quintile)
SIZE_LABELS = ["Small", "ME2", "ME3", "ME4", "Big"]
BM_LABELS = ["LoBM", "BM2", "BM3", "BM4", "HiBM"]

STOCK_PORTFOLIOS = [
    "SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM",
    "ME2 BM1", "ME2 BM2", "ME2 BM3", "ME2 BM4", "ME2 BM5",
    "ME3 BM1", "ME3 BM2", "ME3 BM3", "ME3 BM4", "ME3 BM5",
    "ME4 BM1", "ME4 BM2", "ME4 BM3", "ME4 BM4", "ME4 BM5",
    "BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM",
]

BOND_PORTFOLIOS = ["SHORT_TERM", "LONG_TERM", "AAA", "AA", "A", "BBB", "LOW_GRADE"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(filename, comment="#"):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Skipping dependent figures.")
        return None
    return pd.read_csv(path, comment=comment)


# ---------------------------------------------------------------------------
# Figure 1: Average Excess Returns Heatmap (Table 2)
# ---------------------------------------------------------------------------

def make_fig1():
    """5x5 heatmap of stock portfolio average excess returns."""
    df = load_csv("table2_summary.csv")
    if df is None:
        return

    stock_df = df[df["Type"] == "Stock"].copy()

    # Build 5x5 matrix: rows = Size quintiles (Small→Big), cols = BE/ME quintiles
    returns_matrix = np.full((5, 5), np.nan)
    for i, size_label in enumerate(SIZE_LABELS):
        for j, bm_label in enumerate(BM_LABELS):
            label = STOCK_PORTFOLIOS[i * 5 + j]
            row = stock_df[stock_df["Label"] == label]
            if len(row) > 0:
                returns_matrix[i, j] = row["Mean"].values[0]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    im = ax.imshow(returns_matrix, cmap="RdYlGn", aspect="auto")

    # Annotate cells with values
    for i in range(5):
        for j in range(5):
            val = returns_matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val - returns_matrix[~np.isnan(returns_matrix)].mean()) > 0.25 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(5))
    ax.set_xticklabels(BM_LABELS)
    ax.set_yticks(range(5))
    ax.set_yticklabels(SIZE_LABELS)
    ax.set_xlabel("BE/ME Quintile")
    ax.set_ylabel("Size Quintile")
    ax.set_title("Figure 1: Average Monthly Excess Returns (%)\nStock Portfolios (1963–1991)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Avg Excess Return (%)")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_average_returns_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig1: Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Factor Cumulative Returns Time Series
# ---------------------------------------------------------------------------

def make_fig2():
    """5 subplots showing cumulative returns of each factor."""
    df = load_csv("factors.csv")
    if df is None:
        return

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    df = df.sort_values("Date").reset_index(drop=True)

    factors = ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes_flat = axes.flatten()

    for idx, factor in enumerate(factors):
        ax = axes_flat[idx]
        series = df[factor].dropna()
        dates = df.loc[series.index, "Date"]

        # Cumulative return: product of (1 + r/100)
        cum = (1 + series / 100).cumprod() * 100 - 100
        ax.plot(dates, cum, linewidth=1.2, color="#2c3e50")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(factor, fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplot
    axes_flat[5].set_visible(False)

    fig.suptitle("Figure 2: Cumulative Factor Returns (1963–1991)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_factor_cumulative_returns.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig2: Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: R² Comparison Across Models
# ---------------------------------------------------------------------------

def make_fig3():
    """Grouped bar chart: avg R² for stocks vs bonds across 4 models."""
    t1 = load_csv("table1_market.csv")
    t3 = load_csv("table3_bond.csv")
    t4 = load_csv("table4_stock3f.csv")
    t5 = load_csv("table5_five_factor.csv")
    if any(x is None for x in [t1, t3, t4, t5]):
        return

    # Compute average R² per model per type
    def avg_r2(df, type_val, r2_col="r_squared"):
        subset = df[df["type"] == type_val]
        return subset[r2_col].mean() if len(subset) > 0 else 0.0

    models = ["1-Factor\n(Mkt-RF)", "2-Factor\nBond", "3-Factor\nStock", "5-Factor\nJoint"]
    stock_r2 = [
        avg_r2(t1, "stock"),
        avg_r2(t3, "stock"),
        avg_r2(t4, "stock"),
        avg_r2(t5, "stock"),
    ]
    bond_r2 = [
        avg_r2(t1, "bond"),
        avg_r2(t3, "bond"),
        avg_r2(t4, "bond"),
        avg_r2(t5, "bond"),
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars_s = ax.bar(x - width / 2, stock_r2, width, label="Stock Portfolios", color="#2980b9", edgecolor="white")
    bars_b = ax.bar(x + width / 2, bond_r2, width, label="Bond Portfolios", color="#e67e22", edgecolor="white")

    # Value labels on bars
    for bar in bars_s:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Average R²")
    ax.set_ylim(0, 1.1)
    ax.set_title("Figure 3: Average R² Across Model Specifications\nStock vs Bond Portfolios")
    ax.legend(loc="upper left")
    ax.axhline(1.0, color="grey", linewidth=0.3, linestyle=":")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_r2_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig3: Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Alpha (Intercept) Distribution
# ---------------------------------------------------------------------------

def make_fig4():
    """Histogram of alphas from 5-factor model, stock vs bond."""
    df = load_csv("table5_five_factor.csv")
    if df is None:
        return

    stock_alpha = df[df["type"] == "stock"]["alpha"]
    bond_alpha = df[df["type"] == "bond"]["alpha"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Determine common bin range
    all_alpha = pd.concat([stock_alpha, bond_alpha])
    lo, hi = all_alpha.min(), all_alpha.max()
    bins = np.linspace(lo - 0.1, hi + 0.1, 25)

    ax.hist(stock_alpha, bins=bins, alpha=0.7, color="#2980b9", label="Stock Portfolios", edgecolor="white")
    ax.hist(bond_alpha, bins=bins, alpha=0.7, color="#e67e22", label="Bond Portfolios", edgecolor="white")
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Zero")

    ax.set_xlabel("Alpha (Intercept, %/month)")
    ax.set_ylabel("Count")
    ax.set_title("Figure 4: Distribution of Alphas — Five-Factor Model\nStock vs Bond Portfolios")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_alpha_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig4: Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Factor Loadings Heatmap (Table 5)
# ---------------------------------------------------------------------------

def make_fig5():
    """Heatmap of beta coefficients for 32 portfolios x 5 factors."""
    df = load_csv("table5_five_factor.csv")
    if df is None:
        return

    factor_cols = ["beta_Mkt-RF", "beta_SMB", "beta_HML", "beta_TERM", "beta_DEF"]
    factor_names = ["Mkt-RF", "SMB", "HML", "TERM", "DEF"]

    # Order: stocks first (5x5 grid order), then bonds
    ordered = []
    for p in STOCK_PORTFOLIOS:
        row = df[df["portfolio"] == p]
        if len(row) > 0:
            ordered.append(row)
    for p in BOND_PORTFOLIOS:
        row = df[df["portfolio"] == p]
        if len(row) > 0:
            ordered.append(row)

    if not ordered:
        print("WARNING: No portfolio data found for fig5")
        return

    ordered_df = pd.concat(ordered, ignore_index=True)
    matrix = ordered_df[factor_cols].values
    labels = ordered_df["portfolio"].tolist()
    types = ordered_df["type"].tolist()

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(matrix, cmap="RdBu", aspect="auto", vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(factor_names)))
    ax.set_xticklabels(factor_names)
    ax.set_yticks(range(len(labels)))

    # Color y-tick labels by type
    ytick_colors = ["#2980b9" if t == "stock" else "#e67e22" for t in types]
    ax.set_yticklabels(labels, fontsize=8)
    for ticklabel, color in zip(ax.get_yticklabels(), ytick_colors):
        ticklabel.set_color(color)

    # Add separator line between stocks and bonds
    n_stocks = sum(1 for t in types if t == "stock")
    if 0 < n_stocks < len(types):
        ax.axhline(n_stocks - 0.5, color="black", linewidth=1.5)

    ax.set_xlabel("Factor")
    ax.set_title("Figure 5: Factor Loadings (Betas) — Five-Factor Model\n32 Portfolios × 5 Factors")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Beta Coefficient")

    # Legend for type colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2980b9", label="Stock Portfolios"),
        Patch(facecolor="#e67e22", label="Bond Portfolios"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_factor_loadings_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig5: Saved {path}")


# ---------------------------------------------------------------------------
# Figure 6: SMB vs HML Factor Returns Scatter
# ---------------------------------------------------------------------------

def make_fig6():
    """Scatter plot of SMB vs HML monthly returns, colored by time."""
    df = load_csv("factors.csv")
    if df is None:
        return

    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    df = df.sort_values("Date").reset_index(drop=True)

    smb = df["SMB"].dropna()
    hml = df["HML"].dropna()
    common_idx = smb.index.intersection(hml.index)
    smb = smb.loc[common_idx]
    hml = hml.loc[common_idx]
    dates = df.loc[common_idx, "Date"]

    # Time-based color: early=light, late=dark
    time_norm = (dates - dates.min()) / (dates.max() - dates.min())
    time_vals = time_norm.values.astype(float)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    scatter = ax.scatter(smb, hml, c=time_vals, cmap="viridis", alpha=0.6, s=20, edgecolors="none")

    # Regression line
    mask = ~(smb.isna() | hml.isna())
    if mask.sum() > 2:
        coeffs = np.polyfit(smb[mask], hml[mask], 1)
        x_line = np.linspace(smb.min(), smb.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, color="red", linewidth=1.5, linestyle="--",
                label=f"Fit: slope={coeffs[0]:.3f}")

    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("SMB Return (%/month)")
    ax.set_ylabel("HML Return (%/month)")
    ax.set_title("Figure 6: SMB vs HML Monthly Returns\nColor: Early (light) → Late (dark)")
    ax.legend(loc="upper left")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Time Progression")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["1963", "1977", "1991"])

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig6_smb_hml_scatter.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"fig6: Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Section 6: Visualization Suite - FF(1993) Replication")
    print("=" * 60)

    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()

    print("\nAll 6 figures generated successfully.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
