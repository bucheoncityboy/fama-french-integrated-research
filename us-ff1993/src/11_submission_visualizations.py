"""Generate assignment-facing visualizations from appendix_output."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


OUT_DIR = Path(config.OUTPUT_DIR)
OUT_DIR.mkdir(exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / path, index_col=0)


def fig1_stock_mean_heatmap() -> None:
    df = load_table("table2_panel2_stock_mean_std.csv")
    means = df.map(lambda x: float(str(x).split("/")[0].strip()))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(means.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(means.columns)), means.columns)
    ax.set_yticks(range(len(means.index)), means.index)
    ax.set_title("Figure 1. Mean excess returns of 25 stock portfolios")
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax.text(j, i, f"{means.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean excess return (%/month)")
    save(fig, "submission_fig1_stock_mean_heatmap.png")


def fig2_factor_bar() -> None:
    df = pd.read_csv(OUT_DIR / "table2_panel1_factor_summary.csv")
    factors = df[df["variable"].isin(["RM-RF", "SMB", "HML", "TERM", "DEF"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(factors["variable"], factors["mean"], color=["#4C78A8", "#72B7B2", "#54A24B", "#E45756", "#F58518"])
    ax.set_title("Figure 2. Mean factor premiums")
    ax.set_ylabel("Mean (%/month)")
    for i, v in enumerate(factors["mean"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, "submission_fig2_factor_premiums.png")


def _avg_r2(path: str) -> float:
    df = load_table(path)
    values = [float(str(v).split("/")[0].strip()) for v in df.to_numpy().flatten()]
    return float(np.mean(values))


def fig3_model_r2() -> None:
    labels = ["1-factor", "SMB+HML", "3-factor"]
    values = [
        _avg_r2("table4_panel2_r2_se.csv"),
        _avg_r2("table5_panel3_r2_se.csv"),
        _avg_r2("table6_panel4_r2_se.csv"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average R²")
    ax.set_title("Figure 3. Average explanatory power across stock models")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    save(fig, "submission_fig3_model_r2.png")


def fig4_ep_dp_r2() -> None:
    df = pd.read_csv(OUT_DIR / "table11_ep_dp_long.csv")
    r2 = df[df["stat_type"] == "r_squared"].copy()
    pivot = r2.pivot_table(index=["type", "portfolio"], columns="model", values="value")
    pivot = pivot.reset_index()
    labels = [f"{t}-{p}" for t, p in zip(pivot["type"], pivot["portfolio"])]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, pivot["capm"], width, label="CAPM")
    ax.bar(x + width / 2, pivot["ff3f"], width, label="FF3F")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Figure 4. FF3F improvement over CAPM for E/P and D/P portfolios")
    ax.legend()
    save(fig, "submission_fig4_ep_dp_r2.png")


def fig5_alpha_and_joint_test() -> None:
    alpha_df = pd.read_csv(OUT_DIR / "table9a_stock_alphas.csv")
    joint_df = pd.read_csv(OUT_DIR / "table9c_joint_tests.csv")

    def mean_abs_alpha(col: str) -> float:
        values = []
        for value in alpha_df[col]:
            alpha = float(str(value).split("/")[0].strip())
            values.append(abs(alpha))
        return float(np.mean(values))

    alpha_models = ["(ii)", "(iii)", "(iv)"]
    alpha_values = [mean_abs_alpha(col) for col in alpha_models]
    f_values = [
        float(joint_df.loc[joint_df["model"] == "(ii)", "F_stat"].iloc[0]),
        float(joint_df.loc[joint_df["model"] == "(iii)", "F_stat"].iloc[0]),
        float(joint_df.loc[joint_df["model"] == "(iv)", "F_stat"].iloc[0]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["CAPM", "SMB+HML", "FF3F"]
    axes[0].bar(labels, alpha_values, color="#4C78A8")
    axes[0].set_title("Figure 5A. Mean absolute alpha by model")
    axes[0].set_ylabel("Mean |alpha|")
    for i, v in enumerate(alpha_values):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, f_values, color="#E45756")
    axes[1].set_title("Figure 5B. Joint alpha F-statistics")
    axes[1].set_ylabel("F-statistic")
    for i, v in enumerate(f_values):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    save(fig, "submission_fig5_alpha_tests.png")


def main() -> None:
    fig1_stock_mean_heatmap()
    fig2_factor_bar()
    fig3_model_r2()
    fig4_ep_dp_r2()
    fig5_alpha_and_joint_test()
    print(f"Submission visualizations written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
