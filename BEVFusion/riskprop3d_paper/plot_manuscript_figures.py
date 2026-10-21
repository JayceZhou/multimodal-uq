#!/usr/bin/env python3
"""Generate the frozen RiskProp3D manuscript figures.

The script reads retained JSON evidence, performs only sign-orientation and
layout transformations documented in the manuscript, writes panel-level CSV
source data, and exports SVG/PDF/TIFF/PNG bundles with editable vector text.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PAPER_DIR.parent / "outputs"
FIGURE_DIR = PAPER_DIR / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"
MPL_DIR = FIGURE_DIR / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyBboxPatch, Patch, Rectangle


# Mandatory editable-text and publication export settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arimo", "Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7.8,
        "axes.titlesize": 8.6,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

COLORS = {
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "blue_light": "#DCE9F5",
    "teal": "#42949E",
    "teal_light": "#DCEEEF",
    "violet": "#7C6CCF",
    "violet_light": "#E8E3F6",
    "orange": "#D9772A",
    "orange_light": "#F7E4D1",
    "gray_dark": "#565656",
    "gray": "#767676",
    "gray_mid": "#A8A8A8",
    "gray_light": "#E5E5E5",
    "gray_pale": "#F5F5F5",
    "black": "#272727",
    "white": "#FFFFFF",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


R0 = load_json(OUTPUTS_DIR / "riskprop3d_r0" / "result.json")
R1 = load_json(OUTPUTS_DIR / "riskprop3d_r1" / "result.json")
R2C = load_json(OUTPUTS_DIR / "riskprop3d_r2_core" / "result.json")
R2D = load_json(OUTPUTS_DIR / "riskprop3d_r2d" / "result.json")
R3P = load_json(OUTPUTS_DIR / "riskprop3d_r3p" / "result.json")
EXTERNAL = load_json(
    OUTPUTS_DIR / "riskprop3d_final_confirmation" / "external" / "result.json"
)["results"]
OFFICIAL = load_json(
    OUTPUTS_DIR / "riskprop3d_final_confirmation" / "official" / "result.json"
)["results"]


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-10) -> None:
    if abs(float(actual) - expected) > tol:
        raise ValueError(f"Frozen anchor mismatch for {label}: {actual} != {expected}")


def validate_frozen_anchors() -> None:
    assert_close(
        R0["clean_proper_scores"]["CV"]["energy"]["P2"]["mean"],
        0.016142558202168197,
        "R0 clean CV HTC-over-Gaussian energy",
    )
    assert_close(
        R1["distribution_encoding"]["energy"]["mean"],
        0.01965378189836927,
        "R1 initial DSE energy",
    )
    assert_close(
        R2D["efficiency"]["speedup_mean"],
        38.55141891371385,
        "R2-D speedup",
    )
    assert_close(
        OFFICIAL["r0"]["CV"]["P2"]["metrics"]["energy"]["mean"],
        0.0206040110612517,
        "official CV HTC-over-Gaussian energy",
    )
    assert_close(
        OFFICIAL["r3p"]["P0_mean_only"]["metrics"]["energy"]["mean"],
        0.027020911415362055,
        "official direct-token over mean energy",
    )


def effect(entry: dict, flip: bool = False) -> tuple[float, float, float]:
    mean = float(entry["mean"])
    low, high = map(float, entry["ci95"])
    if flip:
        return -mean, -high, -low
    return mean, low, high


def classify(mean: float, low: float, high: float) -> str:
    if low > 0:
        return "significant_positive"
    if high < 0:
        return "significant_negative"
    return "nonsignificant_positive" if mean >= 0 else "nonsignificant_negative"


def seed_reversal(mean: float, seeds: list[float] | None) -> bool:
    if not seeds:
        return False
    if mean >= 0:
        return any(float(value) < 0 for value in seeds)
    return any(float(value) > 0 for value in seeds)


def write_source_csv(name: str, rows: list[dict]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    preferred = [
        "panel",
        "population",
        "condition",
        "probe",
        "contrast",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_scenes",
        "seed_1",
        "seed_2",
        "seed_3",
        "area_ratio",
        "status",
        "seed_reversal",
        "source_stage",
        "note",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fields = [key for key in preferred if any(key in row for row in rows)] + extra
    with (SOURCE_DIR / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_bundle(fig: mpl.figure.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve the declared publication canvas.  bbox_inches="tight" changes
    # the physical width according to label extent and can silently make a
    # nominal 7.16-inch figure too wide or too narrow at submission time.
    kwargs = {"bbox_inches": None, "facecolor": "white"}
    fig.savefig(FIGURE_DIR / f"{stem}.svg", **kwargs)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", **kwargs)
    fig.savefig(
        FIGURE_DIR / f"{stem}.tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **kwargs,
    )
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=240, **kwargs)
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.11, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["black"],
    )


def style_effect_axis(ax: mpl.axes.Axes, xlabel: str, xlim: tuple[float, float]) -> None:
    ax.axvline(0, color=COLORS["gray_mid"], lw=0.9, ls="--", zorder=0)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="both", length=2.5, width=0.7)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def errorbar_h(
    ax: mpl.axes.Axes,
    y: np.ndarray | list[float],
    means: np.ndarray | list[float],
    lows: np.ndarray | list[float],
    highs: np.ndarray | list[float],
    color: str,
    marker: str,
    label: str | None = None,
    ms: float = 4.8,
    zorder: int = 3,
) -> None:
    y_arr = np.asarray(y, dtype=float)
    mean_arr = np.asarray(means, dtype=float)
    low_arr = np.asarray(lows, dtype=float)
    high_arr = np.asarray(highs, dtype=float)
    ax.errorbar(
        mean_arr,
        y_arr,
        xerr=np.vstack([mean_arr - low_arr, high_arr - mean_arr]),
        fmt=marker,
        color=color,
        ecolor=color,
        elinewidth=1.1,
        capsize=2.2,
        capthick=0.9,
        markersize=ms,
        markeredgecolor="white",
        markeredgewidth=0.55,
        label=label,
        zorder=zorder,
    )


def rounded_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 7.0,
    lw: float = 1.0,
    linestyle: str = "-",
    text_color: str | None = None,
    radius: float = 0.02,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color or COLORS["black"],
        linespacing=1.15,
    )
    return patch


def arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.0,
            "color": color,
            "mutation_scale": 9,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
    )


def make_figure_1() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 3.85))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.97,
        "Three non-equivalent perception-to-forecast interfaces",
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.035,
        0.917,
        "One frozen probabilistic 3D perception output; three distinct scientific questions",
        fontsize=7.2,
        color=COLORS["gray"],
        ha="left",
        va="top",
    )

    identity = FancyBboxPatch(
        (0.735, 0.895),
        0.23,
        0.047,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        facecolor=COLORS["gray_pale"],
        edgecolor=COLORS["gray_light"],
        linewidth=0.7,
    )
    ax.add_patch(identity)
    ax.text(
        0.85,
        0.918,
        "Conditioned on correct object identity",
        fontsize=6.8,
        color=COLORS["gray_dark"],
        ha="center",
        va="center",
    )
    ax.text(
        0.85,
        0.88,
        "association and track management outside scope",
        fontsize=6.1,
        color=COLORS["gray"],
        ha="center",
        va="top",
    )

    # Compact common upstream interface.
    upstream = FancyBboxPatch(
        (0.035, 0.685),
        0.93,
        0.155,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="#F8F9FB",
        edgecolor=COLORS["gray_light"],
        linewidth=0.8,
    )
    ax.add_patch(upstream)
    ax.text(0.052, 0.818, "FROZEN UPSTREAM", fontsize=6.3, color=COLORS["gray"], fontweight="bold", ha="left")
    ax.text(0.052, 0.792, "SENSORS", fontsize=6.0, color=COLORS["gray_mid"], ha="left")
    rounded_box(ax, (0.052, 0.712), 0.072, 0.055, "Camera", COLORS["white"], COLORS["gray_light"], 6.5, lw=0.7, radius=0.012)
    rounded_box(ax, (0.137, 0.712), 0.072, 0.055, "LiDAR", COLORS["white"], COLORS["gray_light"], 6.5, lw=0.7, radius=0.012)
    ax.text(0.1305, 0.7395, "+", fontsize=6.2, color=COLORS["gray"], fontweight="bold", ha="center", va="center", zorder=5)

    ax.text(0.258, 0.792, "PERCEPTION", fontsize=6.0, color=COLORS["gray_mid"], ha="left")
    rounded_box(ax, (0.252, 0.705), 0.15, 0.072, "Frozen 3D detector", COLORS["white"], COLORS["gray_mid"], 6.6, lw=0.8, radius=0.012)
    arrow(ax, (0.209, 0.74), (0.252, 0.741), COLORS["gray"])

    ax.text(0.472, 0.792, "OBJECT-STATE OUTPUT", fontsize=6.0, color=COLORS["gray_mid"], ha="left")
    output = FancyBboxPatch(
        (0.466, 0.705),
        0.405,
        0.072,
        boxstyle="round,pad=0.009,rounding_size=0.012",
        facecolor=COLORS["white"],
        edgecolor=COLORS["blue"],
        linewidth=0.9,
    )
    ax.add_patch(output)
    ax.plot([0.657, 0.657], [0.711, 0.771], color=COLORS["gray_light"], lw=0.7)
    ax.text(0.56, 0.749, "Shared state mean  μ", fontsize=6.5, ha="center", va="center")
    ax.text(0.56, 0.724, "stop-gradient", fontsize=6.1, color=COLORS["gray"], ha="center", va="center")
    ax.text(0.764, 0.749, "Trained laws  G · HTC", fontsize=6.5, ha="center", va="center", color=COLORS["blue"])
    ax.text(0.764, 0.724, "complete predictive laws", fontsize=6.1, color=COLORS["gray"], ha="center", va="center")
    arrow(ax, (0.402, 0.741), (0.466, 0.741), COLORS["gray"])

    # A shared bus makes the common upstream explicit without implying that
    # evidence at one downstream level establishes another.
    card_x = [0.035, 0.3525, 0.67]
    card_w = 0.295
    card_centers = [x + card_w / 2 for x in card_x]
    ax.plot([0.668, 0.668], [0.705, 0.635], color=COLORS["gray_mid"], lw=0.8)
    ax.plot([card_centers[0], card_centers[-1]], [0.635, 0.635], color=COLORS["gray_mid"], lw=0.8)
    for center in card_centers:
        arrow(ax, (center, 0.635), (center, 0.595), COLORS["gray_mid"])

    def taxonomy_card(
        x: float,
        numeral: str,
        title: str,
        edge: str,
        pale: str,
        steps: list[str],
        question: str,
    ) -> None:
        y0, height = 0.095, 0.49
        shadow = FancyBboxPatch(
            (x + 0.004, y0 - 0.006),
            card_w,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#ECEFF3",
            edgecolor="none",
            alpha=0.7,
            zorder=0,
        )
        ax.add_patch(shadow)
        card = FancyBboxPatch(
            (x, y0),
            card_w,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=COLORS["white"],
            edgecolor=COLORS["gray_light"],
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(card)
        ax.plot([x + 0.016, x + card_w - 0.016], [0.571, 0.571], color=edge, lw=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([x + 0.041], [0.529], s=94, color=edge, edgecolor=COLORS["white"], linewidth=0.8, zorder=4)
        ax.text(x + 0.041, 0.529, numeral, color=COLORS["white"], fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(x + 0.071, 0.529, title, fontsize=7.2, fontweight="bold", ha="left", va="center", zorder=5)

        step_y = [0.415, 0.325, 0.235]
        for idx, (y_step, text_step) in enumerate(zip(step_y, steps)):
            face = pale if idx in (0, 2) else COLORS["gray_pale"]
            step_edge = edge if idx in (0, 2) else COLORS["gray_light"]
            rounded_box(
                ax,
                (x + 0.033, y_step),
                card_w - 0.066,
                0.057,
                text_step,
                face,
                step_edge,
                6.4,
                lw=0.65,
                radius=0.01,
            )
            if idx < 2:
                arrow(ax, (x + card_w / 2, y_step), (x + card_w / 2, step_y[idx + 1] + 0.057), edge)

        question_box = FancyBboxPatch(
            (x + 0.024, 0.118),
            card_w - 0.048,
            0.071,
            boxstyle="round,pad=0.007,rounding_size=0.012",
            facecolor=pale,
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(question_box)
        ax.text(
            x + card_w / 2,
            0.153,
            question,
            fontsize=6.2,
            color=edge,
            fontweight="bold",
            ha="center",
            va="center",
            linespacing=1.1,
            zorder=3,
        )

    taxonomy_card(
        card_x[0],
        "I",
        "Stochastic propagation",
        COLORS["blue"],
        COLORS["blue_light"],
        ["64 state samples", "Fixed CV / MLP probes", "Future-law comparison"],
        "Does the trained-law\ncontrast survive?",
    )
    taxonomy_card(
        card_x[1],
        "II",
        "Representation utility",
        COLORS["violet"],
        COLORS["violet_light"],
        ["DSE or probability token", "GT-supervised forecaster", "Matched-control comparison"],
        "Is explicit probability\nincrementally useful?",
    )
    taxonomy_card(
        card_x[2],
        "III",
        "Objective necessity",
        COLORS["teal"],
        COLORS["teal_light"],
        ["MC future-law teacher", "Preservation student", "vs identical GT-only student"],
        "Is preserving the pushforward\nnecessary?",
    )

    ax.text(
        0.5,
        0.035,
        "Same frozen perception · matched downstream controls · no map or interaction inputs",
        fontsize=6.2,
        color=COLORS["gray_dark"],
        ha="center",
        va="center",
    )
    save_bundle(fig, "figure_1_interface_taxonomy")


def make_figure_2() -> None:
    conditions = ["Clean", "Lowlight", "LiDAR point dropout", "Beam reduction"]
    keys = ["clean", "lowlight", "dropout", "beam"]
    probes = ["CV", "MLP"]
    probe_style = {
        "CV": (COLORS["blue"], "o"),
        "MLP": (COLORS["teal"], "s"),
    }
    rows: list[dict] = []

    energy_g: dict[str, list[tuple[float, float, float]]] = {probe: [] for probe in probes}
    for probe in probes:
        for label, key in zip(conditions, keys):
            entry = (
                R0["clean_proper_scores"][probe]["energy"]["P2"]
                if key == "clean"
                else R0["family_energy"][probe][key]
            )
            value = effect(entry)
            energy_g[probe].append(value)
            rows.append(
                {
                    "panel": "a",
                    "population": "development",
                    "condition": label,
                    "probe": probe,
                    "contrast": "HTC - Gaussian",
                    "metric": "trajectory energy improvement",
                    "estimate": value[0],
                    "ci_low": value[1],
                    "ci_high": value[2],
                    "n_scenes": entry["count"],
                    "source_stage": "R0",
                }
            )

    mean_clean: dict[str, tuple[float, float, float]] = {}
    risk_clean: dict[str, tuple[float, float, float]] = {}
    area: dict[str, list[float]] = {probe: [] for probe in probes}
    for probe in probes:
        entry = R0["clean_proper_scores"][probe]["energy"]["P1"]
        mean_clean[probe] = effect(entry)
        rows.append(
            {
                "panel": "b",
                "population": "development",
                "condition": "Clean",
                "probe": probe,
                "contrast": "HTC - mean-state current interface",
                "metric": "trajectory energy improvement",
                "estimate": mean_clean[probe][0],
                "ci_low": mean_clean[probe][1],
                "ci_high": mean_clean[probe][2],
                "n_scenes": entry["count"],
                "source_stage": "R0",
            }
        )
        for label, key in zip(conditions, keys):
            ratio = float(R0["sharpness"][probe][key]["area_ratio"])
            area[probe].append(ratio)
            rows.append(
                {
                    "panel": "c",
                    "population": "development",
                    "condition": label,
                    "probe": probe,
                    "contrast": "HTC / Gaussian",
                    "metric": "equal-coverage area ratio",
                    "area_ratio": ratio,
                    "source_stage": "R0",
                }
            )
        risk_entry = R0["risk"][probe]["clean_s0"]["high_fde_auprc"]
        risk_clean[probe] = effect(risk_entry)
        rows.append(
            {
                "panel": "d",
                "population": "development",
                "condition": "Clean",
                "probe": probe,
                "contrast": "HTC - Gaussian",
                "metric": "high-FDE AUPRC improvement",
                "estimate": risk_clean[probe][0],
                "ci_low": risk_clean[probe][1],
                "ci_high": risk_clean[probe][2],
                "n_scenes": risk_entry["count"],
                "source_stage": "R0",
            }
        )

    write_source_csv("figure_2.csv", rows)

    fig = plt.figure(figsize=(7.16, 4.7))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.55, 1.15, 1.05], hspace=0.78, wspace=0.68)
    ax_a = fig.add_subplot(gs[:, :2])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 2])
    ax_d = fig.add_subplot(gs[2, 2])

    y_base = np.arange(len(conditions))[::-1]
    offsets = {"CV": 0.11, "MLP": -0.11}
    for probe in probes:
        vals = energy_g[probe]
        color, marker = probe_style[probe]
        errorbar_h(
            ax_a,
            y_base + offsets[probe],
            [v[0] for v in vals],
            [v[1] for v in vals],
            [v[2] for v in vals],
            color,
            marker,
            probe,
            ms=5.2,
        )
    ax_a.set_yticks(y_base)
    ax_a.set_yticklabels(conditions)
    style_effect_axis(ax_a, "Trajectory-energy gain (HTC over Gaussian)", (-0.004, 0.064))
    ax_a.legend(loc="upper right", ncol=2, handletextpad=0.4, columnspacing=0.9)
    ax_a.set_title("Development effects across original degradations", loc="left", pad=7, fontweight="bold")
    panel_label(ax_a, "a", x=-0.09, y=1.02)
    for y in y_base:
        ax_a.axhline(y, color=COLORS["gray_light"], lw=0.45, zorder=0)

    support_y = np.array([1, 0], dtype=float)
    for idx, probe in enumerate(probes):
        value = mean_clean[probe]
        color, marker = probe_style[probe]
        errorbar_h(ax_b, [support_y[idx]], [value[0]], [value[1]], [value[2]], color, marker, ms=5.0)
    ax_b.set_yticks(support_y)
    ax_b.set_yticklabels(probes)
    style_effect_axis(ax_b, "Energy gain", (-0.005, 0.14))
    ax_b.set_title("HTC over mean state\n(clean)", loc="left", pad=4, fontweight="bold")
    panel_label(ax_b, "b", x=-0.35, y=1.04)

    for probe in probes:
        color, marker = probe_style[probe]
        # Conditions are categorical; unconnected points avoid implying a
        # continuous degradation trajectory.
        ax_c.scatter(area[probe], y_base, marker=marker, color=color, s=28, label=probe, zorder=3)
    ax_c.axvline(1.0, color=COLORS["gray_mid"], lw=0.9, ls="--")
    ax_c.set_xlim(0.82, 1.02)
    ax_c.set_yticks(y_base)
    ax_c.set_yticklabels(["Clean", "Lowlight", "LiDAR point dropout", "Beam reduction"])
    ax_c.set_xlabel("Area ratio (HTC/Gaussian)", labelpad=1)
    ax_c.set_title("Equal-coverage area\n(<1: HTC sharper)", loc="left", pad=4, fontweight="bold")
    ax_c.tick_params(axis="y", length=0)
    ax_c.spines["left"].set_visible(False)
    panel_label(ax_c, "c", x=-0.35, y=1.04)

    for idx, probe in enumerate(probes):
        value = risk_clean[probe]
        color, marker = probe_style[probe]
        errorbar_h(ax_d, [support_y[idx]], [value[0]], [value[1]], [value[2]], color, marker, ms=5.0)
    ax_d.set_yticks(support_y)
    ax_d.set_yticklabels(probes)
    style_effect_axis(ax_d, "AUPRC gain", (-0.002, 0.058))
    ax_d.set_title("High-FDE AUPRC\n(clean)", loc="left", pad=4, fontweight="bold")
    panel_label(ax_d, "d", x=-0.35, y=1.04)

    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.115, top=0.94)
    save_bundle(fig, "figure_2_stochastic_propagation")


def plot_aggregate_and_seeds(
    ax: mpl.axes.Axes,
    labels: list[str],
    aggregates: list[tuple[float, float, float]],
    seeds: list[list[float]],
    xlim: tuple[float, float],
    xlabel: str,
    colors: list[str] | None = None,
) -> None:
    y = np.arange(len(labels))[::-1]
    if colors is None:
        colors = [COLORS["blue"]] * len(labels)
    for i, (agg, seed_values, color) in enumerate(zip(aggregates, seeds, colors)):
        errorbar_h(ax, [y[i]], [agg[0]], [agg[1]], [agg[2]], color, "D", ms=5.0, zorder=4)
        jitter = np.array([-0.12, 0.0, 0.12])
        ax.scatter(
            seed_values,
            y[i] + jitter[: len(seed_values)],
            s=13,
            facecolors="white",
            edgecolors=COLORS["gray"],
            linewidths=0.75,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    style_effect_axis(ax, xlabel, xlim)


def make_figure_3() -> None:
    """Plot the frozen descriptive upstream-to-downstream transfer analysis."""
    analysis_dir = OUTPUTS_DIR / "riskprop3d_upstream_downstream_analysis"
    with (analysis_dir / "seed_cell_source_data.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    with (analysis_dir / "spearman_cluster_bootstrap.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        correlation_rows = list(csv.DictReader(handle))

    panel_specs = [
        (
            "a",
            "xy_nll_gain_htc",
            "Upstream XY NLL gain, Gaussian - HTC",
            "Likelihood transfer",
            (-0.012, 0.245),
        ),
        (
            "b",
            "coverage_error_reduction_htc",
            "Natural 90% coverage-error reduction",
            "Coverage transfer",
            (-0.045, 0.092),
        ),
        (
            "c",
            "eq90_log_sharpness_gain_htc",
            "Equal-coverage log-area gain",
            "Sharpness transfer",
            (-0.057, 0.138),
        ),
    ]
    probe_specs = {
        "CV": ("downstream_es_gain_htc_cv", "o"),
        "MLP": ("downstream_es_gain_htc_mlp", "^"),
    }
    family_style = {
        ("development", "clean"): ("Development clean", COLORS["gray"]),
        ("development", "camera_lowlight"): ("Camera lowlight", COLORS["violet"]),
        ("development", "lidar_point_dropout"): ("LiDAR point dropout", COLORS["blue2"]),
        ("development", "lidar_beam_reduction"): ("Beam reduction", "#E6A23C"),
        ("held_out_degradation", "camera_extrinsic_rotation"): ("Extrinsic rotation", "#C76A87"),
        ("held_out_degradation", "camera_extrinsic_translation"): ("Extrinsic translation", "#3F9B83"),
        ("held_out_degradation", "camera_temporal_offset"): ("Temporal offset", "#69A45B"),
        ("final_validation", "clean"): ("Final validation", COLORS["black"]),
    }
    correlation = {
        (row["upstream_metric"], row["downstream_metric"]): row
        for row in correlation_rows
    }

    figure_rows: list[dict] = []
    for panel, upstream_metric, _, _, _ in panel_specs:
        for source in source_rows:
            for probe, (downstream_metric, _) in probe_specs.items():
                figure_rows.append(
                    {
                        "panel": panel,
                        "population": source["population"],
                        "condition": source["condition"],
                        "severity": int(source["severity"]),
                        "head_seed": int(source["head_seed"]),
                        "probe": probe,
                        "upstream_metric": upstream_metric,
                        "upstream_effect": float(source[upstream_metric]),
                        "downstream_metric": downstream_metric,
                        "downstream_effect": float(source[downstream_metric]),
                        "source_stage": "read-only upstream/downstream analysis",
                    }
                )
    write_source_csv("figure_3.csv", figure_rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.65), sharey=True)
    for ax, (panel, upstream_metric, xlabel, title, xlim) in zip(axes, panel_specs):
        for source in source_rows:
            key = (source["population"], source["condition"])
            _, color = family_style[key]
            severity = int(source["severity"])
            for probe, (downstream_metric, marker) in probe_specs.items():
                ax.scatter(
                    float(source[upstream_metric]),
                    float(source[downstream_metric]),
                    s=24 + 8 * severity,
                    marker=marker,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.82,
                    zorder=3,
                )
        ax.axhline(0, color=COLORS["gray_mid"], lw=0.8, ls="--", zorder=0)
        ax.axvline(0, color=COLORS["gray_light"], lw=0.8, ls=":", zorder=0)
        ax.grid(color=COLORS["gray_light"], linewidth=0.45, alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.005, 0.108)
        ax.set_xlabel(xlabel, fontsize=8.0)
        ax.set_title(title, loc="left", pad=6, fontsize=8.6, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8.0, length=2.5, width=0.7)
        panel_label(ax, panel, x=-0.13, y=1.035)

        cv = correlation[(upstream_metric, "downstream_es_gain_htc_cv")]
        mlp = correlation[(upstream_metric, "downstream_es_gain_htc_mlp")]
        ax.text(
            0.025,
            0.965,
            (
                f"CV: ρ={float(cv['spearman_rho']):.2f} "
                f"[{float(cv['cluster_bootstrap_ci95_low']):.2f}, {float(cv['cluster_bootstrap_ci95_high']):.2f}]\n"
                f"MLP: ρ={float(mlp['spearman_rho']):.2f} "
                f"[{float(mlp['cluster_bootstrap_ci95_low']):.2f}, {float(mlp['cluster_bootstrap_ci95_high']):.2f}]"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": COLORS["gray_light"],
                "alpha": 0.94,
            },
            zorder=5,
        )

    axes[0].set_ylabel("Downstream trajectory-energy gain, Gaussian - HTC", fontsize=8.0)

    beam_s3 = [
        row
        for row in source_rows
        if row["condition"] == "lidar_beam_reduction" and int(row["severity"]) == 3
    ]
    leverage = max(beam_s3, key=lambda row: float(row["downstream_es_gain_htc_cv"]))
    axes[0].annotate(
        "Beam reduction S3",
        xy=(float(leverage["xy_nll_gain_htc"]), float(leverage["downstream_es_gain_htc_cv"])),
        xytext=(-62, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["orange"],
        arrowprops={"arrowstyle": "->", "color": COLORS["orange"], "lw": 0.8},
        zorder=6,
    )

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=5.4,
            label=label,
        )
        for label, color in family_style.values()
    ]
    probe_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["black"],
            markeredgewidth=1.0,
            markersize=5.4,
            label=probe,
        )
        for probe, (_, marker) in probe_specs.items()
    ]
    fig.legend(
        handles=family_handles + probe_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=5,
        handletextpad=0.35,
        columnspacing=0.85,
        fontsize=7.6,
    )
    fig.subplots_adjust(left=0.083, right=0.99, bottom=0.265, top=0.91, wspace=0.20)
    save_bundle(fig, "figure_3_upstream_downstream_transfer")


def make_figure_4() -> None:
    rows: list[dict] = []
    r2_clean = R2C["per_condition_proper_scores"]["clean_s0"]

    a_labels = ["DSE vs raw concat", "Full vs mean-matched", "Full vs Gaussianized"]
    a_aggs = [
        effect(R1["distribution_encoding"]["energy"]),
        effect(r2_clean["C0_mean_matched"]["energy"]),
        effect(r2_clean["C2_gaussian_dse"]["energy"]),
    ]
    a_seeds = [
        list(map(float, R1["replicate_clean_energy_directions"]["R1_DSE_over_B2"])),
        list(map(float, R2C["seed_directions"]["C0_mean_matched"])),
        list(map(float, R2C["seed_directions"]["C2_gaussian_dse"])),
    ]
    for label, agg, seeds in zip(a_labels, a_aggs, a_seeds):
        rows.append(
            {
                "panel": "a",
                "population": "development",
                "condition": "Clean",
                "contrast": label,
                "metric": "trajectory energy improvement",
                "estimate": agg[0],
                "ci_low": agg[1],
                "ci_high": agg[2],
                "n_scenes": 70,
                "seed_1": seeds[0],
                "seed_2": seeds[1],
                "seed_3": seeds[2],
                "source_stage": "R1/R2-Core",
            }
        )

    baselines = ["P0_mean_only", "P1_confidence", "P2_gaussian"]
    b_labels = ["HTC token vs mean", "HTC token vs confidence", "HTC token vs Gaussian"]
    b_aggs = [effect(R3P["energy_P3_over_baselines"][key]["clean"]) for key in baselines]
    b_seeds = [list(map(float, R3P["seed_clean_energy_directions"][key])) for key in baselines]
    for label, agg, seeds in zip(b_labels, b_aggs, b_seeds):
        rows.append(
            {
                "panel": "b",
                "population": "development",
                "condition": "Clean",
                "contrast": label,
                "metric": "trajectory energy improvement",
                "estimate": agg[0],
                "ci_low": agg[1],
                "ci_high": agg[2],
                "n_scenes": 70,
                "seed_1": seeds[0],
                "seed_2": seeds[1],
                "seed_3": seeds[2],
                "source_stage": "R3-P",
            }
        )

    fidelity_keys = [
        ("Teacher energy distance", "teacher_energy_distance"),
        ("First-moment error", "first_moment_error"),
        ("Second-moment error", "second_moment_error"),
        ("90% coverage gap", "coverage90_gap"),
    ]
    fidelity = []
    for label, key in fidelity_keys:
        entry = R2D["teacher_fidelity_D3_over_D2"][key]
        agg = effect(entry)
        fidelity.append(agg)
        rows.append(
            {
                "panel": "c",
                "population": "development",
                "condition": "Clean",
                "contrast": "Preservation student vs GT-only student",
                "metric": label + " reduction",
                "estimate": agg[0],
                "ci_low": agg[1],
                "ci_high": agg[2],
                "n_scenes": entry["count"],
                "source_stage": "R2-D",
                "note": "Metric-specific normalized scale; magnitudes are not compared across rows.",
            }
        )

    mc_entry = R2D["mc_equivalence_D3_vs_D1"]["clean"]
    gt_entry = R2D["downstream_energy_D3_over_D2"]["clean"]
    # Keep the preservation student as the candidate in both comparisons.
    # Positive means lower trajectory energy than the named reference.
    downstream = [effect(mc_entry, flip=True), effect(gt_entry)]
    downstream_labels = ["vs MC teacher", "vs GT-only student"]
    for label, agg in zip(downstream_labels, downstream):
        rows.append(
            {
                "panel": "d",
                "population": "development",
                "condition": "Clean",
                "contrast": label,
                "metric": "trajectory energy improvement",
                "estimate": agg[0],
                "ci_low": agg[1],
                "ci_high": agg[2],
                "n_scenes": 70,
                "source_stage": "R2-D",
            }
        )
    d1_time = list(map(float, R2D["efficiency"]["D1_seconds"]))
    d3_time = list(map(float, R2D["efficiency"]["D3_seconds"]))
    for idx, (mc_time, student_time) in enumerate(zip(d1_time, d3_time), start=1):
        rows.extend(
            [
                {
                    "panel": "e",
                    "population": "development",
                    "contrast": "MC teacher runtime",
                    "metric": "prediction time (s)",
                    "estimate": mc_time,
                    "seed_1": idx,
                    "source_stage": "R2-D",
                },
                {
                    "panel": "e",
                    "population": "development",
                    "contrast": "one-pass preservation runtime",
                    "metric": "prediction time (s)",
                    "estimate": student_time,
                    "seed_1": idx,
                    "source_stage": "R2-D",
                },
            ]
        )
    write_source_csv("figure_4.csv", rows)

    fig = plt.figure(figsize=(7.16, 5.15))
    outer = fig.add_gridspec(2, 2, height_ratios=[0.96, 1.08], hspace=0.52, wspace=0.78)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    bottom = outer[1, :].subgridspec(1, 3, width_ratios=[1.42, 1.12, 0.78], wspace=0.72)
    ax_c1 = fig.add_subplot(bottom[0, 0])
    ax_c2 = fig.add_subplot(bottom[0, 1])
    ax_c3 = fig.add_subplot(bottom[0, 2])

    plot_aggregate_and_seeds(
        ax_a,
        a_labels,
        a_aggs,
        a_seeds,
        (-0.022, 0.038),
        "Energy gain",
        [COLORS["violet"]] * 3,
    )
    ax_a.set_title("Structured representation", loc="left", pad=6, fontweight="bold")
    panel_label(ax_a, "a", x=-0.17, y=1.02)
    ax_a.axhline(1.5, color=COLORS["gray_light"], lw=0.6, zorder=0)
    ax_a.text(
        0.015,
        0.965,
        "Initial audit",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=COLORS["gray"],
    )
    ax_a.text(
        0.015,
        0.625,
        "Matched audit",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=COLORS["gray"],
    )

    plot_aggregate_and_seeds(
        ax_b,
        b_labels,
        b_aggs,
        b_seeds,
        (-0.020, 0.020),
        "Energy gain",
        [COLORS["violet"]] * 3,
    )
    ax_b.set_title("Direct probability conditioning", loc="left", pad=6, fontweight="bold")
    panel_label(ax_b, "b", x=-0.17, y=1.02)

    y_fid = np.arange(len(fidelity_keys))[::-1]
    fidelity_display_labels = [
        "Energy distance",
        "First-moment error",
        "Second-moment error",
        "90% coverage gap",
    ]
    ax_c1.set_xlim(0, 1)
    ax_c1.set_ylim(-0.70, 4.05)
    ax_c1.set_xticks([])
    ax_c1.set_yticks([])
    for spine in ax_c1.spines.values():
        spine.set_visible(False)
    ax_c1.text(
        0.0,
        3.70,
        "Metric-specific native scales\nCompare sign and CI within rows only",
        ha="left",
        va="center",
        fontsize=7.1,
        color=COLORS["orange"],
        fontweight="bold",
        linespacing=1.15,
        bbox={"facecolor": COLORS["orange_light"], "edgecolor": "none", "pad": 2.2},
    )
    fidelity_units = ["m", "m", "relative", "proportion"]
    for y_row, label, value, unit in zip(y_fid, fidelity_display_labels, fidelity, fidelity_units):
        ax_c1.text(0.0, y_row + 0.13, label, ha="left", va="center", fontsize=7.3)
        ax_c1.text(
            0.0,
            y_row - 0.16,
            f"{value[0]:.3f} [{value[1]:.3f}, {value[2]:.3f}]  {unit}",
            ha="left",
            va="center",
            fontsize=7.3,
            color=COLORS["teal"],
            fontweight="bold",
        )
        if y_row > 0:
            ax_c1.axhline(y_row - 0.5, color=COLORS["gray_light"], lw=0.45)
    ax_c1.set_title("Teacher fidelity", loc="left", pad=5, fontweight="bold")
    panel_label(ax_c1, "c", x=-0.17, y=1.07)

    y_down = np.array([1, 0], dtype=float)
    errorbar_h(
        ax_c2,
        y_down,
        [v[0] for v in downstream],
        [v[1] for v in downstream],
        [v[2] for v in downstream],
        COLORS["teal"],
        "D",
        ms=5.0,
    )
    ax_c2.set_yticks(y_down)
    ax_c2.set_yticklabels(downstream_labels)
    style_effect_axis(ax_c2, "Preservation-student energy advantage", (-0.095, 0.13))
    ax_c2.set_title("Downstream quality", loc="left", pad=5, fontweight="bold")
    panel_label(ax_c2, "d", x=-0.22, y=1.07)

    for idx, (mc_time, student_time) in enumerate(zip(d1_time, d3_time)):
        x_shift = (idx - 1) * 0.035
        ax_c3.plot([x_shift, 1 + x_shift], [mc_time, student_time], color=COLORS["gray_mid"], lw=0.8, zorder=1)
        ax_c3.scatter([x_shift], [mc_time], color=COLORS["gray"], marker="o", s=25, zorder=2)
        ax_c3.scatter([1 + x_shift], [student_time], color=COLORS["teal"], marker="D", s=25, zorder=2)
    ax_c3.set_yscale("log")
    ax_c3.set_xticks([0, 1])
    ax_c3.set_xticklabels(["64×\nMC", "One\npass"])
    ax_c3.set_xlim(-0.13, 1.13)
    ax_c3.set_ylabel("Prediction time (s, log scale)")
    ax_c3.set_yticks([0.1, 1.0, 5.0])
    ax_c3.set_yticklabels(["0.1", "1", "5"])
    ax_c3.minorticks_off()
    ax_c3.set_title("Runtime", loc="left", pad=5, fontweight="bold")
    panel_label(ax_c3, "e", x=-0.27, y=1.07)
    ax_c3.text(
        0.5,
        0.63,
        f"{R2D['efficiency']['speedup_mean']:.2f}×",
        transform=ax_c3.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["teal"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )

    figure_key = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["gray"], markeredgecolor="white", markersize=5.5, label="paired aggregate (95% CI)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["gray"], markersize=5.0, label="individual seed"),
    ]
    fig.legend(handles=figure_key, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.12, top=0.90)
    save_bundle(fig, "figure_4_learned_interfaces")


def matrix_cell(
    ax: mpl.axes.Axes,
    x: int,
    y: int,
    status: str | None,
    reversal: bool = False,
) -> None:
    if status is None:
        rect = Rectangle(
            (x - 0.46, y - 0.42),
            0.92,
            0.84,
            facecolor=COLORS["white"],
            edgecolor=COLORS["gray_light"],
            linewidth=0.7,
            hatch="////",
        )
        ax.add_patch(rect)
        return

    positive = "positive" in status
    significant = status.startswith("significant")
    face = COLORS["blue_light"] if positive else COLORS["orange_light"]
    edge = COLORS["blue"] if positive else COLORS["orange"]
    marker = "^" if positive else "v"
    rect = Rectangle(
        (x - 0.46, y - 0.42),
        0.92,
        0.84,
        facecolor=face if significant else COLORS["white"],
        edgecolor=COLORS["gray_light"],
        linewidth=0.7,
    )
    ax.add_patch(rect)
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=42,
        facecolor=edge if significant else COLORS["white"],
        edgecolor=edge,
        linewidth=1.0,
        zorder=3,
    )
    if reversal:
        cx, cy = x + 0.28, y - 0.23
        ax.add_patch(Arc((cx, cy), 0.18, 0.18, theta1=20, theta2=200, color=COLORS["blue"], lw=1.0))
        ax.add_patch(Arc((cx, cy), 0.18, 0.18, theta1=200, theta2=380, color=COLORS["orange"], lw=1.0))


def make_figure_5() -> None:
    rows: list[dict] = []
    probes = ["CV", "MLP"]
    stochastic_conditions = [
        ("Clean", "development", "clean"),
        ("Lowlight", "development", "lowlight"),
        ("LiDAR point dropout", "development", "dropout"),
        ("Beam reduction", "development", "beam"),
        ("Extrinsic rotation", "held-out", "rotation"),
        ("Extrinsic translation", "held-out", "translation"),
        ("Temporal offset", "held-out", "temporal"),
        ("Final validation", "final", "final"),
    ]
    stochastic: dict[str, list[tuple[float, float, float]]] = {probe: [] for probe in probes}
    for label, population, key in stochastic_conditions:
        for probe in probes:
            if population == "development":
                entry = (
                    R0["clean_proper_scores"][probe]["energy"]["P2"]
                    if key == "clean"
                    else R0["family_energy"][probe][key]
                )
                seeds = R0["replicate_clean_energy_direction"][probe] if key == "clean" else []
                source = "R0"
            elif population == "held-out":
                block = EXTERNAL[key]["r0"][probe]["P2"]
                entry = block["metrics"]["energy"]
                seeds = block["energy_seed_directions"]
                source = "held-out confirmation"
            else:
                block = OFFICIAL["r0"][probe]["P2"]
                entry = block["metrics"]["energy"]
                seeds = block["energy_seed_directions"]
                source = "final validation"
            agg = effect(entry)
            stochastic[probe].append(agg)
            row = {
                "panel": "a",
                "population": population,
                "condition": label,
                "probe": probe,
                "contrast": "HTC - Gaussian",
                "metric": "trajectory energy improvement",
                "estimate": agg[0],
                "ci_low": agg[1],
                "ci_high": agg[2],
                "n_scenes": entry["count"],
                "status": classify(*agg),
                "seed_reversal": seed_reversal(agg[0], seeds),
                "source_stage": source,
            }
            for index, value in enumerate(seeds[:3], start=1):
                row[f"seed_{index}"] = value
            rows.append(row)

    columns = ["Development", "Rotation", "Translation", "Temporal", "Final val."]
    learned_rows = [
        ("Full encoder vs mean", "structured", "C0_mean_matched"),
        ("Full encoder vs Gaussianized", "structured", "C2_gaussian_dse"),
        ("HTC token vs mean", "direct", "P0_mean_only"),
        ("HTC token vs confidence", "direct", "P1_confidence"),
        ("HTC token vs Gaussian", "direct", "P2_gaussian"),
    ]
    matrix: dict[tuple[int, int], tuple[str | None, bool]] = {}
    external_column_keys = {1: "rotation", 2: "translation", 3: "temporal"}
    for row_idx, (label, kind, baseline) in enumerate(learned_rows):
        for col_idx, column in enumerate(columns):
            entry = None
            seeds: list[float] | None = None
            source = ""
            if kind == "structured":
                if col_idx == 0:
                    entry = R2C["per_condition_proper_scores"]["clean_s0"][baseline]["energy"]
                    seeds = list(map(float, R2C["seed_directions"][baseline]))
                    source = "R2-Core development"
                elif col_idx == 4:
                    block = OFFICIAL["r2_core"][baseline]
                    entry = block["metrics"]["energy"]
                    seeds = list(map(float, block["energy_seed_directions"]))
                    source = "final validation"
            else:
                if col_idx == 0:
                    entry = R3P["energy_P3_over_baselines"][baseline]["clean"]
                    seeds = list(map(float, R3P["seed_clean_energy_directions"][baseline]))
                    source = "R3-P development"
                elif col_idx in external_column_keys:
                    block = EXTERNAL[external_column_keys[col_idx]]["r3p"][baseline]
                    entry = block["metrics"]["energy"]
                    seeds = list(map(float, block["energy_seed_directions"]))
                    source = "held-out confirmation"
                elif col_idx == 4:
                    block = OFFICIAL["r3p"][baseline]
                    entry = block["metrics"]["energy"]
                    seeds = list(map(float, block["energy_seed_directions"]))
                    source = "final validation"
            if entry is None:
                matrix[(row_idx, col_idx)] = (None, False)
                rows.append(
                    {
                        "panel": "b",
                        "population": column,
                        "contrast": label,
                        "metric": "trajectory energy improvement",
                        "status": "not evaluated",
                        "source_stage": "not evaluated",
                    }
                )
                continue
            agg = effect(entry)
            status = classify(*agg)
            reversal = seed_reversal(agg[0], seeds)
            matrix[(row_idx, col_idx)] = (status, reversal)
            row = {
                "panel": "b",
                "population": column,
                "contrast": label,
                "metric": "trajectory energy improvement",
                "estimate": agg[0],
                "ci_low": agg[1],
                "ci_high": agg[2],
                "n_scenes": entry["count"],
                "status": status,
                "seed_reversal": reversal,
                "source_stage": source,
            }
            for index, value in enumerate((seeds or [])[:3], start=1):
                row[f"seed_{index}"] = value
            rows.append(row)

    write_source_csv("figure_5.csv", rows)

    fig = plt.figure(figsize=(7.16, 5.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    y = np.arange(len(stochastic_conditions))[::-1]
    offsets = {"CV": 0.11, "MLP": -0.11}
    styles = {"CV": (COLORS["blue"], "o"), "MLP": (COLORS["teal"], "s")}
    for probe in probes:
        values = stochastic[probe]
        color, marker = styles[probe]
        errorbar_h(
            ax_a,
            y + offsets[probe],
            [v[0] for v in values],
            [v[1] for v in values],
            [v[2] for v in values],
            color,
            marker,
            probe,
            ms=4.8,
        )
    ax_a.axhspan(y[3] - 0.45, y[0] + 0.45, color="#F4F7FA", zorder=-3)
    ax_a.axhspan(y[6] - 0.45, y[4] + 0.45, color="#F7F4FA", zorder=-3)
    ax_a.axhspan(y[7] - 0.45, y[7] + 0.45, color="#F4F8F6", zorder=-3)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([label for label, _, _ in stochastic_conditions])
    style_effect_axis(ax_a, "Trajectory-energy gain (HTC over Gaussian)", (-0.004, 0.064))
    ax_a.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        ncol=2,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0,
    )
    ax_a.set_title("Stochastic propagation across evaluation populations", loc="left", pad=7, fontweight="bold")
    panel_label(ax_a, "a", x=-0.085, y=1.02)
    ax_a.text(0.99, 0.89, "Development + original degradations", transform=ax_a.transAxes, ha="right", va="center", fontsize=7.4, color=COLORS["gray"])
    ax_a.text(0.99, 0.36, "Held-out degradations", transform=ax_a.transAxes, ha="right", va="center", fontsize=7.4, color=COLORS["gray"])
    ax_a.text(0.99, 0.055, "Final validation", transform=ax_a.transAxes, ha="right", va="center", fontsize=7.4, color=COLORS["gray"])

    n_rows = len(learned_rows)
    n_cols = len(columns)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            status, reversal = matrix[(row_idx, col_idx)]
            matrix_cell(ax_b, col_idx, n_rows - 1 - row_idx, status, reversal)
    ax_b.set_xlim(-0.55, n_cols - 0.45)
    ax_b.set_ylim(-0.55, n_rows - 0.45)
    ax_b.set_xticks(np.arange(n_cols))
    ax_b.set_xticklabels(columns)
    ax_b.xaxis.tick_top()
    ax_b.tick_params(axis="x", length=0, pad=4)
    ax_b.set_yticks(np.arange(n_rows))
    ax_b.set_yticklabels([label for label, _, _ in learned_rows][::-1])
    ax_b.tick_params(axis="y", length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    ax_b.set_title("Learned representations: direction, CI, and seed stability", loc="left", pad=22, fontweight="bold")
    panel_label(ax_b, "b", x=-0.085, y=1.15)
    ax_b.axhline(2.5, color=COLORS["gray_mid"], lw=0.8)

    legend_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLORS["blue"], markeredgecolor=COLORS["blue"], markersize=6, label="95% CI > 0"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor=COLORS["blue"], markersize=6, label="positive, CI spans 0"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="white", markeredgecolor=COLORS["orange"], markersize=6, label="adverse, CI spans 0"),
        Patch(facecolor="white", edgecolor=COLORS["gray_mid"], hatch="////", label="not evaluated"),
        Line2D([0], [0], marker="o", color=COLORS["gray"], markerfacecolor="none", markersize=6, lw=0, label="split ring: seed reversal"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.008),
        ncol=3,
        handletextpad=0.35,
        columnspacing=0.9,
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.195, right=0.985, bottom=0.19, top=0.96)
    save_bundle(fig, "figure_5_interface_stability")


def side_by_side_metric_axes(
    ax_left: mpl.axes.Axes,
    ax_right: mpl.axes.Axes,
    labels: list[str],
    left_values: list[tuple[float, float, float]],
    right_values: list[tuple[float, float, float]],
    left_xlabel: str,
    right_xlabel: str,
    left_xlim: tuple[float, float],
    right_xlim: tuple[float, float],
) -> None:
    y = np.arange(len(labels))[::-1]
    errorbar_h(
        ax_left,
        y,
        [v[0] for v in left_values],
        [v[1] for v in left_values],
        [v[2] for v in left_values],
        COLORS["violet"],
        "D",
        ms=4.8,
    )
    errorbar_h(
        ax_right,
        y,
        [v[0] for v in right_values],
        [v[1] for v in right_values],
        [v[2] for v in right_values],
        COLORS["blue"],
        "o",
        ms=4.8,
    )
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels([])
    style_effect_axis(ax_left, left_xlabel, left_xlim)
    style_effect_axis(ax_right, right_xlabel, right_xlim)


def make_figure_s1() -> None:
    rows: list[dict] = []

    dev_baselines = ["P0_mean_only", "P1_confidence"]
    dev_labels = ["Mean-only", "Confidence"]
    dev_energy = [effect(R3P["energy_P3_over_baselines"][key]["clean"]) for key in dev_baselines]
    dev_risk = [effect(R3P["risk_P3_over_baselines"][key]["risk_coverage"]) for key in dev_baselines]

    official_baselines = ["P0_mean_only", "P1_confidence", "P2_gaussian"]
    official_labels = ["Mean-only", "Confidence", "Gaussian token"]
    official_energy = [effect(OFFICIAL["r3p"][key]["metrics"]["energy"]) for key in official_baselines]
    official_risk = [effect(OFFICIAL["r3p"][key]["risk"]["risk_coverage"]) for key in official_baselines]

    for panel, population, labels, baseline_keys, energies, risks in [
        ("a", "development", dev_labels, dev_baselines, dev_energy, dev_risk),
        ("b", "final validation", official_labels, official_baselines, official_energy, official_risk),
    ]:
        for label, key, energy_value, risk_value in zip(labels, baseline_keys, energies, risks):
            rows.extend(
                [
                    {
                        "panel": panel,
                        "population": population,
                        "contrast": f"HTC token - {label}",
                        "metric": "trajectory energy improvement",
                        "estimate": energy_value[0],
                        "ci_low": energy_value[1],
                        "ci_high": energy_value[2],
                        "n_scenes": 70 if panel == "a" else 150,
                        "source_stage": "R3-P" if panel == "a" else "final validation",
                    },
                    {
                        "panel": panel,
                        "population": population,
                        "contrast": f"HTC token - {label}",
                        "metric": "risk-coverage improvement",
                        "estimate": risk_value[0],
                        "ci_low": risk_value[1],
                        "ci_high": risk_value[2],
                        "n_scenes": 70 if panel == "a" else 150,
                        "source_stage": "R3-P" if panel == "a" else "final validation",
                    },
                ]
            )

    risk_metrics = [
        ("High-FDE\nAUPRC", "high_fde_auprc"),
        ("Risk--\ncoverage", "risk_coverage"),
        ("False-safe\nrate", "false_safe"),
    ]
    stochastic_risk: dict[str, list[tuple[float, float, float]]] = {"CV": [], "MLP": []}
    for probe in ["CV", "MLP"]:
        for label, key in risk_metrics:
            entry = OFFICIAL["r0"][probe]["P2"]["risk"][key]
            agg = effect(entry)
            stochastic_risk[probe].append(agg)
            rows.append(
                {
                    "panel": "c",
                    "population": "final validation",
                    "probe": probe,
                    "contrast": "HTC - Gaussian",
                    "metric": label.replace("\n", " "),
                    "estimate": agg[0],
                    "ci_low": agg[1],
                    "ci_high": agg[2],
                    "n_scenes": entry["count"],
                    "source_stage": "final validation",
                }
            )
    write_source_csv("figure_s1.csv", rows)

    fig = plt.figure(figsize=(7.16, 4.45))
    outer = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.0], hspace=0.55, wspace=0.55)
    block_a = outer[0, 0].subgridspec(1, 2, wspace=0.5)
    block_b = outer[0, 1].subgridspec(1, 2, wspace=0.5)
    ax_a1 = fig.add_subplot(block_a[0, 0])
    ax_a2 = fig.add_subplot(block_a[0, 1])
    ax_b1 = fig.add_subplot(block_b[0, 0])
    ax_b2 = fig.add_subplot(block_b[0, 1])
    ax_c = fig.add_subplot(outer[1, :])

    side_by_side_metric_axes(
        ax_a1,
        ax_a2,
        dev_labels,
        dev_energy,
        dev_risk,
        "Energy gain",
        "Risk--coverage gain",
        (-0.020, 0.020),
        (-0.003, 0.016),
    )
    ax_a1.set_title("Development: direct HTC token", loc="left", pad=7, fontweight="bold")
    panel_label(ax_a1, "a", x=-0.38, y=1.04)

    side_by_side_metric_axes(
        ax_b1,
        ax_b2,
        official_labels,
        official_energy,
        official_risk,
        "Energy gain",
        "Risk--coverage gain",
        (-0.008, 0.044),
        (-0.010, 0.038),
    )
    ax_b1.set_title("Final validation: direct HTC token", loc="left", pad=7, fontweight="bold")
    panel_label(ax_b1, "b", x=-0.38, y=1.04)

    x = np.arange(len(risk_metrics), dtype=float)
    offsets = {"CV": -0.09, "MLP": 0.09}
    styles = {"CV": (COLORS["blue"], "o"), "MLP": (COLORS["teal"], "s")}
    for probe in ["CV", "MLP"]:
        values = stochastic_risk[probe]
        color, marker = styles[probe]
        means = np.array([v[0] for v in values])
        lows = np.array([v[1] for v in values])
        highs = np.array([v[2] for v in values])
        ax_c.errorbar(
            x + offsets[probe],
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.2,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=probe,
        )
    ax_c.axhline(0, color=COLORS["gray_mid"], lw=0.9, ls="--")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([label for label, _ in risk_metrics])
    ax_c.set_ylabel("Absolute metric improvement\n(HTC over Gaussian)")
    ax_c.set_ylim(-0.004, 0.047)
    ax_c.set_title("Final validation: stochastic forecast-error ranking", loc="left", pad=7, fontweight="bold")
    ax_c.legend(loc="upper right", ncol=2, handletextpad=0.4, columnspacing=0.9)
    panel_label(ax_c, "c", x=-0.06, y=1.04)

    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.14, top=0.94)
    save_bundle(fig, "figure_s1_risk_boundary")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)
    validate_frozen_anchors()
    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_4()
    make_figure_5()
    make_figure_s1()
    print(f"Generated figure bundle in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
