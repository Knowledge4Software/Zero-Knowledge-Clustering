#!/usr/bin/env python
"""schematics.py -- conceptual diagrams (matplotlib, no data dependency):
protocol, security game, generator, dataflow, architecture, workflow,
motivation. Writes PDF+PNG to results/figures/.

Usage: python -m figures.schematics --diagram protocol
       python -m figures.schematics --diagram all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from experiments.common import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
LABELS = ["protocol", "security_game", "generator", "dataflow", "architecture", "workflow", "motivation"]


def _save(fig, label: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{label}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIR / (label + '.pdf')} (+.png)")


def _box(ax, xy, text, w=2.2, h=0.8, fc="#eaf2f8"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="black")
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)
    return (x, y, w, h)


def _arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, color="black"))


def _chain_diagram(steps, title):
    n = len(steps)
    fig, ax = plt.subplots(figsize=(max(6, 2.4 * n), 2.2))
    boxes = []
    for i, s in enumerate(steps):
        b = _box(ax, (i * 2.6, 0), s, w=2.2, h=1.0)
        boxes.append(b)
    for i in range(n - 1):
        x0, y0, w0, h0 = boxes[i]
        x1, y1, w1, h1 = boxes[i + 1]
        _arrow(ax, (x0 + w0, y0 + h0 / 2), (x1, y1 + h1 / 2))
    ax.set_xlim(-0.3, n * 2.6)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    return fig


def diagram_protocol():
    fig = _chain_diagram(
        ["Prover commits\n$C_X$", "Publish\n$(C_X,C_\\theta)$", "Verifier issues\n$\\rho$",
         "Derive seed $s$,\nexpand $A$, $C_A$", "Project & cluster\n$Z=XA^T$", "Prove $\\pi$"],
        "Alg. 1: commit-then-sample sparse projection protocol")
    _save(fig, "protocol")


def diagram_security_game():
    fig, ax = plt.subplots(figsize=(6, 4))
    _box(ax, (0, 2.4), "Challenger:\nrecord $(C_X,C_\\theta)$", w=2.6)
    _box(ax, (3.2, 2.4), "sample $\\rho$", w=2.0)
    _box(ax, (0, 0.8), "Adversary:\nchoose $X$ before $\\rho$", w=2.8)
    _box(ax, (3.4, 0.8), "wins if projected\ngeometry favorable\nafter seeing $\\rho$", w=2.8)
    _arrow(ax, (2.6, 2.8), (3.2, 2.8))
    _arrow(ax, (1.3, 2.4), (1.3, 1.6))
    _arrow(ax, (4.3, 2.4), (4.3, 1.6))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_title("Commit-then-sample security game (grinding adversary)", fontsize=10)
    _save(fig, "security_game")


def diagram_generator():
    fig = _chain_diagram(
        ["seed $s$", "SHAKE256\nstream", "per-column\nrejection sample", "distinct rows\n+ signs",
         "sparse list $A$\n(t,ell,row,sign)"],
        "SparseMatrixGen(s, m, d, q)")
    _save(fig, "generator")


def diagram_dataflow():
    fig = _chain_diagram(
        ["raw dataset", "canonical\npreprocessing", "quantize\n(scale S, bound B)",
         "candidate\nprojection", "native reference\nclustering", "circuit\nwitness/proof",
         "metrics &\nanalysis"],
        "End-to-end evaluation dataflow")
    _save(fig, "dataflow")


def diagram_architecture():
    fig, ax = plt.subplots(figsize=(7, 4))
    modules = ["dataset\nencoder", "commitment\nbuilder", "randomness\nadapter",
               "sparse-matrix\ngenerator", "native reference\nimpl.", "circuit\nbuilder",
               "prover/\nverifier", "metrics\ncollector"]
    for i, m in enumerate(modules):
        row, col = divmod(i, 4)
        _box(ax, (col * 2.0, -row * 1.4), m, w=1.7, h=1.0,
             fc="#fdebd0" if m.startswith("circuit") or m.startswith("prover") else "#eaf2f8")
    ax.text(-0.3, -2.6, "trusted preprocessing (left of circuit builder) vs. constrained computation (right)",
            fontsize=8)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-3.2, 1.5)
    ax.axis("off")
    ax.set_title("Implementation architecture", fontsize=10)
    _save(fig, "architecture")


def diagram_workflow():
    fig = _chain_diagram(
        ["raw datasets", "canonical\npreprocessing", "full-dim\nreference clustering",
         "candidate\nprojections", "projected\nclustering", "circuit\ncompilation",
         "proof runs", "analysis"],
        "End-to-end evaluation workflow")
    _save(fig, "workflow")


def diagram_motivation():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.85, "Why project before proving?", ha="center", fontsize=12, weight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.55,
            "Full-dimensional distance/argmin circuits scale with $d$;\n"
            "commit-then-sample sparse projection reduces $d \\to m \\ll d$\n"
            "while a JL-type guarantee bounds the resulting distortion.",
            ha="center", va="center", fontsize=9, transform=ax.transAxes, wrap=True)
    ax.axis("off")
    _save(fig, "motivation")


DISPATCH = {
    "protocol": diagram_protocol, "security_game": diagram_security_game,
    "generator": diagram_generator, "dataflow": diagram_dataflow,
    "architecture": diagram_architecture, "workflow": diagram_workflow,
    "motivation": diagram_motivation,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagram", required=True, choices=LABELS + ["all"])
    args = ap.parse_args()
    labels = LABELS if args.diagram == "all" else [args.diagram]
    for label in labels:
        DISPATCH[label]()


if __name__ == "__main__":
    main()
