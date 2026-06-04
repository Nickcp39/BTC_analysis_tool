"""
Review Paper — Logic Flowchart (思维流程图)
Shows the argumentative flow of the paper:
  premise → evidence → bridge → opportunity → barriers → roadmap → conclusion
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis('off')

# ---------- color scheme ----------
C = {
    "premise":  "#E3F2FD",  "premise_e":  "#1565C0",
    "evidence": "#BBDEFB",  "evidence_e": "#0D47A1",
    "bridge":   "#FFF3E0",  "bridge_e":   "#E65100",
    "future":   "#FCE4EC",  "future_e":   "#AD1457",
    "infra":    "#E8F5E9",  "infra_e":    "#2E7D32",
    "roadmap":  "#F3E5F5",  "roadmap_e":  "#6A1B9A",
    "concl":    "#ECEFF1",  "concl_e":    "#37474F",
}

def box(ax, x, y, w, h, text, fc, ec, fontsize=9, bold=False, radius=0.03):
    fancy = mpatches.FancyBboxPatch((x, y), w, h,
                                     boxstyle=f"round,pad=0.15,rounding_size={radius}",
                                     facecolor=fc, edgecolor=ec, linewidth=2, zorder=3)
    ax.add_patch(fancy)
    fw = 'bold' if bold else 'normal'
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fw, color='#212121', zorder=4, wrap=True,
            linespacing=1.4)
    return (x + w / 2, y, y + h)  # cx, bottom, top

def arrow(ax, x1, y1, x2, y2, color='#616161', style='->', lw=1.8, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)

def label_arrow(ax, x1, y1, x2, y2, text, color='#616161', rad=0.0):
    arrow(ax, x1, y1, x2, y2, color=color, rad=rad)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx + 0.15, my + 0.12, text, fontsize=7, color=color, style='italic', zorder=5)

# ============================================================
# Row 1: Core premise (top)
# ============================================================
cx1, b1, t1 = box(ax, 1, 11.2, 4.5, 1.2,
    "S1 Introduction\nPA imaging + AI = important\nBut field is still in DL era",
    C["premise"], C["premise_e"], fontsize=9, bold=True)

cx2, b2, t2 = box(ax, 7.5, 11.2, 5, 1.2,
    "S2 Existing Reviews\nPA-DL reviews exist, but none cover\nLLM / VLM / MLLM → gap identified",
    C["premise"], C["premise_e"], fontsize=9, bold=True)

label_arrow(ax, cx1 + 2.25, 11.8, 7.5, 11.8, "justifies new review", C["premise_e"])

# ============================================================
# Row 2: Evidence base
# ============================================================
cx3, b3, t3 = box(ax, 0.5, 8.8, 5.5, 1.8,
    "S3 Deep Learning Era in PA\n─────────────────────\n• Reconstruction (sparse/limited view)\n• Artifact removal & enhancement\n• Segmentation & understanding\n• Quantification & uncertainty",
    C["evidence"], C["evidence_e"], fontsize=8.5, bold=False)

arrow(ax, cx1, b1, 3.25, 8.8 + 1.8, C["evidence_e"])

# ============================================================
# Row 2 right: Bridge
# ============================================================
cx4, b4, t4 = box(ax, 7.5, 8.8, 5.5, 1.8,
    "S4 DL → Foundation Model Transition\n─────────────────────────\n• Self-supervised learning\n• Diffusion & generative priors\n• Transformer-based PA\n• Promptable / training-free FMs",
    C["bridge"], C["bridge_e"], fontsize=8.5, bold=False)

label_arrow(ax, 6.0, 9.7, 7.5, 9.7, "evolves into", C["bridge_e"])
arrow(ax, cx2, b2, 10.25, 8.8 + 1.8, C["bridge_e"])

# ============================================================
# Row 3: Future opportunities
# ============================================================
cx5, b5, t5 = box(ax, 4, 6.2, 6.5, 2.0,
    "S5 Where LLM/VLM Could Enter PA\n──────────────────────────\n• Report generation & structured interpretation\n• Visual QA & multimodal querying\n• Literature-grounded assistants\n• Workflow copilots (acq→recon→quant→interp)\n• Multi-modal decision support",
    C["future"], C["future_e"], fontsize=8.5, bold=False)

arrow(ax, 3.25, b3, 5.5, 6.2 + 2.0, C["future_e"], rad=0.15)
arrow(ax, 10.25, b4, 9.0, 6.2 + 2.0, C["future_e"], rad=-0.15)

# ============================================================
# Row 4 left: Barriers
# ============================================================
cx6, b6, t6 = box(ax, 0.3, 3.3, 5.2, 2.3,
    "S6 Why PA Lags Behind\n────────────────\n• Small fragmented datasets\n• No shared benchmarks\n• No paired image-text data\n• Cross-device variability\n• Sim-to-real domain shift",
    C["infra"], C["infra_e"], fontsize=8.5, bold=False)

# Row 4 right: Data gaps
cx7, b7, t7 = box(ax, 6.5, 3.3, 5.2, 2.3,
    "S7 Datasets & Benchmarks\n────────────────────\n• Current PA data landscape\n• Benchmark fragmentation\n• Missing paired text corpora\n• Missing QA/caption resources\n• Resource-building priorities",
    C["infra"], C["infra_e"], fontsize=8.5, bold=False)

label_arrow(ax, 5.5, 6.2, 2.9, 3.3 + 2.3, "blocked by", C["infra_e"], rad=0.1)
label_arrow(ax, 8.5, 6.2, 9.1, 3.3 + 2.3, "needs", C["infra_e"], rad=-0.1)
label_arrow(ax, 5.5, 4.45, 6.5, 4.45, "overlap\n(consider merge)", '#F44336')

# ============================================================
# Row 4 far right: Ethics
# ============================================================
cx8, b8, t8 = box(ax, 12.5, 3.8, 4.5, 1.5,
    "S8 Validation, Ethics\n& Regulation\n──────────────\nHallucination · Bias\nHuman-in-loop · Safety",
    C["future"], C["future_e"], fontsize=8.5, bold=False)

arrow(ax, 10.5, 7.0, 12.5, 4.55, C["future_e"], rad=-0.2)

# ============================================================
# Row 5: Roadmap
# ============================================================
cx9, b9, t9 = box(ax, 4, 1.0, 6.5, 1.8,
    "S9 Roadmap for Next Stage of PA-AI\n──────────────────────────\nShort-term → Mid-term → Long-term\n(concrete, actionable development agenda)",
    C["roadmap"], C["roadmap_e"], fontsize=9, bold=True)

arrow(ax, 2.9, b6, 5.5, 1.0 + 1.8, C["roadmap_e"], rad=0.1)
arrow(ax, 9.1, b7, 8.5, 1.0 + 1.8, C["roadmap_e"], rad=-0.1)
arrow(ax, 14.75, b8, 10.5, 1.9, C["roadmap_e"], rad=-0.3)

# ============================================================
# Conclusion at far right bottom
# ============================================================
cx10, b10, t10 = box(ax, 12.5, 1.2, 4.5, 1.2,
    "S10 Conclusion\nPA = DL era → FM transition\nLLM/VLM = limited but timely",
    C["concl"], C["concl_e"], fontsize=8.5, bold=True)

arrow(ax, 10.5, 1.9, 12.5, 1.8, C["concl_e"])

# ============================================================
# Title
# ============================================================
ax.text(9, 12.8, "Review Paper — Argumentative Logic Flow (思维流程图)",
        fontsize=16, fontweight='bold', ha='center', va='center', color='#212121')

# Legend
legend_items = [
    ("Premise / Framing", C["premise"], C["premise_e"]),
    ("Evidence Base", C["evidence"], C["evidence_e"]),
    ("Bridge / Transition", C["bridge"], C["bridge_e"]),
    ("Forward-looking", C["future"], C["future_e"]),
    ("Infrastructure / Barriers", C["infra"], C["infra_e"]),
    ("Roadmap / Synthesis", C["roadmap"], C["roadmap_e"]),
]
for i, (label, fc, ec) in enumerate(legend_items):
    y_pos = 12.6 - i * 0.35
    fancy = mpatches.FancyBboxPatch((14.8, y_pos - 0.12), 0.5, 0.24,
                                     boxstyle="round,pad=0.05", facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(fancy)
    ax.text(15.5, y_pos, label, fontsize=8, va='center', color='#424242')

plt.tight_layout()
plt.savefig(r"I:\yc_research\claude code\BTC_analysis_tool\output\paper_logic_flowchart.png", dpi=180, bbox_inches='tight')
plt.show()
print("Done — saved to output/paper_logic_flowchart.png")
