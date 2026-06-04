"""
Review Paper Structure — 4D Bubble Chart
X: Technology maturity (0=established DL, 10=future LLM/VLM)
Y: Literature evidence density (number of primary refs)
Bubble size: Expected section weight in the paper
Color: Section role (evidence / bridge / forward-looking / infrastructure)
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- data ----------
sections = {
    "S1\nIntroduction":        {"x": 3,  "y": 5,  "weight": 8,  "role": "evidence"},
    "S2\nExisting\nReviews":   {"x": 2,  "y": 10, "weight": 6,  "role": "evidence"},
    "S3\nDL Era\nin PA":       {"x": 1,  "y": 18, "weight": 14, "role": "evidence"},
    "S4\nDL→FM\nTransition":   {"x": 4.5,"y": 10, "weight": 12, "role": "bridge"},
    "S5\nLLM/VLM\nOpportunities":{"x":7.5,"y":13, "weight": 13, "role": "forward"},
    "S6\nWhy PA\nLags Behind":  {"x": 5.5,"y": 16, "weight": 10, "role": "infra"},
    "S7\nDatasets &\nBenchmarks":{"x":5,  "y": 14, "weight": 10, "role": "infra"},
    "S8\nValidation\n& Ethics": {"x": 8,  "y": 11, "weight": 8,  "role": "forward"},
    "S9\nRoadmap":              {"x": 9,  "y": 6,  "weight": 9,  "role": "forward"},
    "S10\nConclusion":          {"x": 6,  "y": 5,  "weight": 4,  "role": "bridge"},
}

role_colors = {
    "evidence": "#2196F3",   # blue
    "bridge":   "#FF9800",   # orange
    "forward":  "#E91E63",   # pink/red
    "infra":    "#4CAF50",   # green
}
role_labels = {
    "evidence": "Evidence-based (证据型)",
    "bridge":   "Bridge (桥梁型)",
    "forward":  "Forward-looking (前瞻型)",
    "infra":    "Infrastructure (基础设施型)",
}

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(14, 9))

# background zones
ax.axvspan(-0.5, 3.2, alpha=0.06, color='#2196F3', zorder=0)
ax.axvspan(3.2, 6.0, alpha=0.06, color='#FF9800', zorder=0)
ax.axvspan(6.0, 10.5, alpha=0.06, color='#E91E63', zorder=0)
ax.text(1.3, 19.5, "Established DL", fontsize=10, color='#1565C0', ha='center', style='italic')
ax.text(4.6, 19.5, "Transition Zone", fontsize=10, color='#E65100', ha='center', style='italic')
ax.text(8.2, 19.5, "Future LLM / VLM", fontsize=10, color='#AD1457', ha='center', style='italic')

# draw bubbles
for name, d in sections.items():
    color = role_colors[d["role"]]
    size = d["weight"] * 120  # scale for visibility
    ax.scatter(d["x"], d["y"], s=size, c=color, alpha=0.55, edgecolors='white', linewidth=1.5, zorder=3)
    ax.annotate(name, (d["x"], d["y"]),
                fontsize=8, ha='center', va='center', fontweight='bold',
                color='#212121', zorder=4)

# draw flow arrows between consecutive sections
keys = list(sections.keys())
for i in range(len(keys) - 1):
    a = sections[keys[i]]
    b = sections[keys[i + 1]]
    dx = b["x"] - a["x"]
    dy = b["y"] - a["y"]
    dist = np.sqrt(dx**2 + dy**2)
    # shorten arrow to avoid overlapping bubbles
    shrink = 0.35
    ax.annotate("", xy=(b["x"] - dx * shrink / dist, b["y"] - dy * shrink / dist),
                xytext=(a["x"] + dx * shrink / dist, a["y"] + dy * shrink / dist),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.2, connectionstyle='arc3,rad=0.15'),
                zorder=2)

# legend for role colors
legend_handles = []
for role, color in role_colors.items():
    legend_handles.append(plt.scatter([], [], s=150, c=color, alpha=0.6, edgecolors='white', label=role_labels[role]))
# size legend entries
for w, label in [(5, "Small section"), (10, "Medium section"), (15, "Large section")]:
    legend_handles.append(plt.scatter([], [], s=w * 120, c='grey', alpha=0.25, edgecolors='white', label=label))

ax.legend(handles=legend_handles, loc='lower left', fontsize=8, framealpha=0.9, ncol=2,
          title="Color = Section Role · Size = Section Weight", title_fontsize=9)

# axes
ax.set_xlabel("Technology Maturity  →  (0 = established DL · 10 = future LLM/VLM)", fontsize=11)
ax.set_ylabel("Literature Evidence Density  (# primary refs supporting section)", fontsize=11)
ax.set_title("Review Paper 4D Structure Map\n"
             "X = maturity · Y = evidence · size = weight · color = role",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0, 21)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(r"I:\yc_research\claude code\BTC_analysis_tool\output\paper_4d_structure.png", dpi=180)
plt.show()
print("Done — saved to output/paper_4d_structure.png")
