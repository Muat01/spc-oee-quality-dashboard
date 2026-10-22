"""
Generate documentation figures (docs/) and a 6-slide LinkedIn carousel (carousel/).
Optional helper: requires matplotlib and pillow (pip install matplotlib pillow).
"""
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import oee
import spc

os.makedirs("docs", exist_ok=True)
os.makedirs("carousel", exist_ok=True)

BG, CARD, TEAL = "#0a0c10", "#131822", "#00d4aa"
TEXT, MUTED = "#e8eaf0", "#8b92a8"
ORANGE, BLUE, RED = "#ffa502", "#3d8eff", "#ff4757"
W, H, TOTAL = 1080, 1350, 6
LSL, USL, BASELINE = 24.96, 25.04, 30
plt.rcParams["font.family"] = "DejaVu Sans"

# ─── Results (real, computed from the sample data) ──────────────────────────
ids, M = spc.to_subgroup_matrix(pd.read_csv("data/sample_spc.csv"))
CC = spc.xbar_r_chart(M, BASELINE)
CAP = spc.capability(M.ravel(), CC.sigma_within, LSL, USL)
prod, down = pd.read_csv("data/sample_production_log.csv"), pd.read_csv("data/sample_downtime_log.csv")
T = oee.shift_table(prod, down)
R = oee.ratios(T)
WF = oee.loss_waterfall(T)
PAR = oee.pareto(down)
BYS = oee.by_shift(T).set_index("shift")
N_SIG = len(CC.signals)


def style_axes(ax):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color("#232b3b")
    ax.tick_params(colors=MUTED, labelsize=13)
    ax.grid(color="#1c2433", lw=1)
    ax.set_axisbelow(True)


def draw_control_chart(ax, big=True):
    style_axes(ax)
    x = np.arange(1, len(CC.x) + 1)
    ax.axvspan(0.5, BASELINE + 0.5, color=TEAL, alpha=0.07)
    for k in (1, 2):
        for s in (1, -1):
            ax.axhline(CC.cl + s * k * CC.sigma_stat, color="#2a3550", lw=1, ls=":")
    ax.axhline(CC.cl, color=TEAL, lw=2.2)
    ax.axhline(CC.ucl, color=RED, lw=2, ls="--")
    ax.axhline(CC.lcl, color=RED, lw=2, ls="--")
    ax.plot(x, CC.x, color=BLUE, lw=1.6, marker="o", ms=4)
    fl = sorted(CC.signals)
    ax.scatter(x[fl], CC.x[fl], color=RED, s=46, zorder=5, edgecolor="white", lw=0.8)
    ax.set_xlim(0.5, len(x) + 0.5)
    ax.ticklabel_format(useOffset=False, axis="y")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    ax.text(len(x) + 1.5, CC.ucl, "UCL", color=RED, va="center", fontsize=13)
    ax.text(len(x) + 1.5, CC.lcl, "LCL", color=RED, va="center", fontsize=13)
    ax.text(len(x) + 1.5, CC.cl, "CL", color=TEAL, va="center", fontsize=13)
    ax.text(BASELINE / 2, CC.ucl + 0.0015, "baseline", color=TEAL, ha="center", fontsize=12)


def draw_waterfall(ax):
    style_axes(ax)
    vals = [WF["planned"], -WF["downtime"], -WF["speed"], -WF["quality"], WF["productive"]]
    labels = ["Planned\ntime", "Downtime", "Speed\nloss", "Quality\nloss", "Fully\nproductive"]
    cum, bottoms, heights, colors = 0, [], [], []
    for i, v in enumerate(vals):
        if i == 0:
            bottoms.append(0); heights.append(v); colors.append(MUTED); cum = v
        elif i == 4:
            bottoms.append(0); heights.append(v); colors.append(TEAL)
        else:
            cum += v; bottoms.append(cum); heights.append(-v); colors.append(RED)
    ax.bar(range(5), np.array(heights) / 60, bottom=np.array(bottoms) / 60, color=colors, width=0.62)
    for i, v in enumerate(vals):
        top = (bottoms[i] + heights[i]) / 60
        ax.text(i, top + 12, f"{abs(v) / 60:,.0f} h", ha="center", color=TEXT, fontsize=15, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, color=TEXT, fontsize=13)
    ax.set_ylabel("Hours", color=MUTED, fontsize=13)
    ax.set_ylim(0, WF["planned"] / 60 * 1.12)


def draw_pareto(ax):
    style_axes(ax)
    ax.bar(PAR["reason"], PAR["minutes"] / 60, color=ORANGE, width=0.62)
    ax.set_ylabel("Hours", color=MUTED, fontsize=13)
    ax.tick_params(axis="x", labelrotation=18, labelcolor=TEXT)
    ax2 = ax.twinx()
    ax2.plot(PAR["reason"], PAR["cum_pct"], color=TEAL, lw=3, marker="o")
    ax2.axhline(80, color=MUTED, ls=":", lw=1.2)
    ax2.set_ylim(0, 105); ax2.tick_params(colors=MUTED, labelsize=13)
    for sp in ax2.spines.values():
        sp.set_color("#232b3b")


# ─── README figures ─────────────────────────────────────────────────────────
for name, fn, size in (("control_chart.png", draw_control_chart, (11, 4.6)),
                       ("oee_waterfall.png", draw_waterfall, (9, 4.6)),
                       ("downtime_pareto.png", draw_pareto, (9, 4.6))):
    fig, ax = plt.subplots(figsize=size, facecolor=BG)
    fn(ax)
    fig.tight_layout()
    fig.savefig(f"docs/{name}", dpi=150, facecolor=BG)
    plt.close(fig)


# ─── Carousel ───────────────────────────────────────────────────────────────
def new_slide(n):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, H - 10), W, 10, color="#1c2230"))
    ax.add_patch(plt.Rectangle((0, H - 10), W * n / TOTAL, 10, color=TEAL))
    ax.text(W - 60, 70, f"{n}/{TOTAL}", color=MUTED, fontsize=15, ha="right", va="center")
    ax.text(60, 70, "MUATASEM HADI", color=MUTED, fontsize=15, ha="left", va="center", fontweight="bold")
    return fig, ax


def card(ax, x, y, w, h, edge="#232b3b"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=22", fc=CARD, ec=edge, lw=2))


def inset(fig, x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def save(fig, n):
    fig.savefig(f"carousel/slide_{n}.png", dpi=100, facecolor=BG)
    plt.close(fig)


# 1 cover
fig, ax = new_slide(1)
ax.text(60, 230, "SPC + OEE  ·  PYTHON + STREAMLIT", color=TEAL, fontsize=17, fontweight="bold")
ax.text(60, 400, "Is your process\nreally under\ncontrol?", color=TEXT, fontsize=64, fontweight="bold", va="center", linespacing=1.15)
ax.text(60, 650, "Control charts, process capability\nand OEE in one dashboard.", color=MUTED, fontsize=27, va="center", linespacing=1.5)
a = inset(fig, 60, 760, 960, 330)
draw_control_chart(a, big=False)
a.set_xticks([]); a.set_yticks([])
ax.text(60, 1180, "Muatasem Hadi", color=TEXT, fontsize=26, fontweight="bold")
ax.text(60, 1220, "MEng Candidate, Industrial Engineering · Telkom University", color=MUTED, fontsize=18)
ax.text(W - 60, 1215, "Swipe  →", color=TEAL, fontsize=24, ha="right", fontweight="bold")
save(fig, 1)

# 2 two tools
fig, ax = new_slide(2)
ax.text(60, 190, "Two questions every\nplant must answer", color=TEXT, fontsize=46, fontweight="bold", va="center", linespacing=1.2)
for i, (col, tag, q, body) in enumerate([
    (BLUE, "SPC", "Is the process stable and capable?",
     "Control charts show when something changed.\nCp / Cpk / Ppk show if parts meet spec."),
    (TEAL, "OEE", "How much planned time is productive?",
     "Availability × Performance × Quality\nshows where capacity is lost.")]):
    y = 380 + i * 360
    card(ax, 60, y, 960, 320, edge=col)
    ax.text(100, y + 60, tag, color=col, fontsize=30, fontweight="bold", va="center")
    ax.text(100, y + 140, q, color=TEXT, fontsize=29, fontweight="bold", va="center")
    ax.text(100, y + 235, body, color=MUTED, fontsize=22, va="center", linespacing=1.6)
ax.text(60, 1170, "Both answered from simulated plant data\n(Python, Streamlit, Plotly).", color=MUTED, fontsize=20, va="top", linespacing=1.6, style="italic")
save(fig, 2)

# 3 control chart
fig, ax = new_slide(3)
ax.text(60, 190, "Step 1: spot what changed", color=TEXT, fontsize=45, fontweight="bold", va="center")
ax.text(60, 255, "X̄ chart of a shaft diameter · 120 subgroups of 5", color=MUTED, fontsize=22, va="center")
a = inset(fig, 175, 310, 800, 520)
draw_control_chart(a)
card(ax, 60, 880, 960, 330, edge=RED)
ax.text(100, 935, f"{N_SIG} of {len(CC.x)} subgroups raise a signal", color=RED, fontsize=29, fontweight="bold", va="center")
ax.text(100, 1085, "• Sudden mean shift (new material lot)\n• Tool-wear drift, reset by a tool change\n• Range chart catches an outlier and an unstable fixture",
        color=TEXT, fontsize=22, va="center", linespacing=1.65)
save(fig, 3)

# 4 capability
fig, ax = new_slide(4)
ax.text(60, 190, "Step 2: capable, but only\nin the short term", color=TEXT, fontsize=45, fontweight="bold", va="center", linespacing=1.2)
a = inset(fig, 70, 360, 940, 420)
style_axes(a)
names, vals = ["Cp", "Cpk", "Pp", "Ppk"], [CAP.cp, CAP.cpk, CAP.pp, CAP.ppk]
colors = [BLUE, BLUE, ORANGE, ORANGE]
a.bar(names, vals, color=colors, width=0.6)
a.axhline(1.33, color=TEAL, ls="--", lw=2)
a.text(3.45, 1.35, "1.33 target", color=TEAL, fontsize=14, ha="right", va="bottom")
for i, v in enumerate(vals):
    a.text(i, v + 0.03, f"{v:.2f}", ha="center", color=TEXT, fontsize=24, fontweight="bold")
a.set_ylim(0, 1.7); a.tick_params(axis="x", labelsize=22, labelcolor=TEXT)
ax.text(60, 835, "Short-term (σ within subgroups)", color=BLUE, fontsize=17, fontweight="bold")
ax.text(1020, 835, "Long-term (overall σ)", color=ORANGE, fontsize=17, fontweight="bold", ha="right")
card(ax, 60, 900, 960, 310, edge=ORANGE)
ax.text(100, 955, f"Cpk {CAP.cpk:.2f} → Ppk {CAP.ppk:.2f}", color=ORANGE, fontsize=31, fontweight="bold", va="center")
ax.text(100, 1085, f"Shifts and drift between subgroups cost {round(CAP.cpk, 2) - round(CAP.ppk, 2):.2f} of capability.\nExpected out-of-spec: about {CAP.ppm_overall:,.0f} ppm.\nA stable process would not show this gap.",
        color=TEXT, fontsize=22, va="center", linespacing=1.65)
save(fig, 4)

# 5 OEE waterfall
fig, ax = new_slide(5)
ax.text(60, 190, f"Step 3: OEE is {R['oee'] * 100:.1f}%", color=TEXT, fontsize=46, fontweight="bold", va="center")
ax.text(60, 255, "Where 30 days × 3 shifts of planned time go", color=MUTED, fontsize=22, va="center")
a = inset(fig, 70, 320, 940, 480)
draw_waterfall(a)
for i, (lab, v, c) in enumerate([("Availability", R["availability"], BLUE), ("Performance", R["performance"], ORANGE), ("Quality", R["quality"], TEAL)]):
    x = 60 + i * 330
    card(ax, x, 860, 300, 190, edge=c)
    ax.text(x + 150, 915, lab, color=MUTED, fontsize=19, ha="center", va="center")
    ax.text(x + 150, 985, f"{v * 100:.1f}%", color=c, fontsize=40, fontweight="bold", ha="center", va="center")
ax.text(60, 1130, f"Benchmark often quoted: 85%.  Gap: {(0.85 - R['oee']) * 100:.1f} points.", color=TEXT, fontsize=24, fontweight="bold")
ax.text(60, 1190, "Downtime is the biggest single loss.", color=MUTED, fontsize=21)
save(fig, 5)

# 6 pareto + CTA
fig, ax = new_slide(6)
top2 = PAR["cum_pct"].iloc[1]
ax.text(60, 190, "Step 4: attack the vital few", color=TEXT, fontsize=45, fontweight="bold", va="center")
ax.text(60, 255, f"{PAR['reason'].iloc[0]} + {PAR['reason'].iloc[1]} = {top2:.0f}% of downtime", color=TEAL, fontsize=26, fontweight="bold", va="center")
a = inset(fig, 70, 320, 940, 400)
draw_pareto(a)
card(ax, 60, 770, 960, 150)
ax.text(100, 815, "NIGHT vs MORNING SHIFT", color=TEAL, fontsize=16, fontweight="bold", va="center")
ax.text(100, 872, f"OEE {BYS.loc['Night', 'oee'] * 100:.0f}% vs {BYS.loc['Morning', 'oee'] * 100:.0f}%", color=TEXT, fontsize=33, fontweight="bold", va="center")
card(ax, 60, 950, 960, 150, edge=TEAL)
ax.text(100, 995, "CODE + LIVE DEMO", color=TEAL, fontsize=16, fontweight="bold", va="center")
ax.text(100, 1050, "github.com/Muat01/spc-oee-quality-dashboard", color=TEXT, fontsize=21, fontweight="bold", va="center")
ax.text(60, 1150, "Sample data is simulated, not from a real factory.", color=MUTED, fontsize=19, style="italic")
ax.text(60, 1225, "What should I analyze next?", color=TEAL, fontsize=23, fontweight="bold")
save(fig, 6)

pages = [Image.open(f"carousel/slide_{i}.png").convert("RGB") for i in range(1, TOTAL + 1)]
pages[0].save("carousel/spc-oee-carousel.pdf", save_all=True, append_images=pages[1:], resolution=100.0)

print(f"signals {N_SIG}/{len(CC.x)} | Cp {CAP.cp:.2f} Cpk {CAP.cpk:.2f} Pp {CAP.pp:.2f} Ppk {CAP.ppk:.2f} "
      f"PPM {CAP.ppm_overall:.0f} | OEE {R['oee']:.3f} A {R['availability']:.3f} P {R['performance']:.3f} Q {R['quality']:.3f}")
print("done")
