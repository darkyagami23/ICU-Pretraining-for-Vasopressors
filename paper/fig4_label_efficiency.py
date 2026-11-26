import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42})
x = [0, 1, 2, 3]; lab = ["10%\n(5 seeds)", "25%\n(5)", "50%\n(3)", "100%\n(3)"]
ssl = ([0.801, 0.870, 0.861, 0.869], [0.052, 0.030, 0.068, 0.020])
scr = ([0.762, 0.843, 0.870, 0.880], [0.062, 0.042, 0.012, 0.008])
fig, ax = plt.subplots(figsize=(3.45, 2.2))
for (m, s), col, mk, ls, name, dx in [(ssl, "#257de0", "o", "-", "Pretrained", -0.06),
                                      (scr, "#f47230", "s", "--", "From scratch", 0.06)]:
    xx = [v + dx for v in x]
    ax.errorbar(xx, m, yerr=s, color=col, marker=mk, ms=4, lw=1.5, ls=ls, capsize=2.5,
                elinewidth=0.9, label=name)
ax.set_xticks(x); ax.set_xticklabels(lab)
ax.set_xlabel("Labelled training patients"); ax.set_ylabel("Test AUROC (5 min)")
ax.set_ylim(0.68, 0.95); ax.set_xlim(-0.35, 3.35)
ax.grid(axis="y", color="#dddddd", lw=0.5); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
fig.tight_layout(pad=0.3)
fig.savefig("/home/claude/figures/fig4_label_efficiency.pdf")
