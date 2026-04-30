from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def save_figure_pdf(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)

    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )

    plt.close(fig)
    return output_path


fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])

save_figure_pdf(fig, "chart.pdf")
