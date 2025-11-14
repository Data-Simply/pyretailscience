"""This module provides functionality for creating Venn and Euler diagrams from pandas DataFrames.

It is designed to visualize relationships between sets, highlighting intersections and differences between them.

### Core Features

- **Supports 2-set and 3-set Diagrams**: Allows visualization of up to three overlapping sets.
- **Venn and Euler Diagrams**: Uses Venn diagrams by default; switches to Euler diagrams when `vary_size=True`.
- **Customizable Colors and Labels**: Automatically assigns colors and labels for subset representation.
- **Dynamic Sizing**: Adjusts circle sizes for Euler diagrams to reflect proportions.
- **Title and Source Attribution**: Optionally adds a title and source text.

### Use Cases

- **Set Comparisons**: Identify shared and unique elements across two or three sets.
- **Proportional Representation**: Euler diagrams ensure area-accurate representation.
- **Data Overlap Visualization**: Helps in understanding relationships within categorical data.

### Limitations and Warnings

- **Only Supports 2 or 3 Sets**: Does not extend to Venn diagrams with more than three sets.
- **Pre-Aggregated Data Required**: The module does not perform data aggregation; input data should already be structured correctly.

"""

from collections.abc import Callable

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes, SubplotBase
from matplotlib_set_diagrams import EulerDiagram, VennDiagram

from pyretailscience.plots.styles import graph_utils as gu
from pyretailscience.plots.styles.styling_helpers import PlotStyler
from pyretailscience.plots.styles.tailwind import get_plot_colors

MAX_SUPPORTED_SETS = 3
MIN_SUPPORTED_SETS = 2


def plot(
    df: pd.DataFrame,
    labels: list[str],
    title: str | None = None,
    source_text: str | None = None,
    vary_size: bool = False,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
    subset_label_formatter: Callable | None = None,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Plots a Venn or Euler diagram using subset sizes extracted from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'groups' and 'percent' columns.
        labels (list[str]): Labels for the sets in the diagram.
        title (str, optional): Title of the plot. Defaults to None.
        source_text (str, optional): Source text for attribution. Defaults to None.
        vary_size (bool, optional): Whether to vary circle size based on subset sizes. Defaults to False.
        figsize (tuple[int, int], optional): Size of the plot. Defaults to None.
        ax (Axes, optional): Matplotlib axes object to plot on. Defaults to None.
        subset_label_formatter (callable, optional): Function to format subset labels. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        SubplotBase: The matplotlib axes object with the plotted diagram.

    Raises:
        ValueError: If the number of sets is not 2 or 3.
    """
    num_sets = len(labels)
    if num_sets not in {MIN_SUPPORTED_SETS, MAX_SUPPORTED_SETS}:
        raise ValueError("Only 2-set or 3-set Venn diagrams are supported.")

    default_colors = get_plot_colors(num_sets)
    colors = kwargs.pop("color", default_colors)

    zero_group = (0, 0) if num_sets == MIN_SUPPORTED_SETS else (0, 0, 0)
    percent_s = df.loc[df["groups"] != zero_group, ["groups", "percent"]].set_index("groups")["percent"]
    subset_sizes = percent_s.to_dict()

    subset_labels = {
        k: subset_label_formatter(v) if subset_label_formatter else str(v) for k, v in subset_sizes.items()
    }

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    diagram_class = EulerDiagram if vary_size else VennDiagram
    diagram = diagram_class(
        set_labels=labels,
        subset_sizes=subset_sizes,
        subset_labels=subset_labels,
        set_colors=colors,
        ax=ax,
        **kwargs,
    )

    center_x, center_y, displacement = 0.5, 0.5, 0.1
    styler = PlotStyler()
    for text in diagram.set_label_artists:
        text.set_fontproperties(styler.context.get_font_properties(styler.context.fonts.label_font))
        text.set_fontsize(styler.context.fonts.label_size)
        if num_sets == MAX_SUPPORTED_SETS and not vary_size:
            x, y = text.get_position()
            direction_x, direction_y = x - center_x, y - center_y
            scale = displacement / (direction_x**2 + direction_y**2) ** 0.5
            text.set_position((x + scale * direction_x, y + scale * direction_y))

    for subset_id in subset_sizes:
        if subset_id not in diagram.subset_label_artists:
            continue
        text = diagram.subset_label_artists[subset_id]
        text.set_fontproperties(styler.context.get_font_properties(styler.context.fonts.label_font))

    if title:
        styler.apply_title(ax, title, pad=styler.context.fonts.title_size + 20)

    if source_text is not None:
        ax.set_xticklabels([], visible=False)
        gu.add_source_text(ax=ax, source_text=source_text, is_venn_diagram=True)

    return ax
