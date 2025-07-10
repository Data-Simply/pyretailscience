"""PlotStyler class that applies styling using the context."""

from matplotlib.axes import Axes
from matplotlib.text import Text

from pyretailscience.plots.styles.styling_context import get_styling_context


class PlotStyler:
    """Helper class for applying all styling - fonts, colors, and hardcoded elements."""

    def __init__(self) -> None:
        """Initialize the PlotStyler with the current styling context."""
        self.context = get_styling_context()

    def apply_base_styling(self, ax: Axes) -> None:
        """Apply base plot styling (spines, grid, background) - using hardcoded defaults."""
        # These remain hardcoded as they represent the pyretailscience visual identity
        ax.set_facecolor("w")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(which="major", axis="x", color="#DAD8D7", alpha=0.5, zorder=1)
        ax.grid(which="major", axis="y", color="#DAD8D7", alpha=0.5, zorder=1)

    def apply_title(self, ax: Axes, title: str, pad: int | None = 10) -> None:
        """Apply title styling using context fonts."""
        fonts = self.context.fonts

        ax.set_title(
            title,
            fontproperties=self.context.get_font_properties(fonts.title_font),
            fontsize=fonts.title_size,
            pad=pad,
        )

    def apply_label(self, ax: Axes, label: str, axis: str, pad: int | None = 10) -> None:
        """Apply axis label styling using context fonts."""
        fonts = self.context.fonts

        font_props = self.context.get_font_properties(fonts.label_font)

        axis_fn = ax.set_xlabel if axis == "x" else ax.set_ylabel
        axis_fn(label, fontproperties=font_props, fontsize=fonts.label_size, labelpad=pad)

    def apply_ticks(self, ax: Axes) -> None:
        """Apply tick styling using context fonts."""
        fonts = self.context.fonts

        ax.tick_params(axis="both", which="both", labelsize=fonts.tick_size)

        tick_font_props = self.context.get_font_properties(fonts.tick_font)
        for tick in [
            *ax.xaxis.get_major_ticks(),
            *ax.xaxis.get_minor_ticks(),
            *ax.yaxis.get_major_ticks(),
            *ax.yaxis.get_minor_ticks(),
        ]:
            tick.label1.set_fontproperties(tick_font_props)
            tick.label2.set_fontproperties(tick_font_props)

    def apply_source_text(
        self,
        ax: Axes,
        text: str,
        font_size: float | None = None,
        vertical_padding: float = 2,
        is_venn_diagram: bool = False,
        **kwargs: object,
    ) -> Text:
        """Apply source text styling using context fonts.

        Args:
            ax (Axes): The graph to add the source text to.
            text (str): The source text.
            font_size (float, optional): The font size of the source text. If None, uses styling context default.
            vertical_padding (float, optional): The padding in ems below the x-axis label. Defaults to 2.
            is_venn_diagram (bool, optional): Flag to indicate if the diagram is a Venn diagram.
                If True, `x_norm` and `y_norm` will be set to fixed values. Defaults to False.
            **kwargs: Additional keyword arguments for positioning (x, y coordinates)

        Returns:
            Text: The source text object.
        """
        fonts = self.context.fonts

        effective_font_size = font_size or fonts.source_size

        # Calculate position if not provided in kwargs
        if "x" not in kwargs or "y" not in kwargs:
            ax.figure.canvas.draw()

            if is_venn_diagram:
                x_norm, y_norm = 0.01, 0.02
            else:
                # Get y coordinate of the text
                xlabel_box = ax.xaxis.label.get_window_extent(renderer=ax.figure.canvas.get_renderer())

                top_of_label_px = xlabel_box.y0
                padding_px = vertical_padding * effective_font_size
                y_disp = top_of_label_px - padding_px - (xlabel_box.height)

                # Convert display coordinates to normalized figure coordinates
                y_norm = y_disp / ax.figure.bbox.height

                # Get x coordinate of the text
                ylabel_box = ax.yaxis.label.get_window_extent(renderer=ax.figure.canvas.get_renderer())
                title_box = ax.title.get_window_extent(renderer=ax.figure.canvas.get_renderer())
                min_x0 = min(ylabel_box.x0, title_box.x0)
                x_norm = ax.figure.transFigure.inverted().transform((min_x0, 0))[0]

            # Use calculated positions if not provided
            x_pos = kwargs.get("x", x_norm)
            y_pos = kwargs.get("y", y_norm)
        else:
            x_pos = kwargs.get("x", 0.01)
            y_pos = kwargs.get("y", 0.02)

        return ax.figure.text(
            x_pos,
            y_pos,
            text,
            ha="left",
            va="bottom",
            transform=ax.figure.transFigure,
            fontsize=effective_font_size,
            fontproperties=self.context.get_font_properties(fonts.source_font),
            color="dimgray",
        )

    def apply_legend(self, ax: Axes, title: str | None = None, outside: bool = False) -> None:
        """Apply legend styling using context fonts."""
        fonts = self.context.fonts

        legend = (
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
            if outside
            else ax.legend(frameon=False)
        )

        if title:
            legend.set_title(title)
            legend.get_title().set_fontproperties(self.context.get_font_properties(fonts.label_font))
            legend.get_title().set_fontsize(fonts.label_size)

        # Apply styling to legend text
        legend_font_props = self.context.get_font_properties(fonts.label_font)
        for text in legend.get_texts():
            text.set_fontproperties(legend_font_props)
            text.set_fontsize(fonts.label_size - 1)
