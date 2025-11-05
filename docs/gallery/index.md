# Plot Gallery

PyRetailScience provides a comprehensive set of plotting functions designed specifically for retail analytics. All plots
use a consistent API and come pre-styled with retail-friendly color schemes and professional styling.

## Plot Types

### Basic Plots

#### [Area Plot](plots/area.ipynb)

![Area plot example](../assets/gallery/area_thumbnail.png)

Visualize data distributions over time or across categories using filled area charts. Perfect for showing trends and
comparisons between different groups through stacking or overlaying areas.

### Coming Soon

The following plot types will be added to the gallery in upcoming releases:

- **Bar Plot** - Compare categorical data with vertical or horizontal bars
- **Line Plot** - Visualize sequential data like daily trends or event impact analysis
- **Scatter Plot** - Explore relationships between two continuous variables
- **Heatmap Plot** - Display data density and patterns through color-coded matrices
- **Time Plot** - Specialized time series visualization with automatic aggregation
- **Histogram Plot** - Show distribution of continuous variables with customizable bins
- **Waterfall Plot** - Track cumulative effect of sequential positive/negative values
- **Cohort Plot** - Visualize cohort retention and behavior over time periods
- **Venn Diagram** - Display set overlaps and relationships between groups
- **Period on Period Plot** - Compare performance across different time periods
- **Broken Timeline Plot** - Show data availability and coverage across categories
- **Index Plot** - Display performance relative to baseline (100) with highlighting
- **Price Plot** - Bubble chart showing price distribution across categories

## Getting Started

Each plot page includes:

- **Basic examples** - Simple usage to get started quickly
- **Configuration options** - Major built-in features demonstrated with code
- **Realistic data** - Retail domain examples you can copy and adapt
- **Best practices** - Tips for effective visualization

All examples use the consistent PyRetailScience import pattern:

```python
from pyretailscience.plots import area
# Then call: area.plot(...)
```

## Additional Resources

- **[API Reference](../api/plots/area.md)** - Complete parameter documentation for all plots
- **[Examples](../examples/retention.ipynb)** - End-to-end analysis workflows combining multiple plots
- **[Styling Guide](../api/plots/styles/tailwind.md)** - Customize colors, themes, and appearance
