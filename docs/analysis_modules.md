---
title: Analysis Modules
social:
  cards_layout_options:
    title: PyRetailScience | Retail Analytics Toolbox
---

## Plots

### Waterfall Plot

<div class="clear" markdown>

![Image title](../assets/images/analysis_modules/waterfall.svg){ align=right loading=lazy width="50%"}

Waterfall plots are particularly good for showing how different things add or subtract from a starting number. For
instance,

- Changes in sales figures from one period to another
- Breakdown of profit margins
- Impact of different product categories on overall revenue

They are often used to identify key drivers of financial performance, highlight areas for improvement, and communicate
complex data stories to stakeholders in an intuitive manner.

</div>

Example:

```python
from pyretailscience.standard_graphs import waterfall_plot

labels = ["New", "Continuning", "Churned"]
amounts = [660000, 420000, -382000]

waterfall_plot(
    labels=labels,
    amounts=amounts,
    title="New customer growth hiding churn issue",
    source_text="Source: PyRetailScience - Sales FY2024 vs FY2023",
    display_net_bar=True,
    rot=0,
)
```
