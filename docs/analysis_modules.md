---
title: Analysis Modules
social:
  cards_layout_options:
    title: PyRetailScience | Retail Analytics Toolbox
---

## Plots

### Waterfall Plot

<div class="clear" markdown>

![Image title](assets/images/analysis_modules/waterfall.svg){ align=right loading=lazy width="50%"}

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

### Timeline Plot

<div class="clear" markdown>

![Image title](assets/images/analysis_modules/time_plot.svg){ align=right loading=lazy width="50%"}

Timeline plots are a fundamental tool for interpreting transactional data within a temporal context. By presenting data
in a chronological sequence, these visualizations reveal patterns and trends that might otherwise remain hidden in raw
numbers, making them essential for both historical analysis and forward-looking insights. They are particularly useful
for:

- Tracking sales performance across different periods (e.g., daily, weekly, monthly)
- Identifying seasonal patterns or promotional impacts on sales
- Comparing the performance of different product categories or store locations over time
- Visualizing customer behavior trends, such as purchase frequency or average transaction value

</div>

Example:

```python
import numpy as np
import pandas as pd

from pyretailscience.standard_graphs import time_plot

# Create a sample DataFrame with 3 groups
rng = np.random.default_rng(42)
df = pd.DataFrame(
    {
        "transaction_datetime": pd.concat([pd.Series(pd.date_range(start="2022-01-01", periods=200, freq="D"))] * 3),
        "total_price": np.concatenate([rng.integers(1, 1000, size=200) * multiplier for multiplier in range(1, 4)]),
        "group": ["Group A"] * 200 + ["Group B"] * 200 + ["Group C"] * 200,
    },
)

time_plot(
    df,
    period="M",
    group_col="group",
    value_col="total_price",
    agg_func="sum",
    title="Monthly Sales by Customer Group",
    y_label="Sales",
    legend_title="Customer Group",
    source_text="Source: PyRetailScience - Sales FY2024",
    move_legend_outside=True,
)
```

## Analysis Modules

### Product Association Rules

The product association module implements functionality for generating product association rules, a powerful technique
in retail analytics and market basket analysis.

Product association rules are used to uncover relationships between different products that customers tend to purchase
together. These rules provide valuable insights into consumer behavior and purchasing patterns, which can be leveraged
by retail businesses in various ways:

1. Cross-selling and upselling: By identifying products frequently bought together, retailers can make targeted product
   recommendations to increase sales and average order value.

2. Store layout optimization: Understanding product associations helps in strategic product placement within stores,
   potentially increasing impulse purchases and overall sales.

3. Inventory management: Knowing which products are often bought together aids in maintaining appropriate stock levels
   and predicting demand.

4. Marketing and promotions: Association rules can guide the creation ofeffective bundle offers and promotional
   campaigns.

5. Customer segmentation: Patterns in product associations can reveal distinct customer segments with specific
   preferences.

6. New product development: Insights from association rules can inform decisions about new product lines or features.

The module uses metrics such as support, confidence, and uplift to quantifythe strength and significance of product
associations:

- Support: The frequency of items appearing together in transactions.
- Confidence: The likelihood of buying one product given the purchase of another.
- Uplift: The increase in purchase probability of one product when another is bought.

Example:

```python
from pyretailscience.product_association import ProductAssociation

pa = ProductAssociation(
    df,
    value_col="product_name",
    group_col="transaction_id",
)
pa.df.head()
```
<!-- markdownlint-disable MD013 -->
| product_name_1   | product_name_2               |  occurrences_1 |  occurrences_2 |  cooccurrences |  support | confidence | uplift |
|:-----------------|:-----------------------------|---------------:|---------------:|---------------:|---------:|-----------:|-------:|
| 100 Animals Book | 100% Organic Cold-Pressed... |             78 |             78 |              1 | 0.000039 |  0.0128205 |   4.18 |
| 100 Animals Book | 20K Sousaphone               |             78 |             81 |              3 | 0.000117 |  0.0384615 |  12.10 |
| 100 Animals Book | 360 Sport 2.0 Boxer Briefs   |             78 |             79 |              1 | 0.000039 |  0.0128205 |   4.13 |
| 100 Animals Book | 4-Series 4K UHD              |             78 |             82 |              1 | 0.000039 |  0.0128205 |   3.98 |
| 100 Animals Book | 700S Eterna Trumpet          |             78 |             71 |              1 | 0.000039 |  0.0128205 |   4.60 |
<!-- markdownlint-enable MD013 -->
