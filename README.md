<!-- README.md -->
![PyRetailScience Logo](https://raw.githubusercontent.com/Data-Simply/pyretailscience/main/readme_assets/logo.png)

# PyRetailScience

⚡ Rapid bespoke and deep dive retail analytics ⚡

PyRetailScience equips you with a wide array of retail analytical capabilities,
from segmentations to gain-loss analysis. Leave the mundane to us and elevate your role
from data janitor to insights virtuoso.

## Installation

To get the latest release:

```bash
pip install pyretailscience
```

Alternatively, if you want the very latest version of the package you can install it from GitHub:

```bash
pip install git+https://github.com/Data-Simply/pyretailscience.git
```

## Features

- **Tailored for Retail**: Leverage pre-built functions designed specifically for retail analytics. From customer
  segmentations to gains loss analysis, PyRetailScience provides over a dozen building blocks you need to tackle
  retail-specific challenges efficiently and effectively.

![New Store Cannibalization Analysis](https://raw.githubusercontent.com/Data-Simply/pyretailscience/main/readme_assets/new_store_cannibalization.png)

- **Reliable Results**: Built with extensive unit testing and best practices, PyRetailScience ensures the accuracy
  and reliability of your analyses. Confidently present your findings, knowing they're backed by a robust,
  well-tested framework.

- **Professional Charts**: Say goodbye to hours of tweaking chart styles. PyRetailScience delivers beautifully
  standardized visualizations that are presentation-ready with just a few lines of code. Impress stakeholders and
  save time with our pre-built, customizable chart templates.

![Cross Shop Analysis Chart](https://raw.githubusercontent.com/Data-Simply/pyretailscience/main/readme_assets/cross_shop_chart.png)

- **Workflow Automation**: PyRetailScience streamlines your workflow by automating common retail analytics tasks.
  Easily loop analyses over different dimensions like product categories or countries, and seamlessly use the output
  of one analysis as input for another. Spend less time on data manipulation and more on generating valuable insights.

## Examples

### Gains Loss Analysis

Here is an excerpt from the gain loss analysis example [notebook](https://pyretailscience.datasimply.co/latest/examples/gain_loss/)

```python
from pyretailscience.analysis.gain_loss import GainLoss

gl = GainLoss(
    df,
    # Flag the rows of period 1
    p1_index=time_period_1,
    # Flag the rows of period 2
    p2_index=time_period_2,
    # Flag which rows are part of the focus group.
    # Namely, which rows are Calvin Klein sales
    focus_group_index=df["brand_name"] == "Calvin Klein",
    focus_group_name="Calvin Klein",
    # Flag which rows are part of the comparison group.
    # Namely, which rows are Diesel sales
    comparison_group_index=df["brand_name"] == "Diesel",
    comparison_group_name="Diesel",
    # Finally we specifiy that we want to calculate
    # the gain/loss in total revenue
    value_col="total_price",
)
# Ok now let's plot the result
gl.plot(
    x_label="Revenue Change",
    source_text="Transactions 2023-01-01 to 2023-12-31",
    move_legend_outside=True,
)
plt.show()
```

![Cross Shop Analysis Chart](https://raw.githubusercontent.com/Data-Simply/pyretailscience/main/readme_assets/gain_loss_chart.png)

### Cross Shop Analysis

Here is an excerpt from the cross shop analysis example [notebook](https://pyretailscience.datasimply.co/latest/examples/cross_shop/)

```python
from pyretailscience.analysis import cross_shop

cs = cross_shop.CrossShop(
    df,
    group_1_col="category_name",
    group_1_val="Jeans",
    group_2_col="category_name",
    group_2_val="Shoes",
    group_3_col="category_name",
    group_3_val="Dresses",
    labels=["Jeans", "Shoes", "Dresses"],
)
cs.plot(
    title="Jeans are a popular cross-shopping category with dresses",
    source_text="Source: Transactions 2023-01-01 to 2023-12-31",
    figsize=(6, 6),
)
plt.show()
# Let's see which customers were in which groups
display(cs.cross_shop_df.head())
# And the totals for all groups
display(cs.cross_shop_table_df)
```

![Cross Shop Analysis Chart](https://raw.githubusercontent.com/Data-Simply/pyretailscience/main/readme_assets/cross_shop_chart.png)

### Customer Retention Analysis

Here is an excerpt from the customer retention analysis example [notebook](https://pyretailscience.datasimply.co/latest/examples/retention/)

```python
ax = dbp.plot(
    figsize=(10, 5),
    bins=20,
    cumulative=True,
    draw_percentile_line=True,
    percentile_line=0.8,
    source_text="Source: Transactions in 2023",
    title="When Do Customers Make Their Next Purchase?",
)

# Let's dress up the chart a bit of text and get rid of the legend
churn_period = dbp.purchases_percentile(0.8)
ax.annotate(
    f"80% of customers made\nanother purchase within\n{round(churn_period)} days",
    xy=(churn_period, 0.81),
    xytext=(dbp.purchase_dist_s.min(), 0.8),
    fontsize=15,
    ha="left",
    va="center",
    arrowprops=dict(facecolor="black", arrowstyle="-|>", connectionstyle="arc3,rad=-0.25", mutation_scale=25),
)
ax.legend().set_visible(False)
plt.show()
```

![Cumulative Next Purchase Chart](https://raw.githubusercontent.com/Data-Simply/pyretailscience/main/readme_assets/days_until_next_purchase.png)

## Documentation

Please see [here](https://pyretailscience.datasimply.co/latest/) for full documentation, which includes:

- [Analysis Modules](https://pyretailscience.datasimply.co/latest/analysis_modules/): Overview of the framework and the
  structure of the docs.
- [Examples](https://pyretailscience.datasimply.co/latest/examples/retention/): If you're looking to build something
  specific or are more of a hands-on learner, check out our examples. This is the best place to get started.
- [API Reference](https://pyretailscience.datasimply.co/latest/api/gain_loss/): Thorough documentation of every class
  and method.

Documentation for specific versions is available at `https://pyretailscience.datasimply.co/<version>/` (e.g., `https://pyretailscience.datasimply.co/1.0.0/`).

## Contributing

We welcome contributions from the community to enhance and improve PyRetailScience. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear messages.
4. Push your changes to your fork.
5. Open a pull request to the main repository's `main` branch.

Please make sure to follow the existing coding style and provide unit tests for new features.

## Contact / Support

This repository is supported by Data simply.

If you are interested in seeing what Data Simply can do for you, then please email
[email us](mailto:murray@datasimply.co). We work with companies at a variety of scales and with varying levels of
data and retail analytics sophistication, to help them build, scale or streamline their analysis capabilities.

## Contributors

<a href="https://github.com/Data-Simply/pyretailscience/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=Data-Simply/pyretailscience" alt="Contributors" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## Acknowledgements

Built with expertise doing analytics and data science for scale-ups to multi-nationals, including:

- Loblaws
- Dominos
- Sainbury's
- IKI
- Migros
- Sephora
- Nectar
- Metro
- Coles
- GANNI
- Mindful Chef
- Auchan
- Attraction Tickets Direct
- Roman Originals

## License

This project is licensed under the Elastic License 2.0 - see the [LICENSE](LICENSE) file for details.

# BigQuery Integration Tests

## Overview

This directory contains integration tests that verify all PyRetailScience analysis modules
work correctly with Google BigQuery as a backend. These tests confirm that the Ibis-based
code paths function correctly when connected to BigQuery.

## Test Coverage

The integration tests cover the analysis modules.

## Prerequisites

To run these tests, you need:

1. Access to a Google Cloud Platform account
2. A service account with BigQuery permissions
3. The service account key JSON file
4. The test dataset must be loaded in BigQuery (dataset: `test_data`, table: `transactions`)

## Running the Tests

### Manual Setup

- Set up authentication:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
export GCP_PROJECT_ID=your-project-id
```

- Install dependencies:

```bash
uv sync
```

- Run the tests:

```bash
# Run all tests
uv run pytest tests/integration/bigquery -v

# Run specific test module
uv run pytest tests/integration/bigquery/test_cohort_analysis.py -v
```

## Using GitHub Actions

These tests can be run manually in GitHub Actions via the "BigQuery Integration Tests" workflow. To run:

1. Go to the "Actions" tab in the GitHub repository
2. Select the "BigQuery Integration Tests" workflow
3. Click "Run workflow"
4. Optionally enter a test filter pattern (e.g., "test_cohort_analysis")
5. Click "Run workflow"

### Required Secrets

To run the workflow in GitHub Actions, add these secrets to your repository:

- `GCP_SA_KEY`: The entire JSON content of your GCP service account key file
- `GCP_PROJECT_ID`: Your GCP project ID
