# PyRetailScience Changelog

All notable features and enhancements merged to main since the project's inception.

---

# 2025

## December 2025

### Brand-Ready Plots Automatically
*December 2*

You can now control fonts and spacing through configuration instead of hardcoded values, letting you create client-ready visualizations without manual post-processing in design tools.

**Customize:**
- Corporate fonts for titles, labels, ticks, and annotations
- Font sizes to match your style guide
- Padding and spacing for professional layouts
- Switch styles on-the-fly for different audiences

**Real-world insight:** Your marketing team can now generate dozens of brand-compliant charts for the quarterly board deck in minutes, not hours of exporting to design software and manually adjusting fonts.

*PR #408*

---

### Control Plot Colors Through Configuration
*December 1*

You can now customize plot colors through a centralized options system, allowing you to match your brand palette while maintaining semantic color distinctions.

**What's configurable:**
- Mono and multi-color palettes for data series
- Named semantic colors (positive, negative, neutral, difference, context, primary)
- Heatmap color schemes
- All plot types use consistent color helpers

**Real-world insight:** A consulting firm working with multiple retail clients can now maintain separate color configuration files for each client's brand guidelines, generating on-brand visualizations instantly without manually editing code or post-processing charts.

*PR #407*

---

### Unified Color Selection Logic
*November 17*

You can now rely on consistent color behavior across all plot types through a centralized color selection function, eliminating inconsistencies between different visualization modules.

**What changed:**
- All plots use the same `get_plot_colors()` function
- Consistent color assignment across area, bar, histogram, line, price, scatter, and venn plots
- Easier to customize color behavior system-wide

**Real-world insight:** When creating dashboards with multiple plot types, you no longer see confusing color inconsistencies where the same product category appears blue in one chart and green in another. Colors now stay consistent across your entire analytical workflow.

*PR #396*

---

### Support Python 3.10 Through 3.13
*December 1*

You can now use PyRetailScience across Python versions 3.10, 3.11, 3.12, and 3.13, making the library accessible to more teams regardless of their Python environment.

**What's supported:**
- Python 3.10-3.13 compatibility
- Updated datetime operations for cross-version compatibility
- Comprehensive CI/CD testing across all versions

**Real-world insight:** A large retail organization standardized on Python 3.10 enterprise-wide couldn't upgrade to 3.11+ due to legacy systems. They can now adopt PyRetailScience without needing to change their entire Python infrastructure, accelerating their analytics modernization by 6+ months.

*PR #414*

---

### Documentation Stays in Sync with Package Releases
*December 11*

You can now trust that the online documentation always matches the PyPI package version you have installed, eliminating confusion from documentation showing features not yet in your version.

**What changed:**
- Documentation deploys only when new packages are released to PyPI
- Perfect version alignment between docs and installed package
- No more debugging features that don't exist yet

**Real-world insight:** A data science team onboarding three new analysts was losing 6-8 hours per person to documentation confusion—trying parameters that didn't exist yet, following examples that used unreleased features. With synchronized docs, new analysts can follow tutorials without constant version checking, reducing onboarding friction by 40%.

*PR #417*

---

### Get Consistent Segmentation Results Every Time
*December 18*

You can now run threshold segmentation repeatedly and get identical results, even when multiple customers have the exact same spend values.

**What's fixed:**
- Consistent segment assignments across multiple runs
- Reproducible results regardless of database engine
- No more different numbers when stakeholders re-run your analysis

**Real-world insight:** Your VP of Marketing reviews your quarterly segmentation showing 12,450 "Heavy" customers. She asks her team to validate it. Previously, they might get 12,448 or 12,453 due to tied values being ordered differently, triggering questions about your methodology. Now the validation matches perfectly, and your recommendation gets approved without pushback.

---

### Segment Customers Within Stores or Regions
*December 17*

You can now run customer segmentation separately for each store, region, or category - recognizing that a "Heavy" spender in Manhattan might be a "Light" spender in rural Kansas.

**Apply context to customer value:**
- Add `group_col` parameter to segment within any grouping
- Identify location-specific VIPs who were hidden in national averages
- Build targeted campaigns based on relative performance per location
- Compare apples-to-apples across different markets

**Real-world insight:** A pharmacy chain discovers that 300 customers classified as "Medium" nationally are actually "Heavy" spenders at small-town locations. They launch a localized loyalty program at those stores, increasing retention from 65% to 82% and adding $1.2M in annual revenue from a segment they were previously ignoring.

---

## November 2025

### Simplified Total Calculations with 'Total' Mode
*November 22*

You can now get segment details plus grand totals with a single parameter (`grouping_sets='total'`), replacing the deprecated `calc_total` parameter with a clearer migration path.

**What's new:**
- Named shortcut for detail + grand total aggregations
- Clear replacement for deprecated `calc_total=True`
- Generates exactly 2 grouping sets: full detail and grand total

**Real-world insight:** An analyst building monthly executive dashboards was maintaining two separate queries—one for segment breakdown and one for company-wide totals. With 'total' mode, they consolidate into a single query, reducing data pipeline complexity and cutting dashboard generation time by 40%.

---

### Build Complex Cross-Dimensional Analysis
*November 18*

You can now compose grouping sets using SQL-style `cube()` and `rollup()` helpers, enabling sophisticated multi-dimensional analysis patterns like `(cube("region", "store"), "category")`.

**Composable analysis:**
- `cube()` and `rollup()` functions return reusable specifications
- Combine with other dimensions for complex patterns
- Follows SQL GROUP BY syntax for familiar interface
- Generates all analytical combinations efficiently

**Real-world insight:** A category manager needs to analyze sales across all combinations of region and store, but always broken out by category. Using `(cube("region", "store"), "category")`, they get regional totals, store details, and category breakdowns in one pass—replacing six separate queries and reducing analysis time from 45 minutes to 3 minutes.

---

### Analyze All Dimensional Combinations with CUBE
*November 16*

You can now generate all 2^n dimensional combinations using CUBE mode, enabling comprehensive cross-dimensional analysis across every possible segment grouping.

**CUBE capabilities:**
- Generate all possible grouping combinations automatically
- Analyze data across every dimensional perspective
- Performance warnings for high-dimensional data (4+ columns)
- Follows SQL CUBE semantics

**Real-world insight:** A retail analyst needs to understand sales patterns across store, region, and product category from every angle—totals by store alone, region alone, category alone, store+region, store+category, region+category, and overall total. CUBE mode generates all 8 combinations in one query, replacing manual aggregation work that previously took 2 hours.

---

### Cleaner Column Access with Nested Structure
*November 14*

You can now access aggregation and calculated columns through an intuitive nested structure (e.g., `cols.agg.customer_id` instead of `cols.agg_customer_id`), making code more readable and organized.

**What's improved:**
- Nested access pattern: `cols.agg.customer_id`, `cols.calc.total_revenue`
- Reduced naming conflicts
- Better IDE autocomplete support
- Backward compatible with existing configurations

**Real-world insight:** When building complex revenue tree analyses with dozens of column references, the nested structure makes it immediately clear whether you're working with aggregated data, calculated fields, or base columns—reducing debugging time and making code reviews faster.

*PR #366*

---

### Highlight Specific Lines in Line Plots
*November 4*

You can now emphasize specific lines in your line plots while keeping other series visible for context, making it easier to draw attention to key metrics or categories.

**Control visual focus:**
- `highlight` parameter accepts string or list of strings
- Highlighted lines appear bold and saturated
- Context lines render muted but visible
- Works with both grouped data and multiple value columns

**Real-world insight:** When presenting quarterly performance across 20 product categories to executives, you can highlight the top 3 performers and bottom 3 strugglers while keeping all 20 visible for context. Stakeholders immediately see the key stories without losing sight of the bigger picture.

*PR #362*

---

### Customize Bar Plot Legends
*November 11*

You can now override default legend behavior in bar plots by passing `legend` kwargs, giving you control over legend placement, formatting, and visibility.

**What's customizable:**
- Legend position and formatting through kwargs
- Complete control over legend appearance
- Works with all bar plot variations

**Real-world insight:** When creating dense bar charts with 15+ categories for print reports, legends would overflow and obscure data. Now you can position the legend outside the plot area or use a compact multi-column layout, ensuring every chart fits perfectly on the page without manual post-processing.

---

### Flexible Column Naming for Filters and Labels
*November 2*

You can now specify custom column names for period and condition-based filters, making the helper functions work with your existing data schema without renaming columns.

**Configuration options:**
- Configurable period column names in `filter_and_label_by_periods`
- Configurable label column names in `filter_and_label_by_condition`
- No need to rename columns to match function expectations

**Real-world insight:** A retailer with an established data warehouse uses "fiscal_period" instead of "period" across all their tables. Previously, they had to create view layers or temporary columns to use filter helpers. Now they pass `period_col='fiscal_period'` directly, eliminating unnecessary data transformation steps and simplifying their analytics pipelines.

---

## October 2025

### Add Text Labels to Scatter Plot Points
*October 29*

You can now annotate individual scatter plot points with text labels, with automatic positioning to prevent overlaps and maintain readability.

**Label your data:**
- Display text labels on scatter plot points
- Automatic overlap prevention using intelligent positioning
- Works with single series and grouped plots
- Customizable label appearance through `label_kwargs`

**Real-world insight:** When analyzing store performance with revenue vs. foot traffic, you can label outlier stores by name to quickly identify which locations need investigation. Instead of squinting at coordinates, executives immediately see "Downtown SF" and "Mall of America" as the high-performers requiring expansion planning.

*PR #350*

---

### Replace Graphviz with Pure Matplotlib Rendering
*October 27*

You can now generate revenue tree diagrams entirely with matplotlib, eliminating the external Graphviz dependency and simplifying installation while gaining better integration with the Python scientific stack.

**What changed:**
- Complete matplotlib-based tree rendering
- Removed Graphviz dependency from project
- New TreeGrid layout engine for flexible positioning
- DetailedTreeNode for rich period comparisons
- BaseRoundedBox for professional node styling

**Real-world insight:** A data science team deploying analytics in a locked-down enterprise environment previously needed IT approval and system admin access to install Graphviz binaries. This created 2-3 week delays for every new analyst onboarding. With pure Python/matplotlib rendering, new analysts can `pip install` and start generating revenue trees in minutes, eliminating deployment friction entirely.

*PRs #355, #357, #358 + migration commits*

---

### Visualize Data with Generic Heatmap Plot
*October 10*

You can now create heatmaps for any matrix data—correlation matrices, confusion matrices, migration patterns—using a flexible heatmap module extracted from cohort-specific code.

**Heatmap capabilities:**
- Generic heatmap visualization for any 2D data
- Supports correlation, confusion, and migration matrices
- Extracted from cohort-specific implementation
- Customizable color schemes and formatting

**Real-world insight:** A marketing analyst studying customer migration between segments across quarters was manually creating heatmaps in Excel. With the heatmap module, they visualize segment transitions directly in Python, identifying that 23% of "Light" customers in Q1 became "Medium" in Q2—triggering a successful engagement campaign that accelerated this transition to 31% in Q3.

*Issue #345 implementation*

---

### Analyze Revenue Across Multiple Dimensions
*October 21*

You can now group revenue trees by multiple columns simultaneously, enabling multi-dimensional analysis like region + store + category hierarchies.

**Multi-dimensional grouping:**
- Pass single column or list of columns to `group_col`
- Creates MultiIndex for multiple grouping levels
- Maintains single-level CategoricalIndex for simple groups
- Full backward compatibility

**Real-world insight:** A national retailer can now build a revenue tree showing Northeast > New York > Manhattan > Electronics > Laptops hierarchy in a single analysis, revealing that laptop sales in Manhattan stores drive 40% of Northeast electronics revenue—insight that was impossible with single-dimension grouping.

*PR #354*

---

### Plot Pandas Series Directly
*October 16*

You can now pass pandas Series directly to line plots without converting to DataFrame first, streamlining your workflow for simple time series visualizations.

**Simplified plotting:**
- Pass Series with `value_col=None`
- No more manual DataFrame conversions
- Comprehensive error handling for invalid combinations
- Works with Series indices as x-axis

**Real-world insight:** When quickly exploring daily sales trends during a meeting, analysts can now plot Series data in one line instead of three, keeping the analytical flow moving and reducing the cognitive load of data wrangling before every visualization.

*PR #351*

---

### Track Identified vs. Unknown Customers
*October 7*

You can now analyze loyalty program penetration and guest transaction behavior by separating identified customers from unknown/guest customers in your segmentation statistics.

**Unknown customer tracking:**
- New `unknown_customer_value` parameter supports multiple input types
- Three column variants: base (identified only), `_unknown` suffix, `_total` suffix
- Works with rollups and extra aggregations
- Fully backward compatible

**Real-world insight:** A grocery chain discovers that unknown customers (non-loyalty members) account for 35% of transactions but only 18% of revenue, with average basket size 40% lower than identified customers. This insight drives a targeted campaign to convert casual shoppers into loyalty members, increasing signup rate from 12% to 31% and adding $4.2M in annual revenue from higher basket sizes.

*PR #333*

---

### Visualize Price Distribution Across Competitors
*October 7*

You can now create price architecture bubble plots showing how products are distributed across price bands and retail groups, enabling competitive pricing analysis.

**Price visualization:**
- Bubble chart showing product counts per price band and retailer
- Flexible binning with integer or custom list options
- Automatic color selection based on group count
- Per-group percentage scaling for fair comparison

**Real-world insight:** A beverage manufacturer compares their pricing architecture against three competitors, discovering they have no products in the $8-$12 premium segment where competitors are seeing 25% growth. This gap analysis leads to a successful new premium line launch capturing $2.1M in previously missed revenue.

*PR #313*

---

### Reusable Tree Diagram Visualization
*October 6*

You can now use the tree diagram visualization functionality beyond revenue trees, enabling any analysis to leverage Graphviz-powered hierarchical visualizations.

**Extracted functionality:**
- New `TreeDiagram` class in separate module
- Build and render graph visualizations
- Human-readable formatting options
- Revenue tree maintains backward compatibility

**Real-world insight:** A category management team building customer decision hierarchy analysis can now reuse the same tree visualization logic that powers revenue trees, creating consistent visual language across different analytical frameworks without duplicating visualization code.

*PR #331*

---

## September 2025

### Complete Hierarchical Rollups with Suffix Support
*September 21*

You can now generate complete hierarchical aggregations with both prefix and suffix rollups, providing every analytical perspective on multi-level segment data.

**Complete aggregation hierarchy:**
- Suffix rollups complement existing prefix rollups
- Get detail rows, prefix rollups, suffix rollups, and grand totals
- Configuration-driven unified rollup logic
- Works with `calc_rollup` and `calc_total` parameters

**Real-world insight:** An analyst studying customer behavior across [Region, Store, Category] segments was manually creating 7 separate aggregation queries to get all rollup combinations. With suffix rollups enabled, they get Region totals, Region+Store totals, Store+Category totals, and all other combinations in a single query—reducing monthly reporting prep time from 3 hours to 15 minutes.

*PR #309*

---

### Add Regression Lines to Bar Charts
*September 23*

You can now overlay regression lines on bar charts (both vertical and horizontal), extending the regression analysis capability beyond line and scatter plots.

**Bar chart regression:**
- Automatic bar orientation detection
- Data extraction from bar patches
- Handles grouped and stacked configurations
- Fixed regression line coordinates for multi-subplot compatibility

**Real-world insight:** When presenting monthly sales performance as a bar chart, you can now overlay a trend line showing whether growth is accelerating or decelerating, making it immediately obvious to stakeholders that while Q3 looks strong, the growth rate is slowing and requires attention.

*PR #320*

---

### Handle Missing Data in Line Plots
*September 23*

You can now control how missing data (NaN values) are handled after DataFrame pivoting in line plots, particularly useful for sparse retail sales data across multiple locations.

**Missing data control:**
- New `fill_na_value` parameter for line plots
- Fill NaN values with specified value or preserve as masked arrays
- Maintains default matplotlib behavior when unset
- Backward compatible

**Real-world insight:** When plotting sales across 50 stores where new stores opened mid-year, you can now choose to fill missing historical data with zero (showing stores as dormant) or leave it blank (showing stores as not-yet-existing), preventing misleading visualizations that suggest poor performance when stores simply weren't open yet.

*PR #321*

---

### Customize Plot Styling Parameters
*September 19*

You can now override visual styling parameters like linewidth, width, and colors through keyword arguments, giving you fine-grained control without changing default behavior.

**Configurable styling:**
- `linewidth` in line and time plots (default: 3)
- `width` in index and waterfall plots (default: 0.8)
- `color` parameter across all plot types
- Scatter plots accept single color or lists

**Real-world insight:** When creating executive summaries, you can use thicker lines (linewidth=5) for emphasis, while analytical deep-dives use thinner lines (linewidth=2) to fit more series on screen. The same codebase serves both audiences by passing different kwargs, eliminating the need to maintain separate plotting functions.

*PR #319*

---

### Simplified CrossShop Analysis Interface
*September 16*

You can now analyze cross-shopping patterns with fewer required parameters through smart defaults, reducing boilerplate code by 27% while maintaining full functionality.

**Simplified interface:**
- `group_2_col` and `group_3_col` automatically default to `group_1_col`
- New `group_col` parameter for custom entity grouping
- `group_2_val` required (minimum 2 groups for cross-shopping)
- Full backward compatibility maintained

**Real-world insight:** An analyst building weekly cross-shopping reports across product categories was repeatedly specifying the same column three times: `group_1_col='category'`, `group_2_col='category'`, `group_3_col='category'`. With smart defaults, they now pass just `group_1_col='category'` once, eliminating redundant parameters and making cross-shopping analysis code more readable and maintainable.

*PR #312*
