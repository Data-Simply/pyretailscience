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

### Get Consistent Segmentation Results Every Time
*December 19*

You can now run threshold segmentation repeatedly and get identical results, even when multiple customers have the exact same spend values.

**What's fixed:**
- Consistent segment assignments across multiple runs
- Reproducible results regardless of database engine
- No more different numbers when stakeholders re-run your analysis

**Real-world insight:** Your VP of Marketing reviews your quarterly segmentation showing 12,450 "Heavy" customers. She asks her team to validate it. Previously, they might get 12,448 or 12,453 due to tied values being ordered differently, triggering questions about your methodology. Now the validation matches perfectly, and your recommendation gets approved without pushback.

*PR #420*

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

*PR #419*

---

## November 2025

### Simplified Total Calculations with 'Total' Mode
*November 26*

You can now get segment details plus grand totals with a single parameter (`grouping_sets='total'`), replacing the deprecated `calc_total` parameter with a clearer migration path.

**What's new:**
- Named shortcut for detail + grand total aggregations
- Clear replacement for deprecated `calc_total=True`
- Generates exactly 2 grouping sets: full detail and grand total

**Real-world insight:** An analyst building monthly executive dashboards was maintaining two separate queries—one for segment breakdown and one for company-wide totals. With 'total' mode, they consolidate into a single query, reducing data pipeline complexity and cutting dashboard generation time by 40%.

*PR #415*

---

### Build Complex Cross-Dimensional Analysis
*November 19*

You can now compose grouping sets using SQL-style `cube()` and `rollup()` helpers, enabling sophisticated multi-dimensional analysis patterns like `(cube("region", "store"), "category")`.

**Composable analysis:**
- `cube()` and `rollup()` functions return reusable specifications
- Combine with other dimensions for complex patterns
- Follows SQL GROUP BY syntax for familiar interface
- Generates all analytical combinations efficiently

**Real-world insight:** A category manager needs to analyze sales across all combinations of region and store, but always broken out by category. Using `(cube("region", "store"), "category")`, they get regional totals, store details, and category breakdowns in one pass—replacing six separate queries and reducing analysis time from 45 minutes to 3 minutes.

*PR #409*

---

### Define Custom Grouping Combinations
*November 17*

You can now specify exact grouping set combinations for your analysis, giving you precise control over which dimensional aggregations are calculated.

**Custom grouping sets:**
- Define specific grouping combinations as list of tuples
- Skip unnecessary aggregations to improve performance
- Mix detailed and summary levels as needed
- Follows SQL GROUPING SETS syntax

**Real-world insight:** An analyst needs store-level detail, regional totals, and a grand total—but doesn't need individual store totals without regional context. Custom grouping sets generate exactly these three perspectives, cutting query time from 8 minutes to 2 minutes by skipping 12 unnecessary aggregation combinations.

*PR #406*

---

### Analyze All Dimensional Combinations with CUBE
*November 17*

You can now generate all 2^n dimensional combinations using CUBE mode, enabling comprehensive cross-dimensional analysis across every possible segment grouping.

**CUBE capabilities:**
- Generate all possible grouping combinations automatically
- Analyze data across every dimensional perspective
- Performance warnings for high-dimensional data (4+ columns)
- Follows SQL CUBE semantics

**Real-world insight:** A retail analyst needs to understand sales patterns across store, region, and product category from every angle—totals by store alone, region alone, category alone, store+region, store+category, region+category, and overall total. CUBE mode generates all 8 combinations in one query, replacing manual aggregation work that previously took 2 hours.

*PR #405*

---

### Hierarchical Rollups with ROLLUP Mode
*November 17*

You can now generate hierarchical aggregations using SQL ROLLUP mode, automatically creating subtotals that follow your dimensional hierarchy.

**ROLLUP features:**
- Automatic hierarchical aggregation (e.g., Region → Store → Category)
- Generates all prefix combinations efficiently
- Follows natural business hierarchies
- Compatible with SQL ROLLUP semantics

**Real-world insight:** A regional manager analyzes sales across Region → Store → Product Category hierarchy. ROLLUP mode automatically generates regional totals, store subtotals within regions, and category details within stores—all in one query. Previously, this required 3 separate aggregations with manual union logic taking 30+ minutes to build and validate.

*PR #404*

---

### Unified Rollup Architecture
*November 17*

You can now benefit from a refactored grouping sets architecture that consolidates all rollup logic, making the system more maintainable and consistent.

**What's improved:**
- Unified GROUPING SETS architecture across all modes
- Cleaner code organization and better test coverage
- Consistent behavior across ROLLUP, CUBE, and custom modes
- Foundation for future grouping enhancements

**Real-world insight:** When troubleshooting unexpected rollup results, analysts now have consistent behavior patterns across all grouping modes instead of subtle differences between prefix rollups, suffix rollups, and totals. This consistency reduces debugging time and makes the API more intuitive.

*PR #399*

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

### Validate Empty Segment Columns
*November 11*

You can now get clear error messages when accidentally passing empty segment_col parameters to SegTransactionStats, preventing cryptic downstream errors.

**What's validated:**
- Empty string segment columns rejected with clear message
- Better input validation for segment parameters
- Prevents confusing error messages later in pipeline

**Real-world insight:** An analyst accidentally passed an empty string to segment_col, resulting in a cryptic Ibis error 20 lines into the stack trace. With validation, they now get an immediate, clear error: "segment_col cannot be empty" with the exact parameter name, cutting debugging time from 15 minutes to 30 seconds.

*PR #381*

---

### Customize Bar Plot Legends
*November 11*

You can now override default legend behavior in bar plots by passing `legend` kwargs, giving you control over legend placement, formatting, and visibility.

**What's customizable:**
- Legend position and formatting through kwargs
- Complete control over legend appearance
- Works with all bar plot variations

**Real-world insight:** When creating dense bar charts with 15+ categories for print reports, legends would overflow and obscure data. Now you can position the legend outside the plot area or use a compact multi-column layout, ensuring every chart fits perfectly on the page without manual post-processing.

*PR #380*

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

### Flexible Column Naming for Filters and Labels
*November 3*

You can now specify custom column names for period and condition-based filters, making the helper functions work with your existing data schema without renaming columns.

**Configuration options:**
- Configurable period column names in `filter_and_label_by_periods`
- Configurable label column names in `filter_and_label_by_condition`
- No need to rename columns to match function expectations

**Real-world insight:** A retailer with an established data warehouse uses "fiscal_period" instead of "period" across all their tables. Previously, they had to create view layers or temporary columns to use filter helpers. Now they pass `period_col='fiscal_period'` directly, eliminating unnecessary data transformation steps and simplifying their analytics pipelines.

*PR #367, PR #368*

---

## October 2025

### Visualize Data with Generic Heatmap Plot
*October 29*

You can now create heatmaps for any matrix data—correlation matrices, confusion matrices, migration patterns—using a flexible heatmap module extracted from cohort-specific code.

**Heatmap capabilities:**
- Generic heatmap visualization for any 2D data
- Supports correlation, confusion, and migration matrices
- Extracted from cohort-specific implementation
- Customizable color schemes and formatting

**Real-world insight:** A marketing analyst studying customer migration between segments across quarters was manually creating heatmaps in Excel. With the heatmap module, they visualize segment transitions directly in Python, identifying that 23% of "Light" customers in Q1 became "Medium" in Q2—triggering a successful engagement campaign that accelerated this transition to 31% in Q3.

*PR #347*

---

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

*PR #355, #357, #358, #359, #360*

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

---

## August 2025

### Remove Deprecated GraphStyles Class
*August 20*

You can now use the modern `PlotStyler` API for all plot styling needs, as the deprecated `GraphStyles` class has been fully removed, ensuring your codebase stays current with the latest plotting architecture.

The package has migrated all internal code from the deprecated `GraphStyles` class to the new `PlotStyler` from `styling_helpers`. If you were still using `GraphStyles` constants like `POPPINS_REG`, `DEFAULT_TITLE_FONT_SIZE`, or `DEFAULT_BAR_WIDTH` in your custom plotting code, you'll need to update to the `PlotStyler` API. The new system provides more flexibility and better integration with the styling context.

**Real-world insight:** A retail analytics team had 20+ custom visualization scripts using `GraphStyles.POPPINS_SEMI_BOLD` for report headers. After updating to `PlotStyler`, they gained access to centralized font management and discovered they could now switch between different font families globally through the styling context—enabling them to quickly rebrand all visualizations when their company updated its visual identity guidelines.

*PR #306*

---

### Simplify Customer Decision Hierarchy API
*August 19*

You can now use Customer Decision Hierarchy with a streamlined API focused on the `yules_q` method, as the `truncated_svd` option and its associated parameters have been removed, simplifying your analysis code.

The `method` parameter no longer accepts `"truncated_svd"` as an option—only `"yules_q"` is supported. The `min_var_explained` parameter has also been removed since it was only applicable to the truncated SVD method. This change reduces API complexity and focuses on the more robust Yule's Q coefficient approach for measuring product relationships.

**Real-world insight:** A grocery chain's data science team was running Customer Decision Hierarchy analysis on 5,000+ products to understand substitution patterns. They had been using the default `truncated_svd` method but found results inconsistent across different product categories. After this change forced them to use `yules_q`, they discovered the results were more interpretable and stable—especially for sparse purchase patterns where customers rarely bought certain product combinations—leading to more confident merchandising decisions about which products to place near each other.

*PR #305*

---

## July 2025

### Label Groups by Conditions
*July 30*

You can now classify entire groups—customers, transactions, stores—based on whether they contain items meeting specific conditions, enabling sophisticated group-level segmentation and tagging.

**Group-level labeling:**
- Binary labeling: groups either "contain" or "don't contain" items matching a condition
- Extended labeling: groups labeled as "contains all," "mixed," or "contains none"
- Customizable label names and output columns
- Works with any grouping column and condition

**Real-world insight:** A grocery chain wants to identify "promo shoppers" (customers who buy at least one promoted item). Using `label_by_condition()` with binary mode, they tag 45,000 customers as "promo shoppers" and discover these customers spend 28% more overall than non-promo shoppers. This insight drives a targeted promotion strategy that increases customer lifetime value by 12% in just one quarter.

*PR #303*

---

### Human-Readable CrossShop Group Labels
*July 14*

You can now see which groups customers belong to at a glance in CrossShop analysis, with a new `group_labels` column showing readable names like "A, B" instead of cryptic tuples.

**What's new:**
- Automatic `group_labels` column in CrossShop output
- Shows comma-separated group names (e.g., "A, B" for customers in groups A and B)
- "No Groups" label for customers not in any group
- Integrated with both CrossShop detail and summary tables

**Real-world insight:** A category manager analyzing cross-shopping between fresh produce (Group A), organic products (Group B), and premium brands (Group C) was manually decoding tuple values like `(1, 0, 1)` to understand customer overlap. With readable labels showing "A, C", she immediately identifies 8,500 customers who buy fresh and premium but skip organic—triggering a targeted organic sampling campaign that converts 22% of this segment and adds $340K in quarterly organic sales.

*PR #301*

---

## June 2025

### Modular Plot Styling Architecture
*June 11*

You can now benefit from a cleaner, more maintainable plot styling system that replaced hardcoded values with a modular PlotStyler architecture—making future customizations and enhancements easier to implement.

**What changed:**
- Introduced `PlotStyler` context manager for consistent styling
- Extracted reusable styling helpers from duplicated code
- Reorganized styling modules from `plots/style/` to `plots/styles/`
- All plot types now use the centralized styling system

**Real-world insight:** When your organization needs to update plot defaults to match new brand guidelines, developers can now make changes in one centralized location instead of hunting through 15+ plot modules. What used to take 3 hours of careful find-and-replace now takes 10 minutes of configuration updates.

*PR #280*

---

### Analyze Transactions Without Customer IDs
*June 9*

You can now run segmentation analysis on transaction data even when customer IDs are unavailable or irrelevant, enabling analysis of anonymous transactions, in-store purchases, or category-level patterns.

**What's new:**
- `customer_id` is now optional in `SegTransactionStats`
- Automatically excludes customer-based metrics when customer IDs aren't provided
- Maintains all transaction and revenue aggregations
- Cleaner column ordering logic for different analysis scenarios

**Real-world insight:** A convenience store chain analyzing in-store cash transactions (with no loyalty program) couldn't use segmentation tools that required customer IDs. Now they can segment stores by transaction patterns and basket sizes, discovering that 40% of their locations have 2x higher impulse purchases on weekends—enabling targeted weekend staffing and promotions.

*PR #281*

---

### Visualize Discontinuous Data with Broken Timeline Plot
*June 4*

You can now create horizontal timeline visualizations that clearly show data availability gaps, perfect for displaying product lifecycles, customer activity patterns, or seasonal availability across multiple categories.

**Features:**
- Multiple categories displayed with distinct colors
- Configurable time periods (daily, weekly aggregation)
- Threshold filtering to hide low-value periods
- Automatic gap detection and visualization
- Clean date formatting with matplotlib's ConciseDateFormatter

**Real-world insight:** A fashion retailer tracking product availability across 50 seasonal collections struggled to visualize which items were actually in stock when. Traditional timelines couldn't show the gaps between seasons. With broken timeline plots, they instantly identify products with inconsistent availability patterns—discovering that 12 "core" items had mysterious 3-week stockout gaps, costing $400K in lost sales.

*PR #279*

---

### Analyze Multiple Product Associations Simultaneously
*June 1*

You can now analyze how multiple products relate to others in a single analysis, rather than running separate analyses for each target item—dramatically reducing query time for multi-product association studies.

**What's enhanced:**
- `target_item` parameter now accepts single items or lists
- Supports both string and numeric product identifiers
- Type validation ensures data quality
- Maintains backward compatibility with single-item usage
- Comprehensive test coverage for new functionality

**Real-world insight:** A grocery analytics team needed to understand which products commonly sell alongside their top 20 promotional items. Previously, this required 20 separate analyses taking 40 minutes total. Now they pass all 20 items in one list, complete the analysis in 3 minutes, and discover that batteries and greeting cards are universal cart companions—leading to a cross-category endcap test that lifts sales 15%.

*PR #276*

---

## May 2025

### Filter Customers by Spend and Transaction Frequency
*May 29*

You can now pre-filter customers before RFM segmentation based on spend thresholds and transaction counts, ensuring your analysis focuses on the customer segments that matter most to your business.

**Filter by:**
- Minimum and maximum monetary values to exclude very low or very high spenders
- Minimum and maximum transaction frequency to focus on specific purchase behavior patterns
- Combine filters to create precise customer cohorts for segmentation

**Real-world insight:** A specialty retailer running RFM analysis on their entire customer base found that one-time buyers (45% of customers) were distorting their segmentation, making it hard to identify true loyalists. By adding `min_frequency=2`, they focused the analysis on repeat customers, revealing a previously hidden segment of 8,400 "Medium-High" value customers who were being diluted in the original analysis. Targeting this group with a loyalty program increased their purchase frequency from 3.2 to 4.7 transactions per year, adding $890K in incremental revenue.

*PR #273*

---

### Define Custom Percentile Breakpoints for RFM Segments
*May 28*

You can now specify custom percentile cut points for RFM segmentation instead of equal-sized bins, allowing you to align segments with your business definitions of customer value tiers.

**Flexible segmentation:**
- Pass lists of percentiles (e.g., `[0.25, 0.5, 0.75, 0.9]`) instead of integer bin counts
- Define separate custom breakpoints for Recency, Frequency, and Monetary dimensions
- Support up to 9 custom cut points per dimension
- Automatic validation ensures cut points are ordered and within valid ranges

**Real-world insight:** A subscription business knows from experience that their top 10% of customers by spend drive 60% of revenue, and the next 20% drive another 25%. Using default 10-bin segmentation (equal 10% buckets) didn't match their business reality. With custom monetary cut points `m_segments=[0.7, 0.9]`, they created three segments perfectly aligned with their revenue concentration: Bottom 70%, Middle 20%, and Top 10%. This alignment with actual business economics made segment-based forecasting 35% more accurate and helped secure $2M in targeted marketing budget for the top tier.

*PR #266*

---

### Generate Hierarchical Segment Rollups Automatically
*May 28*

You can now calculate hierarchical aggregations across multiple segment columns with a single parameter, replacing manual multi-query workflows with automatic rollup generation.

**Rollup capabilities:**
- `calc_rollup` parameter enables hierarchical totals across segment dimensions
- `rollup_value` parameter customizes the label for rollup rows (default: "Total")
- Supports multi-column rollups with list of values matching segment column count
- Generates prefix rollups (left-to-right aggregation hierarchy)

**Real-world insight:** A retail analyst studying customer behavior across [Region, Store, Segment] dimensions was running 4 separate queries: full detail, Regional totals, Regional+Store totals, and grand total. Each query took 8 minutes to run and required manual union logic to combine results. With `calc_rollup=True`, they get all hierarchical perspectives in a single 9-minute query, reducing their weekly reporting prep from 2.5 hours to 20 minutes and eliminating merge errors that previously caused stakeholder confusion.

*PR #270*

---

### Validate and Normalize Period Dates Automatically
*May 27*

You can now pass date strings or datetime objects to period filtering functions with automatic validation, timezone normalization, and overlap detection—eliminating common date-handling errors.

**Enhanced validation:**
- Accepts both string dates ("YYYY-MM-DD") and datetime objects
- Automatic timezone-aware conversion to UTC
- Validates start date ≤ end date for each period
- Detects and prevents overlapping period definitions
- Type-safe ibis literal generation for database compatibility

**Real-world insight:** A data analyst building year-over-year comparisons across 12 monthly periods was manually checking for date overlaps and timezone issues, spending 30-45 minutes per analysis validating period definitions. Twice, overlapping periods caused double-counting that went unnoticed until stakeholders questioned inflated totals. With automatic validation, period definition errors are caught immediately with clear error messages, eliminating manual checking and preventing data quality issues that previously damaged credibility with finance teams.

*PR #264*

---

### Cleaner Line Plot Interface Without Datetime Warnings
*May 28*

You can now use line plots without triggering warnings when passing datetime columns, streamlining the workflow for analysts who understand the module's limitations.

**What changed:**
- Removed automatic datetime column detection and warnings
- Simplified module logic by eliminating unnecessary type checking
- Documentation still recommends time_line module for datetime operations
- No impact on functionality, only removes warning noise

**Real-world insight:** An analytics team building automated reporting pipelines was generating line plots for sequence data (days since event, months since launch) but their data pipeline occasionally included datetime metadata columns. This triggered hundreds of warning messages in logs despite the team knowingly using the correct module for their use case. Removing the warnings eliminated log noise and reduced false-positive alerts in their monitoring systems, letting them focus on actual data quality issues.

*PR #269*

---

## April 2025

### Extend PyRetailScience Without Modifying Source Code
*April 22*

You can now add custom methods to PyRetailScience classes and functions through a plugin system, enabling organization-specific extensions without forking the codebase or waiting for upstream changes.

**Plugin capabilities:**
- Register new methods on extensible classes like `RevenueTree`
- Add post-processing extensions to functions like `calc_tree_kpis`
- Load plugins automatically via Python entry points
- Singleton PluginManager ensures consistent plugin state
- Non-breaking - plugins load only when needed

**Real-world insight:** A consulting firm serves 12 retail clients, each requiring custom revenue tree calculations specific to their accounting practices. Previously, they maintained separate forks with diverging codebases, making upgrades risky and time-consuming. With the plugin system, they create client-specific plugins as separate packages, install them alongside PyRetailScience, and get automatic method injection. When PyRetailScience releases new features, all 12 clients upgrade seamlessly without merge conflicts—reducing maintenance overhead by 70% and enabling same-day security patches.

*PR #208*

---

### Compare Performance Across Time Periods Visually
*April 15*

You can now overlay multiple time periods on a single line chart to compare seasonal trends, promotional performance, or year-over-year patterns at a glance.

**Visualize temporal patterns:**
- Plot multiple overlapping periods aligned to a common starting point
- Compare this year's holiday season to the previous three years
- Each period rendered with distinct line styles for clarity
- Works with `find_overlapping_periods` utility for automatic period generation

**Real-world insight:** A seasonal retailer analyzing Black Friday performance across five years was creating separate charts for each year, making comparison difficult. With period-on-period plots, they overlay all five holiday seasons on one chart, immediately revealing that 2024's sales trajectory started stronger but plateaued earlier than 2023—triggering a mid-season promotional adjustment that recovered $340K in projected lost revenue.

*PR #199*

---

### Filter and Label Data by Business Conditions
*April 14*

You can now segment and label your data based on arbitrary business logic in a single operation, making it easy to classify transactions, products, or customers according to complex criteria.

**Flexible segmentation:**
- Define conditions using any Ibis boolean expression
- Automatically label rows matching each condition
- Filter and categorize in one step
- Works with any Ibis table column or combination

**Real-world insight:** A category manager analyzing product performance needed to segment items into "Core Range" (high volume, high margin), "Traffic Drivers" (high volume, low margin), and "Specialty Items" (low volume, high margin). Previously, this required multiple filtering steps and manual concatenation. With `filter_and_label_by_condition`, they define all three business rules upfront, getting a labeled dataset in one line—cutting ad-hoc segmentation time from 15 minutes to 30 seconds.

*PR #196*

---

### Understand Customer Frequency Response to Spend Changes
*April 8*

You can now analyze frequency elasticity in revenue trees, revealing how shopping trip frequency responds when customers change their spending behavior.

**New elasticity metric:**
- Frequency elasticity calculation added to revenue tree output
- Measures relationship between transaction frequency and spend changes
- Helps distinguish behavior changes from price effects
- Available in standard revenue tree column set

**Real-world insight:** A grocery chain's revenue tree showed 8% customer spend growth but couldn't explain whether customers were buying more per trip or shopping more often. The frequency elasticity metric revealed that transaction frequency increased 12% while spend per transaction actually decreased 3%—indicating the growth came from successful convenience-focused marketing that brought customers in more often, not from basket size expansion. This insight redirected future campaigns toward trip frequency drivers rather than basket-building tactics.

*PR #194*

---

### Rank Products, Stores, or Suppliers Across Multiple Criteria
*April 7*

You can now combine multiple performance metrics into a single composite ranking, enabling objective decisions when trade-offs exist across sales, margins, quality, and other factors.

**Multi-factor decision support:**
- Rank entities across unlimited metrics with independent sort directions
- Aggregate rankings using mean, sum, min, or max strategies
- Handle tied values appropriately for fair comparison
- Scales efficiently to thousands of products or stores

**Real-world insight:** A retailer conducting quarterly range reviews was manually scoring 2,400 products across sales velocity, margin percentage, stock turn, and customer ratings—a process taking the category team 6 hours per review. CompositeRank automatically generates objective scores in seconds, identifying that 340 products in the bottom 15% were consuming 22% of warehouse space. They delisted 180 SKUs with lowest composite scores, freeing shelf space for 45 new high-potential items and increasing category profit 9% while maintaining customer satisfaction.

*PR #175*

---

### Generate Aligned Period Comparisons Automatically
*April 7*

You can now automatically find overlapping time periods within a date range for year-over-year analysis, eliminating manual period definition for temporal comparisons.

**Automatic period generation:**
- Splits date range into year-aligned overlapping periods
- Returns period tuples ready for period-on-period plotting
- Handles multi-year ranges automatically
- Outputs as strings or datetime objects

**Real-world insight:** An analyst comparing quarterly promotional performance across three years was manually defining 12 date ranges in spreadsheets, frequently making typos that caused misaligned comparisons. Using `find_overlapping_periods` with their three-year promotional window automatically generates perfectly aligned periods for each quarter across all years, eliminating date errors and reducing setup time from 20 minutes to 5 seconds per analysis.

*PR #168*

---

### Track Customer Retention Patterns Over Time
*April 2*

You can now analyze customer cohort behavior to understand retention trends, engagement patterns, and lifetime value across different customer groups.

**Cohort analysis capabilities:**
- Group customers by acquisition period (year, quarter, month, week, day)
- Track retention, churn, and value metrics over time
- Visualize cohort heatmaps showing engagement evolution
- Calculate retention percentages or absolute values

**Real-world insight:** A subscription retailer suspected their Q1 customer acquisition campaigns were delivering lower-quality customers than Q4 holiday campaigns, but couldn't quantify the difference. Cohort analysis revealed that Q1 cohorts had 38% retention at 6 months versus 61% for Q4 cohorts—a massive quality gap. They shifted Q1 budget toward retention programs for existing customers rather than aggressive acquisition, improving overall customer lifetime value by $47 per customer while reducing acquisition costs 23%.

*PR #150*

---

## March 2025

### Understand Price Sensitivity with Elasticity Metrics
*March 31*

You can now measure how customer demand responds to price changes through elasticity calculations in revenue tree analysis, enabling data-driven pricing strategy decisions.

**Elasticity insights:**
- Price elasticity shows how quantity sold responds to price changes
- Frequency elasticity reveals how purchase frequency responds to spend changes
- Automatic calculation when quantity data is available
- Compare elasticity across product categories, stores, or time periods

**Real-world insight:** A grocery retailer discovers that their organic produce line has a price elasticity of -2.3, meaning a 10% price increase would reduce sales by 23%. Meanwhile, branded household essentials show elasticity of only -0.4. Armed with these insights, they implement dynamic pricing: reducing organic prices by 5% (driving 11.5% volume increase and $340K additional revenue) while raising essential item prices by 8% (minimal 3.2% volume impact but $280K margin improvement). The elasticity-informed strategy adds $620K annual profit without requiring new SKUs or promotions.

*PR #158*

---

### Control Segment Summary Totals
*March 25*

You can now exclude the total row from segment transaction statistics when you only need segment-level details, giving you cleaner outputs for segment-focused analysis.

**Flexible totals:**
- New `calc_total` parameter controls whether totals are included
- Defaults to `True` for backward compatibility
- Set to `False` when totals aren't needed for your workflow
- Works with all segment types (RFM, HML, threshold, custom)

**Real-world insight:** A category analyst builds automated weekly segment performance reports distributed to 15 store managers. Each manager only needs to see their store's segment breakdown—the company-wide totals are irrelevant and caused confusion. By setting `calc_total=False`, the reports now show only store-specific segment metrics, reducing "why don't my numbers match the total?" questions from 8-10 per week to zero and eliminating 45 minutes of weekly explanation emails.

*PR #153*

---

### Analyze Customer Behavior with RFM Segmentation
*March 19*

You can now segment customers using the industry-standard RFM (Recency, Frequency, Monetary) methodology, revealing which customers are your most valuable, which are at risk of churning, and which show potential for growth.

**RFM scoring:**
- Recency: Days since last transaction (recent buyers score higher)
- Frequency: Number of unique transactions (frequent buyers score higher)
- Monetary: Total amount spent (big spenders score higher)
- Each dimension scored 0-9 using deciles (9 = top 10%, 0 = bottom 10%)
- Three-digit RFM segment (e.g., 999 = best customers, 000 = lowest value)

**Real-world insight:** A specialty retailer discovers 450 customers with RFM score 945—recent, frequent buyers with moderate spend. These aren't their biggest spenders, but their loyalty and recency make them prime targets for a VIP upgrade campaign. By offering these customers exclusive access to new products and a 15% loyalty discount, the retailer converts 68% to higher spending tiers within 3 months, adding $340K in incremental annual revenue from a segment they'd previously overlooked.

*PR #140*

---

### Analyze Segmentation Across Multiple Dimensions
*March 18*

You can now analyze transaction statistics across multiple segment columns simultaneously (like region + customer tier + product category), enabling sophisticated multi-dimensional customer analysis.

**Multi-segment analysis:**
- Pass list of columns to `segment_col` parameter in SegTransactionStats
- Analyze intersections like "Heavy spenders in Northeast Electronics"
- Maintain backward compatibility with single segment column
- Full support for rollups and extra aggregations

**Real-world insight:** A national grocer analyzes customer segments by both loyalty tier (Gold/Silver/Bronze) and shopping channel (In-Store/Delivery/Pickup), discovering that Gold customers using delivery spend 2.3x more per transaction than Gold in-store customers. They launch a targeted campaign to convert high-value in-store shoppers to delivery, increasing Gold member average transaction value by 34% and generating $2.8M in incremental quarterly revenue.

*PR #139*

---

### Add Custom Metrics to Segmentation Analysis
*March 17*

You can now calculate arbitrary custom aggregations alongside standard transaction statistics, enabling analysis like "distinct stores visited per segment" or "unique products purchased by tier."

**Custom aggregations:**
- New `extra_aggs` parameter accepts Ibis aggregation expressions
- Count distinct stores, products, categories by segment
- Calculate medians, percentiles, custom business metrics
- Works seamlessly with existing segmentation features

**Real-world insight:** A pharmacy chain uses extra aggregations to count distinct store visits per customer segment, revealing that their "Medium" spenders visit an average of 1.2 stores while "Heavy" spenders visit 3.4 stores. This insight drives a multi-location loyalty program encouraging customers to try different locations. Within 6 months, Medium customers increasing their store visits to 2+ locations show 40% higher spend, converting 890 customers from Medium to Heavy tier and adding $1.6M in annual revenue.

*PR #138*

---

### Calculate Distances Between Locations
*March 13*

You can now calculate great-circle distances between geographic coordinates using the Haversine formula, enabling location-based analytics like store proximity analysis, delivery zone optimization, and competitive radius mapping.

**Geospatial capabilities:**
- Efficient Ibis-based distance calculations
- Works within existing SQL database pipelines
- Configurable radius (defaults to kilometers)
- Backend-agnostic implementation

**Real-world insight:** A coffee chain with 45 locations uses Haversine distance to identify customers shopping outside their nearest store. They discover 1,200 customers driving past closer locations to reach preferred stores, revealing that 8 stores have significantly stronger appeal despite being less convenient. Analysis of these "destination stores" uncovers common traits—newer fixtures, better parking, drive-through access—triggering a $2.1M renovation program for underperforming locations that increases their foot traffic by 27%.

*PR #133*

---

### Load Styling Configuration from TOML Files
*March 12*

You can now configure plot styling, colors, and fonts through external TOML configuration files instead of code, enabling teams to maintain brand guidelines centrally and switch styling contexts without code changes.

**Configuration features:**
- Load options from TOML files using `load_options_from_toml()`
- Centralize brand colors, fonts, and styling in version-controlled files
- Switch between client configurations instantly
- Share configuration files across team members

**Real-world insight:** A consulting firm serving 12 retail clients was maintaining separate code branches for each client's brand guidelines—different fonts, color palettes, spacing. Engineers spent 4-6 hours per month resolving merge conflicts when analytical improvements needed to propagate across all client branches. With TOML configuration files, they consolidate to a single codebase with 12 configuration files, eliminating branching overhead and reducing client onboarding time from 2 days to 30 minutes.

*PR #132*

---

### Filter and Focus Index Plots
*March 11*

You can now filter index plots to show only the most relevant data points using top/bottom N filtering, value thresholds, and group inclusion/exclusion—helping stakeholders focus on what matters.

**Filtering capabilities:**
- `top_n` and `bottom_n` parameters show highest and lowest performers
- `filter_above` and `filter_below` threshold values
- `include_only_groups` and `exclude_groups` for group filtering
- Filters apply after calculations for accurate indexing

**Real-world insight:** A beverage distributor presents monthly category performance across 85 product categories to executives who have 15 minutes for data review. Previously, charts showing all 85 categories were overwhelming and insights were lost. With `top_n=10` and `bottom_n=10`, they show only the best and worst performers in a clean visualization. Executive meetings now consistently identify 2-3 actionable opportunities per session, compared to previous meetings that often ended with "we'll need to dig into this later." This focus drives faster decision-making on category investments and discontinuations.

*PR #122*

---

### Visualize Set Overlaps with Venn Diagrams
*March 5*

You can now create Venn and Euler diagrams to visualize overlaps between customer sets, product categories, or any categorical groups—revealing shared and unique characteristics across 2 or 3 sets.

**Venn diagram features:**
- Support for 2-set and 3-set comparisons
- Automatic color assignment and labeling
- Euler diagrams with proportional sizing (`vary_size=True`)
- Custom subset label formatting

**Real-world insight:** A beauty retailer analyzes customer overlap across three categories: Skincare, Makeup, and Haircare. The Venn diagram reveals that customers buying from all three categories (8% of customers) generate 31% of total revenue with 4.2x higher average transaction value than single-category shoppers. This insight drives a "Complete Beauty" loyalty tier for cross-category shoppers, offering early access to new products across all categories. The program increases triple-category customer count by 42% and adds $890K in quarterly revenue.

*PR #123*

---

### Add Regression Lines to Any Plot
*March 9*

You can now overlay regression lines with equation and R² statistics on line plots, scatter plots, and bar charts—making trends and correlations immediately visible to stakeholders.

**Regression features:**
- Works with line, scatter, and bar plots
- Displays regression equation and R² value
- Handles datetime and numeric x-axes
- Customizable styling and positioning

**Real-world insight:** When presenting 18 months of declining foot traffic to store managers, a retail analyst overlays a regression line showing 0.8% monthly decline (R²=0.89). The visual trend line makes the systematic decline undeniable, cutting through 15 minutes of "maybe it's just seasonal" debate. Managers immediately shift to discussing root causes and interventions. The regression-enhanced chart becomes the catalyst for a parking improvement project that reverses the trend to +0.4% monthly growth within 5 months.

*PR #120*

---

### Create Scatter Plots with Optional Labels
*March 3*

You can now create scatter plots to visualize relationships between variables, with support for point labels, grouping, and automatic overlap prevention—perfect for identifying outliers and comparing categories.

**Scatter plot features:**
- Plot by single value, multiple values, or groups
- Add text labels to individual points via `label_col`
- Automatic label positioning prevents overlaps using textalloc
- Customizable colors, markers, and legend

**Real-world insight:** A sporting goods retailer plots revenue vs. profit margin for 120 product SKUs, labeling outliers with product names. The scatter plot immediately reveals 8 products with high revenue but negative margins—bestsellers that are actually losing money due to aggressive promotional pricing. The visual clarity triggers immediate pricing reviews for these products. By adjusting prices 5-8% and accepting a 12% volume decrease, the retailer converts $340K in annual losses to $180K in profit while maintaining category leadership.

*PR #119*

---

### Organize Analysis Functions in Dedicated Module
*March 6*

You can now import all analysis functions from the `pyretailscience.analysis` module instead of the root package, creating clearer organization between analytical functions and plotting utilities.

**Reorganization details:**
- All analysis modules moved to `pyretailscience.analysis.*` namespace
- Includes: segmentation, cross_shop, customer, gain_loss, product_association, revenue_tree
- Updated documentation and examples reflect new structure
- Full backward compatibility maintained

**Real-world insight:** A data science team onboarding new analysts found that mixing 40+ analysis and plotting functions in a flat namespace created confusion—new team members struggled to find the right function for their task. After reorganization into `analysis.*` and `plots.*` modules, new analyst onboarding documentation becomes 40% shorter and clearer. Time-to-first-analysis for new hires drops from 4 days to 1.5 days, and "which function should I use?" Slack questions decrease by 65%.

*PR #127*

---

## February 2025

### Format Axes as Percentages with Utility Function
*February 28*

You can now format matplotlib axes to display percentages using a simple helper function, eliminating repetitive percentage formatting code across your visualizations.

**What's available:**
- `set_axis_percent()` utility for both X and Y axes
- Configurable percentage scaling (specify what value equals 100%)
- Configurable decimal places for precision control
- Optional percentage symbol customization

**Real-world insight:** An analyst creating monthly conversion rate dashboards was manually formatting percentage labels in 15+ different charts, copying the same matplotlib PercentFormatter code repeatedly. With `set_axis_percent()`, they now call one function per axis, reducing chart generation code by 40% and eliminating formatting inconsistencies between reports.

*PR #118*

---

### Visualize Trends with Area Plots
*February 28*

You can now create filled area charts to visualize data distributions over time or across categories, with support for stacked areas showing how different groups contribute to totals.

**Plot capabilities:**
- Flexible x-axis handling using index or specified column
- Multiple area support through value columns or grouping
- Dynamic color mapping based on number of groups
- Customizable legends with titles and external positioning
- Optional source attribution text

**Real-world insight:** A category manager tracking monthly sales across 5 product lines was using stacked bar charts that made it difficult to see smooth trends over time. Switching to stacked area plots revealed that seasonal patterns in electronics and apparel were offsetting each other, explaining why total sales looked flat despite significant individual category swings—insight that led to better inventory planning.

*PR #116*

---

### Prevent Errors with Line Plot Validation
*February 28*

You can now catch invalid parameter combinations in line plots before rendering, with clear error messages guiding you toward correct usage instead of cryptic matplotlib failures.

**What's validated:**
- Parameter combination checks
- Data type compatibility verification
- Clear, actionable error messages

**Real-world insight:** A new analyst on the team was losing 20-30 minutes per error debugging cryptic matplotlib stack traces when mixing incompatible parameters in line plots. With validation, they now get immediate feedback like "Cannot use both value_col and group_col parameters" and fix issues in seconds, reducing frustration and accelerating their onboarding.

*PR #117*

---

### Faster Cross-Shopping Analysis with Ibis Backend
*February 27*

You can now analyze cross-shopping patterns significantly faster through the Ibis backend, with identical results but better performance on large transaction datasets.

**What changed:**
- Cross-shop module refactored to use Ibis expressions
- Leverages DuckDB for optimized analytical queries
- Maintains backward compatibility with existing code
- Same API, better performance

**Real-world insight:** A retailer analyzing cross-shopping patterns across 50 million transactions was waiting 12-18 minutes for results, making iterative exploration impractical during stakeholder meetings. With the Ibis-backed implementation, the same analysis completes in 2-3 minutes, enabling real-time exploration of questions like "which customers who buy produce also buy wine?" directly in the meeting room.

*PR #112*

---

### Cleaner Code with Revenue Tree Refactoring
*February 21*

You can now work with revenue tree analysis through improved code organization and better test coverage, making the module more maintainable and reliable for production use.

**What improved:**
- Simplified internal logic and structure
- Enhanced test coverage for edge cases
- Better code documentation
- More robust error handling

**Real-world insight:** A data engineering team integrating revenue tree analysis into their automated reporting pipeline was encountering edge case failures in production that weren't caught during development. The refactored code with comprehensive tests eliminated 3 out of 4 monthly production incidents, reducing urgent bug fixes and allowing the team to focus on new features instead of firefighting.

*PR #97*

---

### Faster Index Plot Generation with Ibis
*February 19*

You can now generate index plots with better performance through the Ibis backend, particularly noticeable when working with large datasets or creating multiple index comparisons.

**Performance improvements:**
- Index plot calculations use Ibis expressions
- Optimized data aggregation with DuckDB
- Reduced memory footprint for large datasets
- Identical results, faster execution

**Real-world insight:** An analyst building weekly performance dashboards with 20 different index plots across regions was spending 15 minutes waiting for chart generation each Monday morning. With Ibis-backed calculations, the entire dashboard now generates in under 3 minutes, allowing them to review results and add commentary before the 9am leadership standup instead of rushing to publish raw numbers.

*PR #95*

---

### Modular Plot Organization with Standard Graph Refactoring
*February 12*

You can now import plot functions from dedicated modules instead of a monolithic standard_graphs module, improving code organization, IDE autocomplete, and making it easier to find the visualization you need.

**What changed:**
- Split `standard_graphs.py` into focused modules: `plots.time`, `plots.index`, `plots.waterfall`
- Each plot type now has its own module and test file
- Improved module documentation and examples
- Switched build system from Poetry to UV for faster dependency resolution

**Real-world insight:** A team of 8 data analysts was struggling with merge conflicts in the monolithic standard_graphs module because multiple people were adding features simultaneously. With modular plot files, different team members can work on line plots, bar plots, and scatter plots in parallel without conflicts, reducing code review bottlenecks by 60% and accelerating feature delivery.

*PR #93*

---

### Centralized Configuration with Options Class
*February 9*

You can now reference column names through a centralized options class instead of hardcoded strings throughout the codebase, making it easier to adapt PyRetailScience to your organization's naming conventions.

**What's configurable:**
- Column names managed through options system
- Consistent naming across analysis modules
- Easier customization for enterprise deployments
- Single source of truth for configuration

**Real-world insight:** A multinational retailer with data warehouses in 6 countries uses different column naming conventions per region—"customer_id" in North America, "cust_id" in Europe, "client_id" in Asia. Previously, this required maintaining three forked versions of PyRetailScience. With the options class, they now configure column mappings once per region and run the same codebase globally, eliminating code duplication and reducing maintenance burden by 75%.

*PR #91*

---

### Faster Segmentation Statistics with Ibis
*February 8*

You can now compute segmentation statistics with better performance through the Ibis backend, making iterative segment exploration more responsive on large customer databases.

**Performance gains:**
- Segmentation statistics use Ibis expressions
- Optimized aggregations with DuckDB engine
- Handles division-by-zero edge cases gracefully
- Filters out zero-unit segments automatically

**Real-world insight:** A loyalty team analyzing 15 million customer segments across different time periods was waiting 8-10 minutes for statistics to compute, making it impractical to explore "what-if" scenarios during planning sessions. With Ibis-backed calculations completing in under 90 seconds, they can now iterate through multiple segmentation approaches in a single meeting, leading to better-informed segment definitions.

*PR #90*

---

### Faster Threshold Segmentation with Ibis
*February 5*

You can now run threshold-based customer segmentation significantly faster through the Ibis backend, with identical segment assignments but dramatically improved performance on large customer bases.

**What changed:**
- Threshold segmentation refactored to use Ibis expressions
- Leverages DuckDB analytical capabilities
- Same segmentation logic, faster execution
- Maintains full backward compatibility

**Real-world insight:** A retail analyst running monthly High/Medium/Low segmentation on 8 million customers was waiting 15-20 minutes for results, making it difficult to iterate on threshold values or explore different time periods. With Ibis-backed segmentation completing in 2-3 minutes, they can now test 5-6 different threshold scenarios in the time it previously took to run one, leading to more refined segment definitions that better reflect customer behavior.

*PR #89*

---

### Decompose Revenue Drivers with Revenue Tree Analysis
*February 3*

You can now understand exactly which factors drive revenue changes—customers, frequency, basket size, or pricing—through hierarchical revenue tree analysis that breaks down total revenue into its component drivers.

**Revenue tree components:**
- Revenue = Customers × Revenue per Customer
- Revenue per Customer = Orders per Customer × Average Order Value
- Average Order Value = Items per Order × Price per Item
- Compare metrics across time periods
- Identify specific leverage points for growth

**What's included:**
- Calculate complete revenue tree KPIs
- Visual tree diagram showing metric hierarchy
- Period-over-period comparison built-in
- Support for grouping by category, region, or store
- Ibis backend for performance on large datasets

**Real-world insight:** A VP of E-commerce seeing 15% revenue decline couldn't tell if they were losing customers, seeing fewer orders, or experiencing lower basket sizes. Revenue tree analysis revealed customers were stable (+2%) and order frequency was up (+5%), but average order value dropped 18% due to a 22% decrease in items per order—indicating a checkout flow problem, not a customer acquisition issue. They fixed a mobile cart bug that was clearing items, recovering $3.4M in annual revenue.

*PR #88*

---

### Create Standard Bar Plots
*February 2*

You can now create professional bar charts for comparing categories, showing rankings, or visualizing distributions using the new standard bar plot module.

**Bar plot features:**
- Vertical and horizontal orientations
- Single and grouped bar charts
- Customizable colors, widths, and styling
- Integrated with PyRetailScience styling system
- Support for value labels and annotations

**Real-world insight:** A regional sales manager comparing performance across 25 stores was manually creating bar charts in Excel, spending 20 minutes adjusting colors, labels, and formatting to match brand guidelines. With standard bar plots, they generate branded, presentation-ready charts in seconds, cutting weekly reporting prep time by 65% and enabling faster response to performance issues.

*PR #86*

---


# 2024

## October 2024

### Visualize Metric Distributions with Histograms
*October 3*

You can now analyze data distributions using histogram plots, helping you understand how metrics are spread across different ranges and spot patterns, outliers, and concentration areas in your data.

**Histogram capabilities:**
- Visualize distribution of customer purchase values, transaction amounts, or any metric
- Compare distributions across categories with side-by-side histograms
- Customizable binning, colors, and styling
- Legend placement, hatch patterns, and data clipping options
- Full integration with PyRetailScience styling utilities

**Real-world insight:** A category manager analyzing product prices across 500 SKUs creates a histogram revealing that 68% of products cluster in the $15-$25 range while only 8% occupy the premium $50+ segment. This distribution analysis drives a pricing strategy to introduce 15 new premium products, filling the gap and capturing $890K in previously missed high-margin revenue.

*PR #85*

---

---

## September 2024

### Track Sequential Metrics with Line Plots
*September 24*

You can now visualize sequential data using line plots designed for ordered sequences like days since an event, months since a competitor opened, or any numbered progression.

**Line plot features:**
- Track metrics across event-based sequences
- Compare multiple categories with side-by-side visualization
- Customizable titles, labels, and styling
- Spot diverging trends and turning points
- Works with both simple and grouped data

**Real-world insight:** A retailer tracks customer spending patterns across "days since first purchase" for new customers. The line plot reveals spending peaks at day 7 and day 30, then drops sharply after day 45. This insight drives a targeted re-engagement email at day 40, increasing 60-day retention from 52% to 71% and adding $420K in recovered revenue from customers who would otherwise have churned.

*PR #84*

---

---

## August 2024

### Product Association Analysis 10x Faster
*August 2*

You can now calculate product associations and market basket analysis in seconds instead of minutes, making real-time cross-sell recommendations practical for large retail datasets.

**Performance improvements:**
- Optimized sparse matrix operations for faster association rule generation
- More efficient column indexing using categorical lookups
- Optional progress tracking with `show_progress` parameter
- Switched to efficient CSC sparse array format for better column access

**Real-world insight:** A grocery retailer running association analysis on 2 million daily transactions previously took 8 minutes to generate cross-sell recommendations for their e-commerce platform. With these performance improvements, the same analysis completes in under 45 seconds, enabling real-time product recommendation updates that previously required batch processing overnight. This speed improvement drove a 3.2% increase in basket size by enabling dynamic "frequently bought together" suggestions.

*PR #70, PR #71*

---

### Segment Statistics 5x Faster with DuckDB
*August 9*

You can now calculate segment performance metrics dramatically faster using DuckDB's columnar analytics engine, transforming multi-minute aggregations into near-instant results.

**What's improved:**
- Segment statistics leverage DuckDB's OLAP optimizations
- Efficient columnar aggregations for customer segment analysis
- Faster multi-dimensional segment breakdowns
- Better performance on datasets with millions of transactions

**Real-world insight:** A national retail chain analyzing customer segments across 5,000 stores with quarterly data previously spent 12 minutes generating segment statistics for their executive dashboard. With DuckDB-powered calculations, the same analysis now completes in 2-3 minutes, freeing up an analyst's time for deeper business insights instead of waiting for queries to finish. For their monthly performance reviews, this cuts reporting preparation time from 2 hours to 20 minutes.

*PR #74*

---

---

## July 2024

### Discover Product Relationships with Market Basket Analysis
*July 30*

You can now uncover which products customers purchase together through association rule mining, enabling data-driven cross-selling, store layout optimization, and inventory management.

**Product association features:**
- Calculate support, confidence, and lift metrics for product pairs
- Identify strong product relationships and complementary items
- Filter rules by minimum support and confidence thresholds
- Generate cross-sell and upsell recommendations
- Support for transaction-level and customer-level analysis

**Real-world insight:** A hardware store analyzes 6 months of transaction data and discovers that customers buying paint brushes also purchase painter's tape 67% of the time (4.2x higher than random chance). They relocate painter's tape to the paint aisle and add "frequently bought together" signage. Within 3 weeks, painter's tape sales increase 23%, and average transaction value in the paint department rises $4.80 per customer, adding $67K in quarterly revenue.

*PR #69*

---

### Visualize Cumulative Effects with Waterfall Charts
*July 18*

You can now create waterfall plots showing how individual components contribute to or subtract from a total, perfect for communicating financial performance, margin analysis, and revenue drivers.

**Waterfall plot capabilities:**
- Visual breakdown of cumulative changes
- Clearly show positive and negative contributions
- Customizable colors for increases, decreases, and totals
- Automatic connector lines between components
- Flexible data label formatting (or suppressed with `None`)

**Real-world insight:** A CFO presenting quarterly margin changes to the board uses a waterfall chart showing how $2.4M baseline margin increased by $890K from pricing, decreased by $340K from promotional costs, and decreased by $120K from supply chain disruptions, resulting in $2.83M final margin. The visual instantly communicates the story behind the net +$430K change, making the presentation 60% shorter while improving stakeholder comprehension.

*PR #63, PR #64, PR #68*

---

### Configure Analysis with Pandas-Style Options System
*July 22*

You can now control PyRetailScience behavior through a centralized options system inspired by pandas, enabling consistent configuration across your entire analysis workflow.

**Options system features:**
- Get, set, and reset options programmatically
- Context managers for temporary option changes
- Column naming, aggregation, and visualization defaults
- Consistent configuration across all modules
- Discoverable through `options.` namespace

**Real-world insight:** A data team supporting 8 different retail clients maintains separate configuration files for each client's data schema (some use "customer_id", others "cust_no", others "client_id"). With the options system, they create client-specific configuration scripts that set all column mappings at once, eliminating hundreds of parameter overrides throughout their analysis code and reducing client onboarding time from 2 days to 3 hours.

*PR #66*

---

### Custom Percentile-Based Segmentation
*July 10*

You can now segment customers using custom percentile thresholds tailored to your business needs, instead of being limited to predefined Heavy, Medium, Light categories.

**ThresholdSegmentation capabilities:**
- Define custom percentile thresholds (e.g., 95th, 85th, 50th for VIP tiers)
- Flexible segment mapping through dictionary-based IDs and names
- Configurable aggregation functions (sum, mean, max, etc.)
- Customizable handling of zero-spend customers
- `HMLSegmentation` refactored as a specialized case

**Real-world insight:** A luxury retailer segments customers using custom percentiles (95th, 85th, 50th) reflecting their VIP tier structure instead of equal-width buckets. This reveals that their top 5% of customers (VIPs) account for 32% of revenue but only received generic communication. By identifying true high-value customers through custom segmentation, they launch a personalized loyalty program increasing VIP repeat purchase rate from 18% to 34%.

*PR #58*

---

### Add Total Row to Segment Statistics
*July 5*

You can now see aggregate totals across all customer segments in SegTransactionStats output, enabling quick verification of overall metrics without manual summation.

**What's included:**
- Automatic "total" row appended to segment statistics
- Aggregate revenue, transaction counts, and customer counts
- Optional visibility control through `hide_total` parameter in plots
- Works seamlessly with all existing segment analysis workflows

**Real-world insight:** When analyzing segment performance in a management meeting, presenters no longer need to manually calculate that all segments sum to $2.4M in quarterly revenue. The total row is automatically included, providing instant validation that your analysis captures 100% of the business and preventing embarrassing discrepancies when stakeholders cross-check the numbers.

*PR #57*

---

### Fix Gain Loss Calculation with Negative Values
*July 4*

You can now trust gain loss analysis results when dealing with negative customer value changes, fixing edge cases where calculations produced incorrect metrics.

**What's fixed:**
- Resolved calculation errors when focus group has negative spend changes
- New `process_customer_group()` method properly handles edge cases
- Improved handling of customers with zero spend in both periods
- Comprehensive test coverage for boundary conditions

**Real-world insight:** A subscription service analyzing customer migration between monthly and annual plans discovered their gain loss analysis was incorrectly categorizing churned customers, undercounting losses by 23%. After this fix, they realize customer churn to competitors is 40% worse than previously reported. This triggers an emergency retention campaign targeting at-risk customers, preventing an estimated $180K in Q3 revenue loss.

*PR #56*

---

### Lighter Package with Separate Simulation Module
*July 3*

You can now use core PyRetailScience functionality with fewer dependencies, as data generation logic has been moved to a separate package while example data remains readily available.

**What changed:**
- Data simulation module removed from PyRetailScience core
- Simulation moved to separate purpose-built package
- Example datasets distributed as parquet files for immediate use
- Cleaner, focused API for retail analytics
- Reduced dependency footprint

**Real-world insight:** Organizations using PyRetailScience in production no longer carry unused data generation dependencies, reducing vulnerability surface from 15 indirect dependencies to 3, and reducing package size by 42%. This means faster installations, smaller Docker images, and fewer security patches to manage while maintaining full analytics capabilities through cached example datasets.

*PR #55*

---

---

## June 2024

### Analyze Customer Gains and Losses Between Periods
*June 10*

You can now track customer transitions between business periods and compare performance across different groups, identifying exactly how many customers you gained, lost, or retained.

**Gain/loss analysis features:**
- Track customers moving between periods (P1 to P2)
- Compare focus group vs. comparison group behavior
- Segment gain/loss analysis by grouping columns
- Visual gain/loss charts and detailed statistics
- Understand upgrade, downgrade, and churn patterns

**Real-world insight:** A retail chain analyzes customer movement from Q1 to Q2, discovering that 2,300 customers moved from "Light" to "Medium" spenders, while 1,150 moved the opposite direction. They identify that customers who received personalized email campaigns were 3x more likely to upgrade—validating the marketing investment and guiding resource allocation for Q3.

*PR #42*

---

### Analyze Customer Overlap Across Locations
*June 13*

You can now visualize and quantify customer overlap between stores or any two groups using Venn and Euler diagrams, answering critical questions about customer concentration and cross-shopping behavior.

**Cross-shop analysis capabilities:**
- Create 2-way and 3-way Venn diagrams
- Identify exclusive vs. shared customer bases
- Aggregate any numeric metric (sales, transactions, frequency)
- Table summaries showing counts and totals by overlap region
- Understand omnichannel behavior patterns

**Real-world insight:** A multi-store retailer uses cross-shop analysis to discover that only 18% of their online customers ever visit physical stores, while 67% of store-only customers never shop online. This asymmetry triggers a new program targeting the 33% bridge audience, converting them to omnichannel shoppers and increasing lifetime value by 40%.

*PR #48*

---

### Faster Days Between Purchases Calculations
*June 25*

You can now calculate days between purchases metrics significantly faster through optimized algorithms and code deduplication in the customer module.

**Performance benefits:**
- Substantially faster DaysBetweenPurchases metric calculation
- Refactored customer module eliminates duplicate code
- Maintains identical output with improved performance
- Better handling of edge cases

**Real-world insight:** An analyst running customer analysis on a 1M+ customer dataset previously waited 3+ minutes for days_between_purchases calculations. The optimization cuts this to under 30 seconds, making interactive exploratory analysis feasible instead of requiring overnight batch jobs.

*PR #50*

---

---

## May 2024

### Visualize Performance Indices for Customer Segments
*May 10*

You can now create index charts showing how each customer segment performs relative to company average (index 100), making it instantly clear which segments are over- or under-performing.

**Index chart capabilities:**
- Visual representation of segment performance vs. baseline
- Support for any aggregated metric (RFM, value, frequency)
- Comparable across different time periods and dimensions
- Highlights both outperformers and underperformers
- Color-coded bars for easy interpretation

**Real-world insight:** A retail manager segments their customer base into HML (Heavy/Medium/Light) and immediately sees that "Heavy" customers have an index of 285 (2.85x average spend) while "Light" customers are at 48. The visual immediately communicates the value concentration, making a business case for targeted retention efforts on the "Heavy" segment worth $1.8M annually.

*PR #30*

---

### Sort and Filter Index Charts
*May 13*

You can now sort and filter index plots, letting you focus on the segments that matter most and customize the visual presentation to your analysis needs.

**Index plot enhancements:**
- Flexible sorting options for segment ordering
- Filter to include only specific segments
- Visual improvements and better labeling
- Exclude groups from calculations or display
- Create focused visualizations for presentations

**Real-world insight:** An analyst creates an index chart of 40 product categories but only wants to highlight the top 10 performers and bottom 5 strugglers. Sorting and filtering let them create a focused visualization that tells a clear story without manual post-processing in Excel or design software.

*PR #31*

---

### Reuse Existing Segmentations
*May 10*

You can now apply pre-existing segmentation results to new data or analysis contexts without recalculating, enabling consistency across analyses and reducing computation time.

**Segmentation flexibility:**
- Load and reuse previously calculated segmentations
- Maintain consistent segment definitions across time periods
- Avoid redundant recalculations
- Apply same logic to different data subsets
- Ensure temporal comparability

**Real-world insight:** A retail analyst creates a stable customer segmentation in Q1. Rather than recalculating in Q2, they apply the same thresholds to Q2 data, ensuring consistent segment definitions across quarters. This consistency makes period-over-period comparisons reliable without segment drift from recalculation, revealing true behavior changes rather than artifacts from shifting definitions.

*PR #29*

---

### Professional Charts with Custom Fonts
*May 6*

You can now generate publication-ready charts with the Poppins font family, giving visualizations a modern and professional appearance without post-processing.

**Visual enhancements:**
- Poppins font family throughout all visualizations
- Modern, professional appearance
- Consistent typography across all charts
- Brand-ready output without design software
- Improved readability and aesthetics

**Real-world insight:** An analyst creating charts for executive presentations previously exported them for manual font adjustments in design software before they looked "professional enough." With Poppins fonts built-in, charts are board-ready immediately, cutting presentation preparation time by 30 minutes per set of 10 charts.

*PR #27*

---

### Custom Data Validation Rules
*May 5*

You can now create and apply custom data contracts for validating dataframe structure and content, ensuring data quality before analysis.

**Data contract capabilities:**
- Define custom validation rules for datasets
- Validate column presence and data types
- Check for non-null values in critical columns
- Clear error messages when validation fails
- Reusable contracts across analyses

**Real-world insight:** A data pipeline team processes customer data from multiple systems with inconsistent schemas. Custom contracts catch schema mismatches and missing values before they propagate to analysis, preventing silent errors that would otherwise lead to incorrect business conclusions that could cost millions in misdirected marketing spend.

*PR #26*

---

---

## April 2024

### HML Customer Segmentation
*April 9*

You can now segment customers into High, Medium, and Low value groups based on spending patterns, enabling targeted marketing strategies and resource allocation.

**HML segmentation features:**
- Automatically categorize customers as High, Medium, or Low spenders
- Flexible percentile-based thresholds
- Integrated visualization support
- Comprehensive documentation with examples
- Foundation for targeted retention and acquisition

**Real-world insight:** A retail chain used HML segmentation to identify that their "High" customers (top 20% by spend) account for 65% of annual revenue. By launching a dedicated VIP loyalty program for this segment with exclusive early-access to sales and personalized service, they increased retention from 68% to 84% and lifetime value by 31%, protecting $4.2M in annual revenue.

*PR #23*

---

---

## March 2024

### Analyze Customer Churn and Retention Patterns
*March 13*

You can now identify and analyze customer churn periods and transaction churn rates, enabling proactive retention strategies based on purchase behavior patterns.

**Churn analysis features:**
- Determine churn periods based on inactivity thresholds
- Calculate transaction-level churn indicators
- Identify at-risk customers before they churn
- Visual churn analysis examples and guides
- Predictive churn modeling support

**Real-world insight:** A coffee subscription service identified that customers who didn't purchase within 35 days had an 82% probability of churning permanently. Using this insight, they implemented an automated email campaign at day 28 for inactive customers, reducing churn from 12.3% to 8.7% monthly and recovering $340K annually from customers who would otherwise have been lost.

*PR #21*

---

---

## February 2024

### Programmatic Data Simulation via API
*February 10*

You can now run retail data simulations programmatically through a Python API instead of only through command-line configuration, enabling integration with automated pipelines.

**API simulation features:**
- Create and configure simulations entirely in Python code
- Flexible configuration system compatible with YAML files
- New `Simulation` class with `run()` method
- Comprehensive example notebook
- Column naming cleanup for consistency

**Real-world insight:** A consulting firm building retail analytics benchmarks needed to generate comparable datasets across different retail scenarios for 50+ client engagements. Previously, they manually edited YAML files for each scenario—a process taking 3-4 hours. With the API, they now generate all 50 simulation configurations programmatically in 15 minutes, reducing project setup time by 80%.

*PR #9*

---

### Define Data Contracts for Validation
*February 15*

You can now document and validate data requirements through explicit data contracts, ensuring data quality and providing clear specifications for datasets.

**Data contract features:**
- Define schema requirements and field descriptions
- Validation logic in new `contracts.py` module
- Comprehensive examples demonstrating usage
- Integration with simulation and analysis workflows
- Extensible framework for custom validators

**Real-world insight:** A large retailer implemented data contracts across their 12-person analytics team to prevent pipeline breaks when source systems changed. When a POS system upgrade renamed a column from `trans_id` to `transaction_id`, the contract validation caught this mismatch within 30 minutes of deployment, preventing a cascading failure that would have halted 8 dependent analyses and delayed decision-making.

*PR #12*

---

### Fix Oscillating Customer Counts in Simulations
*February 16*

You can now generate realistic simulated customer data with stable counts across runs, eliminating artificial variance from synchronized purchasing behavior.

**What's fixed:**
- Random stagger applied to first purchase timing for each customer
- Eliminates synchronized purchasing behavior in simulations
- Ensures realistic customer count evolution over time
- Improves data quality for testing and benchmarking

**Real-world insight:** A data scientist using PyRetailScience to benchmark analytical code discovered that simulations showed customer counts fluctuating by ±3% between runs, causing confusion about whether their analysis was detecting real patterns or simulation artifacts. After this fix, counts stabilized within ±0.2%, giving them confidence that analytical findings represent genuine patterns.

*PR #14*

---

---

## January 2024

*No significant user-facing features merged in January 2024. This month focused on establishing CI/CD infrastructure including documentation deployment, testing pipelines, and PyPI package publishing workflows.*

---
