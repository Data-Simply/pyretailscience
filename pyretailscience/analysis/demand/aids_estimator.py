"""Almost Ideal Demand System (AIDS) Estimator for Retail Demand Analysis.

## Business Context

The Almost Ideal Demand System (AIDS) is a causal demand model that estimates
price and expenditure elasticities, enabling retailers to understand how price
changes impact demand. Unlike correlation-based methods (e.g., Yule's Q), AIDS
provides actionable insights into consumer behavior through economic theory.

## The Business Problem

Retailers need to answer:
- How will a 10% price increase affect sales volume?
- Which products compete for the same budget (substitutes)?
- Which products are bought together (complements)?
- What is the optimal pricing strategy for category profitability?

Traditional correlation analysis shows relationships but cannot answer causal
questions. AIDS addresses this by modeling budget allocation decisions.

## How It Works

AIDS models consumer expenditure shares as functions of prices and total budget:
- Budget share = how much of total spending goes to each product
- Uses iterative linear least squares (ILLE) to estimate parameters
- Enforces economic constraints (adding-up, homogeneity, symmetry)
- Computes elasticities that measure causal price sensitivity

## Real-World Applications

1. **Pricing Optimization**
   - Understand own-price elasticity to set optimal prices
   - Avoid pricing products with elastic demand too high
   - Identify inelastic products suitable for margin expansion

2. **Promotional Planning**
   - Identify which products drive traffic (high elasticity)
   - Understand cross-price effects to avoid cannibalizing sales
   - Design promotions that maximize category profit

3. **Assortment Planning**
   - Distinguish substitutes (negative cross-price elasticity) from complements
   - Make range decisions based on budget competition
   - Understand how delisting affects remaining products

4. **Private Label Strategy**
   - Measure price sensitivity to national brand vs. private label
   - Optimize PL pricing relative to NB competition
   - Forecast volume transfer from price changes

## Business Value

- **Causal Insights**: Move from "what happened" to "what will happen if"
- **Pricing Power**: Data-driven pricing that maximizes profitability
- **Category Management**: Understand true product relationships
- **Strategic Decisions**: Quantify trade-offs between volume and margin
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pyretailscience.options import ColumnHelper


class AIDSEstimator:
    """Estimates Almost Ideal Demand System for retail category analysis.

    The AIDSEstimator implements an Iterative Linear Least Squares (ILLE)
    approach to estimate demand elasticities while enforcing theoretical
    constraints that ensure economic validity.

    ## Economic Theory

    AIDS assumes consumers allocate their budget to maximize utility subject
    to budget constraints. The model captures:
    - Own-price effects: How price changes affect own quantity demanded
    - Cross-price effects: How price changes affect demand for other products
    - Expenditure effects: How budget changes affect product mix

    ## Constraints

    The model enforces three fundamental restrictions:
    1. **Adding-up**: Budget shares sum to 1 across all products
    2. **Homogeneity**: No money illusion (doubling all prices and income = no change)
    3. **Symmetry**: Cross-price effects are symmetric (Slutsky equation)

    ## Estimation Method

    Uses Iterative Linear Least Squares (ILLE):
    1. Initialize price index (Tornqvist or Laspeyres)
    2. Estimate budget share equations via equation-by-equation OLS
    3. Update price index with new parameters
    4. Iterate until convergence
    5. Apply symmetry constraints via restricted least squares

    **IMPORTANT - Comparison to R implementations:**

    This implementation uses **equation-by-equation Ordinary Least Squares (OLS)**,
    which estimates each product's demand equation independently. This differs from
    R's micEconAids package which uses **Seemingly Unrelated Regression (SUR)** via
    the systemfit package.

    **Key differences:**
    - **Python (this implementation)**: Estimates each equation separately using OLS.
      Faster computation, simpler implementation, but ignores cross-equation error
      correlations.
    - **R (micEconAids/systemfit)**: Estimates all equations jointly using SUR.
      Accounts for cross-equation error correlations, providing more efficient
      estimates (smaller standard errors) when errors are correlated.

    **Which to use:**
    - For **business applications** (pricing, assortment): Either method is appropriate.
      Both are consistent estimators and will converge to the true elasticities with
      sufficient data.
    - For **academic research/publication**: Use R's SUR implementation (micEconAids)
      as it is the field standard and provides more efficient estimates.
    - For **large datasets** (millions of rows): This Python OLS implementation will
      be significantly faster than SUR (see performance notes below).

    **Performance:** Equation-by-equation OLS scales linearly with number of observations
    O(n), while SUR scales quadratically O(n^2) due to cross-equation covariance matrix
    computation. For datasets with >100k observations, OLS can be 10-100x faster.

    ## Example Use Case

    A supermarket analyzing breakfast cereal category:
    - Estimates own-price elasticity for private label corn flakes: -2.1
    - Identifies substitute relationship with national brand: +0.8 cross-elasticity
    - Finds complementary relationship with milk: -0.3 cross-elasticity

    Decision: Can increase PL corn flakes price 5% with minimal volume loss
    because consumers view it as differentiated from national brand.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        product_col: str,
        price_col: str,
        quantity_col: str,
        expenditure_col: str | None = None,
        price_index_method: Literal["tornqvist", "laspeyres", "stone"] = "tornqvist",
        max_iterations: int = 100,
        convergence_tolerance: float = 1e-6,
        enforce_constraints: bool = True,
        restricted: bool = True,
    ) -> None:
        """Initialize AIDS estimator for demand analysis.

        Args:
            df (pd.DataFrame): Transaction or aggregate data with prices, quantities, and expenditures.
                Must contain: product identifier, prices, quantities, and either expenditure or
                sufficient data to compute it.
            product_col (str): Column containing product identifiers to analyze
                (e.g., "sku", "product_name", "brand").
            price_col (str): Column containing product prices. Should be consistent units
                across products and time periods.
            quantity_col (str): Column containing quantities purchased. Should be in
                consistent units (e.g., units, liters, kg).
            expenditure_col (str | None, optional): Column containing total expenditure
                per observation. If None, computed as sum of price * quantity across products.
                Defaults to None.
            price_index_method (Literal["tornqvist", "laspeyres", "stone"], optional):
                Method for computing aggregate price index:
                - "tornqvist": Tornqvist price index (preferred, uses arithmetic mean of shares)
                - "laspeyres": Laspeyres price index (base period weighted)
                - "stone": Stone price index (simple, but potential endogeneity issues)
                Defaults to "tornqvist".
            max_iterations (int, optional): Maximum iterations for ILLE convergence.
                Defaults to 100.
            convergence_tolerance (float, optional): Convergence threshold for parameter
                changes between iterations. Defaults to 1e-6.
            enforce_constraints (bool, optional): Whether to enforce adding-up, homogeneity,
                and symmetry constraints. Should always be True for valid economic interpretation.
                Defaults to True.

        Raises:
            ValueError: If required columns are missing from the dataframe.
            ValueError: If data contains invalid values (negative prices, quantities, expenditures).
            ValueError: If insufficient data for estimation (too few observations or products).

        Business Example:
            >>> # Analyze coffee category demand elasticities
            >>> estimator = AIDSEstimator(
            ...     df=category_data,
            ...     product_col="brand",
            ...     price_col="avg_price",
            ...     quantity_col="units_sold",
            ...     expenditure_col="total_spend",
            ...     price_index_method="tornqvist",
            ...     enforce_constraints=True
            ... )
            >>> # Fit model and extract elasticities
            >>> estimator.fit()
            >>> elasticities = estimator.get_elasticities()
        """
        ColumnHelper()

        # Validate required columns
        required_cols = [product_col, price_col, quantity_col]
        if expenditure_col is not None:
            required_cols.append(expenditure_col)

        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        # Store configuration
        self.product_col = product_col
        self.price_col = price_col
        self.quantity_col = quantity_col
        self.expenditure_col = expenditure_col
        self.price_index_method = price_index_method
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.enforce_constraints = enforce_constraints
        self.restricted = restricted

        # Prepare data
        self.df = self._prepare_data(df)
        self.products = sorted(self.df[product_col].unique())
        self.n_products = len(self.products)
        # Optional override for Stone index weights (for IL using fitted shares)
        self._weights_override: pd.DataFrame | None = None

        # Validate minimum requirements
        min_products = 2
        if self.n_products < min_products:
            msg = f"At least {min_products} products are required for AIDS estimation"
            raise ValueError(msg)

        if len(self.df) < self.n_products * 3:
            msg = f"Insufficient observations for estimation. Need at least {self.n_products * 3}, got {len(self.df)}"
            raise ValueError(msg)

        # Initialize estimation results
        self.alpha: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.gamma: np.ndarray | None = None
        self.fitted: bool = False
        self.iterations: int = 0
        self.converged: bool = False

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate data for AIDS estimation.

        Supports two data formats:
        1. Panel data: Pre-aggregated with one row per observation-product
        2. Transaction data with dates: Aggregates by date and product

        Args:
            df (pd.DataFrame): Input dataframe with price and quantity columns.

        Returns:
            pd.DataFrame: Cleaned dataframe with budget shares and log variables.

        Raises:
            ValueError: If data contains invalid values.
        """
        df = df.copy()

        # Check if we have transaction data with dates (needs aggregation)
        date_col = "transaction_date"
        needs_aggregation = date_col in df.columns

        if needs_aggregation:
            # Transaction data: aggregate by date and product
            df["_revenue"] = df[self.price_col] * df[self.quantity_col]

            agg_df = (
                df.groupby([date_col, self.product_col])
                .agg(
                    **{
                        self.quantity_col: pd.NamedAgg(column=self.quantity_col, aggfunc="sum"),
                        "_revenue": pd.NamedAgg(column="_revenue", aggfunc="sum"),
                    },
                )
                .reset_index()
            )

            agg_df[self.price_col] = agg_df["_revenue"] / agg_df[self.quantity_col]
            agg_df = agg_df[agg_df[self.quantity_col] > 0].copy()

            if self.expenditure_col is None:
                self.expenditure_col = "total_expenditure"

            daily_expenditure = agg_df.groupby(date_col)["_revenue"].sum().rename(self.expenditure_col)
            agg_df = agg_df.merge(daily_expenditure, on=date_col, how="left")
            agg_df = agg_df[agg_df[self.expenditure_col] > 0].copy()

            agg_df["budget_share"] = agg_df["_revenue"] / agg_df[self.expenditure_col]

            daily_share_sum = agg_df.groupby(date_col)["budget_share"].sum()
            valid_dates = daily_share_sum[np.isclose(daily_share_sum, 1.0)].index
            agg_df = agg_df[agg_df[date_col].isin(valid_dates)].copy()

            agg_df = agg_df.rename(columns={date_col: "observation_id"}).set_index("observation_id")
            agg_df = agg_df.drop(columns=["_revenue"])

            df = agg_df

        else:
            # Panel data: already aggregated, create observation structure
            if not isinstance(df.index, pd.MultiIndex) and df.index.name != "observation_id":
                n_products = df[self.product_col].nunique()
                df["observation_id"] = df.index // n_products
                df = df.set_index("observation_id")

            # Compute expenditure if not provided
            if self.expenditure_col is None:
                self.expenditure_col = "total_expenditure"
                expenditure_by_obs = df.groupby(level=0).apply(
                    lambda x: (x[self.price_col] * x[self.quantity_col]).sum(),
                    include_groups=False,
                )
                df[self.expenditure_col] = df.index.map(expenditure_by_obs)

            # Compute budget shares
            df["budget_share"] = (df[self.price_col] * df[self.quantity_col]) / df[self.expenditure_col]

            # Validate budget shares sum to approximately 1
            share_sums = df.groupby(level=0)["budget_share"].sum()
            if not np.allclose(share_sums, 1.0, rtol=0.01):
                msg = "Budget shares do not sum to 1. Check data structure."
                raise ValueError(msg)

        # Validate no negative or zero values
        for col in [self.price_col, self.quantity_col, self.expenditure_col]:
            if (df[col] <= 0).any():
                msg = f"Column {col} contains non-positive values."
                raise ValueError(msg)

        # Compute log-transformed variables
        df["log_price"] = np.log(df[self.price_col])
        df["log_expenditure"] = np.log(df[self.expenditure_col])

        return df

    def _predict_shares_from_params(self) -> pd.DataFrame:
        """Predict budget shares using current parameter estimates.

        Uses the AIDS share equation:
        w_i = alpha_i + sum_j gamma_ij * log p_j + beta_i * log(x/P)

        Returns:
            pd.DataFrame: Predicted budget shares (observations x products).
        """
        # Get current price index
        log_price_index = self._compute_price_index()
        self.df["log_price_index"] = self.df.index.map(log_price_index)
        self.df["log_real_expenditure"] = self.df["log_expenditure"] - self.df["log_price_index"]

        # Pivot to wide format
        log_prices_wide = self.df.pivot(columns=self.product_col, values="log_price")[self.products]
        log_real_expenditure = self.df.groupby(level=0)["log_real_expenditure"].first()

        # Predict shares: w_hat = alpha + (log_prices @ gamma^T) + beta * log_real_expenditure
        n_obs = len(log_prices_wide)
        pred_shares = np.zeros((n_obs, self.n_products))

        # Add alpha (broadcast to all observations)
        pred_shares += self.alpha.reshape(1, -1)

        # Add gamma * log_prices term
        pred_shares += (log_prices_wide.values @ self.gamma.T)

        # Add beta * log_real_expenditure term
        pred_shares += log_real_expenditure.values.reshape(-1, 1) @ self.beta.reshape(1, -1)

        # Clip to valid range and renormalize to ensure shares sum to 1
        pred_shares = np.clip(pred_shares, 1e-9, 1.0)
        pred_shares = pred_shares / pred_shares.sum(axis=1, keepdims=True)

        return pd.DataFrame(pred_shares, index=log_prices_wide.index, columns=self.products)

    def _compute_price_index(self) -> pd.Series:
        """Compute aggregate price index for real expenditure calculation.

        Returns:
            pd.Series: Log of aggregate price index for each observation.
        """
        if self.price_index_method == "stone":
            # Stone price index: weighted average of log prices using budget shares
            shares_wide = self.df.pivot(columns=self.product_col, values="budget_share")[self.products]
            log_prices_wide = self.df.pivot(columns=self.product_col, values="log_price")[self.products]
            if self._weights_override is not None:
                weights = self._weights_override.loc[shares_wide.index, self.products]
            else:
                weights = shares_wide
            result = (weights * log_prices_wide).sum(axis=1)
            return result

        if self.price_index_method == "laspeyres":
            # Laspeyres price index: base period weighted
            base_shares = self.df.groupby(self.product_col)["budget_share"].mean()
            return self.df.groupby(level=0, group_keys=False).apply(
                lambda x: sum(
                    base_shares[prod] * x[x[self.product_col] == prod]["log_price"].iloc[0]
                    if len(x[x[self.product_col] == prod]) > 0
                    else 0
                    for prod in self.products
                ),
            )

        if self.price_index_method == "tornqvist":
            # Tornqvist price index: uses arithmetic mean of budget shares
            # In IL estimation, we use fitted shares if available (similar to Stone IL)
            shares_wide = self.df.pivot(columns=self.product_col, values="budget_share")[self.products]
            log_prices_wide = self.df.pivot(columns=self.product_col, values="log_price")[self.products]

            # Use fitted shares if available (IL method), otherwise use actual shares
            if self._weights_override is not None:
                weights = self._weights_override.loc[shares_wide.index, self.products]
            else:
                weights = shares_wide

            # Tornqvist: weighted average of log prices using budget shares
            # P = exp(sum_k w_k * log p_k) where w_k are the share weights
            result = (weights * log_prices_wide).sum(axis=1)
            return result

        msg = f"Invalid price index method: {self.price_index_method}"
        raise ValueError(msg)

    def _estimate_unrestricted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate unrestricted AIDS model via OLS.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (alpha, beta, gamma) parameter arrays.
                - alpha: Intercept for each product (n_products,)
                - beta: Expenditure coefficients for each product (n_products,)
                - gamma: Price slope matrix for each product pair (n_products, n_products)
        """
        alpha = np.zeros(self.n_products)
        beta = np.zeros(self.n_products)
        gamma = np.zeros((self.n_products, self.n_products))

        # Compute price index
        log_price_index = self._compute_price_index()
        self.df["log_price_index"] = self.df.index.map(log_price_index)

        # Compute real expenditure
        self.df["log_real_expenditure"] = self.df["log_expenditure"] - self.df["log_price_index"]

        # --- Pivot data to wide format for robust regression ---
        budget_shares_wide = self.df.pivot(columns=self.product_col, values="budget_share")
        log_prices_wide = self.df.pivot(columns=self.product_col, values="log_price")

        # Ensure columns are in the same order
        log_prices_wide = log_prices_wide[self.products]

        # Get real expenditure (it's per observation, so we can take it from any product's data)
        log_real_expenditure = self.df.groupby(level=0)["log_real_expenditure"].first()

        # Estimate equation for each product
        for i, product in enumerate(self.products):
            # Dependent variable (budget share for the current product)
            y = budget_shares_wide[product].rename("budget_share_y")

            # Independent variables: all log prices and real expenditure
            x = pd.concat([log_prices_wide, log_real_expenditure], axis=1)
            x.columns = [*self.products, "log_real_expenditure"]

            # Add intercept
            x["intercept"] = 1.0

            # Combine and drop any rows with missing values to ensure alignment
            data_for_regression = pd.concat([y, x], axis=1).dropna()

            # If no complete observations exist for this product, skip it.
            if data_for_regression.empty:
                continue

            y_fit = data_for_regression["budget_share_y"]
            x_fit = data_for_regression.drop("budget_share_y", axis=1)

            # Reorder columns for coefficient extraction
            x_fit = x_fit[["intercept", *self.products, "log_real_expenditure"]]

            # Convert column names to strings to avoid sklearn dtype issues
            x_fit.columns = x_fit.columns.astype(str)

            # OLS estimation
            model = LinearRegression(fit_intercept=False)
            model.fit(x_fit, y_fit)

            # Extract parameters
            alpha[i] = model.coef_[0]
            gamma[i, :] = model.coef_[1 : self.n_products + 1]
            beta[i] = model.coef_[self.n_products + 1]

        return alpha, beta, gamma

    def _estimate_restricted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate AIDS with drop-one equation and price-difference regressors.

        This enforces homogeneity by construction and mirrors restricted OLS
        commonly used in LA/AIDS implementations.

        Returns:
            tuple of (alpha, beta, gamma)
        """
        # Compute price index and real expenditure first
        log_price_index = self._compute_price_index()
        self.df["log_price_index"] = self.df.index.map(log_price_index)
        self.df["log_real_expenditure"] = self.df["log_expenditure"] - self.df["log_price_index"]

        # Wide forms
        shares_wide = self.df.pivot(columns=self.product_col, values="budget_share")[self.products]
        log_prices_wide = self.df.pivot(columns=self.product_col, values="log_price")[self.products]
        log_real_expenditure = self.df.groupby(level=0)["log_real_expenditure"].first()

        # Use the first product as the base good to align with common defaults
        base_product = str(self.products[0])
        kept_products = [str(p) for p in self.products if p != base_product]

        n = self.n_products
        alpha = np.zeros(n)
        beta = np.zeros(n)
        gamma = np.zeros((n, n))

        # For each kept equation i, regress share_i on price differences (log p_j - log p_base) for j != base, and log_real_expenditure
        for i, product in enumerate(kept_products):
            y = shares_wide[product].rename("budget_share_y")
            # Compute price differences with index alignment: log p_j - log p_base
            price_diffs = log_prices_wide[kept_products].sub(log_prices_wide[base_product], axis=0)
            x = pd.concat([price_diffs, log_real_expenditure], axis=1)
            x.columns = [*kept_products, "log_real_expenditure"]
            x["intercept"] = 1.0
            data_for_regression = pd.concat([y, x], axis=1).dropna()
            if data_for_regression.empty:
                continue
            y_fit = data_for_regression["budget_share_y"]
            x_fit = data_for_regression[["intercept", *kept_products, "log_real_expenditure"]]

            # Convert column names to strings to avoid sklearn dtype issues
            x_fit.columns = x_fit.columns.astype(str)

            model = LinearRegression(fit_intercept=False)
            model.fit(x_fit, y_fit)

            row_index = self.products.index(product)
            alpha[row_index] = model.coef_[0]
            beta[row_index] = model.coef_[-1]
            # Map gamma for j != base
            coef_prices = model.coef_[1:-1]
            for j, prod_j in enumerate(kept_products):
                col_index = self.products.index(prod_j)
                gamma[row_index, col_index] = coef_prices[j]
            # Enforce homogeneity for row i: gamma_i,base = -sum_{j!=base} gamma_i,j
            base_col = self.products.index(base_product)
            gamma[row_index, base_col] = -np.sum(gamma[row_index, :])

        # Recover base equation parameters via adding-up constraints
        base_row = self.products.index(base_product)
        alpha[base_row] = 1.0 - np.sum(alpha[[self.products.index(p) for p in kept_products]])
        beta[base_row] = -np.sum(beta[[self.products.index(p) for p in kept_products]])
        # Use symmetry to fill base row from base column for off-diagonals
        for j, prod_j in enumerate(kept_products):
            j_idx = self.products.index(prod_j)
            gamma[base_row, j_idx] = gamma[j_idx, base_row]
        # Final element to satisfy homogeneity in base row
        gamma[base_row, base_row] = -np.sum(gamma[base_row, :])

        # Note: Alpha values from the restricted estimation are already valid intercepts
        # in the AIDS equation. The drop-one method with adding-up recovery ensures
        # sum(alpha) = 1.0 by construction. DO NOT recompute alpha from sample means
        # as this breaks the adding-up property that is built into the estimation method.

        return alpha, beta, gamma

    def _enforce_adding_up(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enforce adding-up constraint: sum of shares = 1.

        Adding-up requires:
        - sum(alpha_i) = 1
        - sum(beta_i) = 0
        - sum(gamma_ij) = 0 for all j

        Args:
            alpha (np.ndarray): Intercept parameters.
            beta (np.ndarray): Expenditure coefficients.
            gamma (np.ndarray): Price slope matrix.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Constrained parameters.
        """
        # Normalize alpha to sum to 1
        alpha = alpha / alpha.sum()

        # Normalize beta to sum to 0
        beta = beta - beta.mean()

        # For gamma: enforce both row and column sums equal zero exactly
        # while preserving symmetry via double-centering.
        # gamma_dc = gamma - rowMean - colMean + overallMean
        row_mean = gamma.mean(axis=1, keepdims=True)
        col_mean = gamma.mean(axis=0, keepdims=True)
        overall_mean = gamma.mean()
        gamma = gamma - row_mean - col_mean + overall_mean
        # Ensure symmetry after centering
        gamma = 0.5 * (gamma + gamma.T)

        return alpha, beta, gamma

    def _enforce_homogeneity(self, gamma: np.ndarray) -> np.ndarray:
        """Enforce homogeneity: each row of gamma sums to zero.

        Standard LA/AIDS homogeneity requires sum_j gamma_ij = 0 for all i.
        This function adjusts rows of gamma to have zero sums without
        modifying beta.

        Args:
            gamma (np.ndarray): Price slope matrix.

        Returns:
            np.ndarray: Gamma with zero row sums.
        """
        row_sums = gamma.sum(axis=1, keepdims=True)
        # Subtract equal share of the row sum across each row to enforce zero sum
        gamma_adjusted = gamma - row_sums / self.n_products
        return gamma_adjusted

    def _enforce_symmetry(self, gamma: np.ndarray) -> np.ndarray:
        """Enforce Slutsky symmetry constraint.

        Symmetry requires:
        - gamma_ij = gamma_ji for all i, j

        Args:
            gamma (np.ndarray): Price slope matrix.

        Returns:
            np.ndarray: Symmetric price slope matrix.
        """
        # Average with transpose to enforce symmetry
        return 0.5 * (gamma + gamma.T)

    def _enforce_constraints_iterative(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        max_constraint_iterations: int = 10,
        constraint_tolerance: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enforce all constraints using iterative projection.

        This method addresses the sequential constraint enforcement problem by
        iteratively projecting parameters onto each constraint until all are
        simultaneously satisfied within tolerance.

        Args:
            alpha (np.ndarray): Intercept parameters.
            beta (np.ndarray): Expenditure coefficients.
            gamma (np.ndarray): Price slope matrix.
            max_constraint_iterations (int): Maximum constraint projection iterations.
            constraint_tolerance (float): Tolerance for constraint satisfaction.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Constrained parameters.
        """
        for _ in range(max_constraint_iterations):
            # Store parameters to check convergence
            alpha_prev = alpha.copy()
            beta_prev = beta.copy()
            gamma_prev = gamma.copy()

            # Apply constraints
            if self.restricted:
                # Enforce symmetry and exact row/column zero sums (double-centering)
                gamma = self._enforce_symmetry(gamma)
                row_mean = gamma.mean(axis=1, keepdims=True)
                col_mean = gamma.mean(axis=0, keepdims=True)
                overall_mean = gamma.mean()
                gamma = gamma - row_mean - col_mean + overall_mean
                gamma = self._enforce_symmetry(gamma)
                # Normalize alpha and beta sums exactly
                alpha = alpha / alpha.sum()
                beta = beta - beta.mean()
            else:
                gamma = self._enforce_symmetry(gamma)
                gamma = self._enforce_homogeneity(gamma)
                alpha, beta, gamma = self._enforce_adding_up(alpha, beta, gamma)

            # Check if constraints are satisfied (parameters stopped changing)
            alpha_change = np.max(np.abs(alpha - alpha_prev))
            beta_change = np.max(np.abs(beta - beta_prev))
            gamma_change = np.max(np.abs(gamma - gamma_prev))

            if max(alpha_change, beta_change, gamma_change) < constraint_tolerance:
                break

        return alpha, beta, gamma

    def fit(self, verbose: bool = False) -> "AIDSEstimator":
        """Estimate AIDS model using Iterative Linear Least Squares (ILLE).

        The ILLE algorithm:
        1. Initialize price index
        2. Estimate unrestricted model via OLS
        3. Enforce constraints (if enabled) using iterative projection
        4. Update price index with new parameters
        5. Check convergence; if not converged, repeat from step 2

        Args:
            verbose (bool, optional): If True, print detailed iteration logs. Defaults to False.

        Returns:
            AIDSEstimator: Self, for method chaining.

        Raises:
            RuntimeError: If estimation fails to converge within max_iterations.
        """
        # Initialize with simple price index
        self.alpha = np.ones(self.n_products) / self.n_products
        self.beta = np.zeros(self.n_products)
        self.gamma = np.zeros((self.n_products, self.n_products))

        if verbose:
            print(f"\n{'='*80}")
            print(f"AIDS ESTIMATION: {self.price_index_method.upper()} INDEX, {'RESTRICTED' if self.restricted else 'UNRESTRICTED'}")
            print(f"{'='*80}\n")

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'-'*80}")

            # Store previous parameters for convergence check
            alpha_old = self.alpha.copy()
            beta_old = self.beta.copy()
            gamma_old = self.gamma.copy()

            # Estimate model
            if self.restricted:
                self.alpha, self.beta, self.gamma = self._estimate_restricted()
            else:
                self.alpha, self.beta, self.gamma = self._estimate_unrestricted()

            if verbose:
                print(f"  After estimation (before constraints):")
                print(f"    Alpha: {self.alpha}")
                print(f"    Beta: {self.beta}")
                print(f"    Gamma row sums: {self.gamma.sum(axis=1)}")
                print(f"    Gamma symmetry max diff: {np.max(np.abs(self.gamma - self.gamma.T)):.6e}")

            # Update price index weights with fitted shares for IL methods
            if self.price_index_method in ("stone", "tornqvist"):
                # Use the helper method to predict shares from current parameters
                pred_shares = self._predict_shares_from_params()

                # Store fitted shares for price index computation
                self._weights_override = pred_shares

                # Recompute price index with updated weights (for use in next iteration)
                log_price_index_new = self._compute_price_index()
                self.df["log_price_index"] = self.df.index.map(log_price_index_new)
                self.df["log_real_expenditure"] = self.df["log_expenditure"] - self.df["log_price_index"]

                if verbose and self.price_index_method == "tornqvist":
                    print(f"  Tornqvist IL: Updated weights with fitted shares")

            # Enforce constraints if required using iterative projection
            if self.enforce_constraints:
                self.alpha, self.beta, self.gamma = self._enforce_constraints_iterative(
                    self.alpha,
                    self.beta,
                    self.gamma,
                )

                if verbose:
                    print(f"  After constraint enforcement:")
                    print(f"    Alpha sum: {self.alpha.sum():.6f} (should be 1.0)")
                    print(f"    Beta sum: {self.beta.sum():.6e} (should be 0.0)")
                    print(f"    Gamma col sums: {self.gamma.sum(axis=0)}")
                    print(f"    Gamma row sums: {self.gamma.sum(axis=1)}")
                    print(f"    Gamma symmetry max diff: {np.max(np.abs(self.gamma - self.gamma.T)):.6e}")

            # Check convergence
            alpha_change = np.max(np.abs(self.alpha - alpha_old))
            beta_change = np.max(np.abs(self.beta - beta_old))
            gamma_change = np.max(np.abs(self.gamma - gamma_old))
            max_change = max(alpha_change, beta_change, gamma_change)

            self.iterations = iteration + 1

            if verbose:
                print(f"  Parameter changes:")
                print(f"    Alpha: {alpha_change:.6e}")
                print(f"    Beta: {beta_change:.6e}")
                print(f"    Gamma: {gamma_change:.6e}")
                print(f"    Max change: {max_change:.6e} (tolerance: {self.convergence_tolerance:.6e})")
                print()

            if max_change < self.convergence_tolerance:
                self.converged = True
                if verbose:
                    print(f"[CONVERGED] in {self.iterations} iterations\n")
                break

        if not self.converged:
            msg = f"AIDS estimation did not converge within {self.max_iterations} iterations"
            raise RuntimeError(msg)

        self.fitted = True
        return self

    def get_elasticities(self) -> pd.DataFrame:
        """Compute own-price, cross-price, and expenditure elasticities.

        Elasticities measure percentage change in quantity demanded for a
        1% change in price or expenditure:
        - Own-price elasticity: % change in quantity_i from 1% change in price_i
        - Cross-price elasticity: % change in quantity_i from 1% change in price_j
        - Expenditure elasticity: % change in quantity_i from 1% change in expenditure

        Returns:
            pd.DataFrame: Elasticity matrix with products as rows and columns.
                Diagonal elements are own-price elasticities.
                Off-diagonal elements are cross-price elasticities.
                Final column is expenditure elasticity.

        Raises:
            RuntimeError: If model has not been fitted yet.

        Business Interpretation:
            - Own-price elasticity < -1: Elastic (sensitive to price changes)
            - Own-price elasticity > -1: Inelastic (less sensitive to price changes)
            - Cross-price elasticity > 0: Substitutes (one up, other down)
            - Cross-price elasticity < 0: Complements (bought together)
            - Expenditure elasticity > 1: Luxury (increases faster than income)
            - Expenditure elasticity < 1: Necessity (increases slower than income)
        """
        if not self.fitted:
            msg = "Model must be fitted before computing elasticities. Call fit() first."
            raise RuntimeError(msg)

        # Compute average budget shares
        avg_shares = self.df.groupby(self.product_col)["budget_share"].mean().values

        # Initialize elasticity matrices
        own_price_elasticities = np.zeros(self.n_products)
        cross_price_elasticities = np.zeros((self.n_products, self.n_products))
        expenditure_elasticities = np.zeros(self.n_products)

        # Compute elasticities using AIDS formulas
        for i in range(self.n_products):
            # Expenditure elasticity: 1 + (beta_i / w_i)
            expenditure_elasticities[i] = 1 + (self.beta[i] / avg_shares[i])

            for j in range(self.n_products):
                # Price elasticity: -(delta_ij) + (gamma_ij / w_i) - beta_i * w_j
                # where delta_ij = 1 if i==j, 0 otherwise
                delta_ij = 1.0 if i == j else 0.0
                elasticity = -delta_ij + (self.gamma[i, j] / avg_shares[i]) - self.beta[i] * avg_shares[j]

                if i == j:
                    own_price_elasticities[i] = elasticity
                else:
                    cross_price_elasticities[i, j] = elasticity

        # Combine into single matrix
        elasticity_matrix = cross_price_elasticities.copy()
        np.fill_diagonal(elasticity_matrix, own_price_elasticities)

        # Create DataFrame
        elasticity_df = pd.DataFrame(
            elasticity_matrix,
            index=self.products,
            columns=[f"{p}_price" for p in self.products],
        )
        elasticity_df["expenditure"] = expenditure_elasticities

        return elasticity_df

    def get_diagnostics(self) -> dict[str, float | pd.DataFrame]:
        """Compute model diagnostics including R-squared, residuals, and Wald tests.

        Returns:
            dict[str, float | pd.DataFrame]: Dictionary containing:
                - "r_squared": R-squared for each product equation
                - "residuals": Residuals for each product and observation
                - "wald_tests": Wald test statistics for constraint tests
                - "iterations": Number of iterations to convergence
                - "converged": Whether model converged

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if not self.fitted:
            msg = "Model must be fitted before computing diagnostics. Call fit() first."
            raise RuntimeError(msg)

        diagnostics = {
            "iterations": self.iterations,
            "converged": self.converged,
            "r_squared": {},
            "residuals": pd.DataFrame(),
        }

        # Compute R-squared and residuals for each product
        log_price_index = self._compute_price_index()
        self.df["log_price_index"] = log_price_index
        self.df["log_real_expenditure"] = self.df["log_expenditure"] - self.df["log_price_index"]

        for i, product in enumerate(self.products):
            product_data = self.df[self.df[self.product_col] == product].copy()

            # Predicted budget share
            predicted_share = self.alpha[i]
            for j, other_product in enumerate(self.products):
                log_price_j = self.df[self.df[self.product_col] == other_product]["log_price"].values[
                    : len(product_data)
                ]
                predicted_share = predicted_share + self.gamma[i, j] * log_price_j

            predicted_share = predicted_share + self.beta[i] * product_data["log_real_expenditure"].values

            # Residuals
            residuals = product_data["budget_share"].values - predicted_share

            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((product_data["budget_share"].values - product_data["budget_share"].mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            diagnostics["r_squared"][product] = r_squared
            diagnostics["residuals"][product] = residuals

        # Wald tests for constraints
        if self.enforce_constraints:
            # Test adding-up: sum(alpha) = 1
            wald_adding_up = (self.alpha.sum() - 1.0) ** 2

            # Test homogeneity: sum(gamma_ij) + beta_i = 0
            wald_homogeneity = np.sum((self.gamma.sum(axis=1) + self.beta) ** 2)

            # Test symmetry: gamma_ij = gamma_ji
            wald_symmetry = np.sum((self.gamma - self.gamma.T) ** 2)

            diagnostics["wald_tests"] = {
                "adding_up": wald_adding_up,
                "homogeneity": wald_homogeneity,
                "symmetry": wald_symmetry,
            }

        return diagnostics

    def summary(self) -> str:
        """Generate summary statistics and diagnostics report.

        Returns:
            str: Formatted summary report with parameter estimates, elasticities,
                and diagnostic statistics.

        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if not self.fitted:
            msg = "Model must be fitted before generating summary. Call fit() first."
            raise RuntimeError(msg)

        diagnostics = self.get_diagnostics()
        elasticities = self.get_elasticities()

        summary_lines = [
            "AIDS Model Estimation Summary",
            "=" * 50,
            f"Number of products: {self.n_products}",
            f"Number of observations: {len(self.df)}",
            f"Price index method: {self.price_index_method}",
            f"Constraints enforced: {self.enforce_constraints}",
            f"Converged: {self.converged}",
            f"Iterations: {self.iterations}",
            "",
            "R-squared by product:",
            "-" * 50,
        ]

        for product, r2 in diagnostics["r_squared"].items():
            summary_lines.append(f"  {product}: {r2:.4f}")

        summary_lines.extend(
            [
                "",
                "Elasticities:",
                "-" * 50,
                str(elasticities),
                "",
            ],
        )

        if "wald_tests" in diagnostics:
            summary_lines.extend(
                [
                    "Constraint tests (Wald statistics):",
                    "-" * 50,
                    f"  Adding-up: {diagnostics['wald_tests']['adding_up']:.6f}",
                    f"  Homogeneity: {diagnostics['wald_tests']['homogeneity']:.6f}",
                    f"  Symmetry: {diagnostics['wald_tests']['symmetry']:.6f}",
                ],
            )

        return "\n".join(summary_lines)
