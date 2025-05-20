"""Customer Segmentation Using RFM Analysis.

This module implements RFM (Recency, Frequency, Monetary) segmentation, a widely used technique in customer analytics
to categorize customers based on their purchasing behavior.

RFM segmentation assigns scores to customers based on:
1. Recency (R): How recently a customer made a purchase.
2. Frequency (F): How often a customer makes purchases.
3. Monetary (M): The total amount spent by a customer.

### Benefits of RFM Segmentation:
- **Customer Value Analysis**: Identifies high-value customers who contribute the most revenue.
- **Personalized Marketing**: Enables targeted campaigns based on customer purchasing behavior.
- **Customer Retention Strategies**: Helps recognize at-risk customers and develop engagement strategies.
- **Sales Forecasting**: Provides insights into future revenue trends based on past spending behavior.

### Scoring Methodology:
- Each metric (R, F, M) is divided into 10 bins (0-9) using the NTILE(10) function.
- A higher score indicates a better customer (e.g., lower recency, higher frequency, and monetary value).
- The final RFM segment is computed as `R*100 + F*10 + M`, providing a unique customer classification.

This module leverages `pandas` and `ibis` for efficient data processing and integrates with retail analytics workflows
to enhance customer insights and business decision-making.
"""

import datetime

import ibis
import pandas as pd

from pyretailscience.options import ColumnHelper, get_option


class RFMSegmentation:
    """Segments customers using the RFM (Recency, Frequency, Monetary) methodology.

    Customers are scored on three dimensions:
    - Recency (R): Days since the last transaction (lower is better).
    - Frequency (F): Number of unique transactions (higher is better).
    - Monetary (M): Total amount spent (higher is better).

    Each metric is ranked into 10 bins (0-9) using NTILE(10) where,
    - 9 represents the best score (top 10% of customers).
    - 0 represents the lowest score (bottom 10% of customers).
    The RFM segment is a 3-digit number (R*100 + F*10 + M), representing customer value.
    """

    _df: pd.DataFrame | None = None

    def __init__(self, df: pd.DataFrame | ibis.Table, current_date: str | datetime.date | None = None) -> None:
        """Initializes the RFM segmentation process.

        Args:
            df (pd.DataFrame | ibis.Table): A DataFrame or Ibis table containing transaction data.
                Must include the following columns:
                - customer_id
                - transaction_date
                - unit_spend
                - transaction_id
            current_date (Optional[Union[str, datetime.date]]): The reference date for calculating recency.
                Can be a string (format: "YYYY-MM-DD"), a date object, or None (defaults to the current system date).

        Raises:
            ValueError: If the dataframe is missing required columns.
            TypeError: If the input data is not a pandas DataFrame or an Ibis Table.
        """
        cols = ColumnHelper()
        required_cols = [
            cols.customer_id,
            cols.transaction_date,
            cols.unit_spend,
            cols.transaction_id,
        ]
        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df)
        elif not isinstance(df, ibis.Table):
            raise TypeError("df must be either a pandas DataFrame or an Ibis Table")

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            error_message = f"Missing required columns: {missing_cols}"
            raise ValueError(error_message)

        if isinstance(current_date, str):
            current_date = datetime.date.fromisoformat(current_date)
        elif current_date is None:
            current_date = datetime.datetime.now(datetime.UTC).date()
        elif not isinstance(current_date, datetime.date):
            raise TypeError("current_date must be a string in 'YYYY-MM-DD' format, a datetime.date object, or None")

        self.table = self._compute_rfm(df, current_date)

    def _compute_rfm(self, df: ibis.Table, current_date: datetime.date) -> ibis.Table:
        """Computes the RFM metrics and segments customers accordingly.

        Args:
            df (ibis.Table): The transaction data table.
            current_date (datetime.date): The reference date for calculating recency.

        Returns:
            ibis.Table: A table with RFM scores and segment values.
        """
        cols = ColumnHelper()
        current_date_expr = ibis.literal(current_date)

        customer_metrics = df.group_by(cols.customer_id).aggregate(
            recency_days=current_date_expr.delta(df[cols.transaction_date].max().cast("date"), unit="day").cast(
                "int32",
            ),
            frequency=df[cols.transaction_id].nunique(),
            monetary=df[cols.unit_spend].sum(),
        )

        window_recency = ibis.window(
            order_by=[ibis.asc(customer_metrics.recency_days), ibis.asc(customer_metrics.customer_id)],
        )
        window_frequency = ibis.window(
            order_by=[ibis.asc(customer_metrics.frequency), ibis.asc(customer_metrics.customer_id)],
        )
        window_monetary = ibis.window(
            order_by=[ibis.asc(customer_metrics.monetary), ibis.asc(customer_metrics.customer_id)],
        )

        rfm_scores = customer_metrics.mutate(
            r_score=(ibis.ntile(10).over(window_recency)),
            f_score=(ibis.ntile(10).over(window_frequency)),
            m_score=(ibis.ntile(10).over(window_monetary)),
        )

        return rfm_scores.mutate(
            rfm_segment=(rfm_scores.r_score * 100 + rfm_scores.f_score * 10 + rfm_scores.m_score),
            fm_segment=(rfm_scores.f_score * 10 + rfm_scores.m_score),
        )

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the segment names."""
        if self._df is None:
            self._df = self.table.execute().set_index(get_option("column.customer_id"))
        return self._df
