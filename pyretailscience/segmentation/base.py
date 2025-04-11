"""This module provides a base class for segmenting customers based on their spend and transaction statistics."""

import pandas as pd

from pyretailscience.options import get_option


class BaseSegmentation:
    """A base class for customer segmentation."""

    def add_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds the segment to the dataframe based on the customer_id column.

        Args:
            df (pd.DataFrame): The dataframe to add the segment to. The dataframe must have a customer_id column.

        Returns:
            pd.DataFrame: The dataframe with the segment added.

        Raises:
            ValueError: If the number of rows before and after the merge do not match.
        """
        rows_before = len(df)
        df = df.merge(
            self.df["segment_name"],
            how="left",
            left_on=get_option("column.customer_id"),
            right_index=True,
        )
        rows_after = len(df)
        if rows_before != rows_after:
            raise ValueError("The number of rows before and after the merge do not match. This should not happen.")

        return df
