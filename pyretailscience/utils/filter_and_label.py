"""filter_and_label Utilities - Filtering and Labeling by Condition.

This module provides utilities to filter an Ibis table based on arbitrary logical conditions
and attach descriptive labels to matched rows. It is useful for segmenting and analyzing data
according to business-defined rules, such as category classification, customer segmentation,
or pricing tiers.

Example use cases:
- Tagging product records based on category.
- Classifying transactions into business segments.
- Preparing labeled datasets for analysis or machine learning.
"""

import ibis


def filter_and_label_by_condition(
    table: ibis.Table,
    conditions: dict[str, ibis.expr.types.BooleanColumn],
) -> ibis.Table:
    """Filters a table based on specified conditions and adds labels.

    This function filters rows in a table based on specified conditions and adds a new column
    indicating the label associated with each condition. It's useful for categorizing data
    based on different criteria.

    Example:
        data = ibis.memtable(df)
        labeled_data = filter_and_label_by_condition(
            data,
            conditions={
                "toys": data["category"] == "toys",
                "shoes": data["category"] == "shoes"
            }
        )
        # labeled_data will only contain rows where category is either "toys" or "shoes",
        # and a new column 'label' will be added indicating which category it belongs to.

    Args:
        table (ibis.Table): An ibis table to filter.
        conditions (dict[str, ibis.expr.types.BooleanColumn]): Dict where keys are labels and
            values are ibis boolean expressions representing filter conditions.

    Returns:
        ibis.Table: An ibis table with filtered rows and an added label column.
    """
    branches = [(condition, ibis.literal(label)) for label, condition in conditions.items()]
    combined_condition = ibis.or_(*[condition for condition, _ in branches])

    return table.filter(combined_condition).mutate(label=ibis.cases(*branches))
