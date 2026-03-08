---
title: Metrics
social:
  cards_layout_options:
    title: PyRetailScience | Retail Metrics
---

## Distribution Metrics

### ACV (All Commodity Volume)

ACV measures the total dollar sales across all products in a set of stores, expressed in millions ($MM). It is commonly
used in retail analytics to quantify the size of a store or group of stores.

$$
\text{ACV} = \frac{\sum \text{unit_spend}}{\text{acv_scale_factor}}
$$

By default, `acv_scale_factor` is 1,000,000 (expressing ACV in $MM).

Example:

```python
import pandas as pd
from pyretailscience.metrics.distribution.acv import Acv

df = pd.DataFrame({
    "store_id": [101, 101, 102, 102, 103],
    "unit_spend": [400_000, 600_000, 300_000, 200_000, 500_000],
})

acv = Acv(df, group_by="store_id")
print(acv.df)
#    store_id  acv
# 0       101  1.0
# 1       102  0.5
# 2       103  0.5
```
