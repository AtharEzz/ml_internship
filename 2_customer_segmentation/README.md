# Customer Segmentation

Identifying distinct customer groups from behavioral data to support targeted marketing, as part of the Elevvo Pathways Machine Learning Internship.

## Problem

Treating all customers the same wastes marketing spend and misses opportunities to tailor offers to what different groups actually value. The goal was to segment customers into meaningful, actionable groups based on income and spending behavior.

## Approach

**K-Means Clustering**
- Used the elbow method (WCSS across k=1 to 10) to determine the optimal number of clusters: **k=5**
- Fit K-Means on standardized income and spending score features
- Profiled each of the 5 resulting clusters by size, average income, and average spending score, and translated each into a specific, actionable marketing recommendation (e.g., loyalty programs for the largest middle-value segment, premium offerings for the high-income/high-spending segment, installment plans for the low-income/high-spending segment)

**DBSCAN (Comparison)**
- Ran DBSCAN as a second clustering approach to validate the K-Means results
- Searched across multiple `eps` values (0.1–0.6) to tune cluster count and outlier detection
- Final DBSCAN configuration (eps=0.4) produced 4 clusters plus 15 outliers — customers who didn't cleanly fit any segment (e.g., low income with high spending, or very high income with very low spending), flagged as worth further investigation

## Decision: K-Means vs. DBSCAN

Comparing cluster assignments between the two methods showed a **46% disagreement rate** — nearly half of customers were assigned to different clusters depending on the method used.

**K-Means was selected as the final approach** because:
- No outliers — every customer is assigned a clear segment, easier to explain to stakeholders
- Spherical cluster shapes aligned well with the underlying business logic (income/spending tiers)
- Simpler to communicate and deploy for marketing use

DBSCAN's flagged outliers remain a useful secondary signal — these customers may represent high-value or high-risk cases worth a closer, individual look outside the standard segmentation.

## Results

| Cluster | Profile | Size | Recommended Strategy |
|---|---|---|---|
| 0 | Middle income, middle spending | 81 | Loyalty programs, seasonal promotions |
| 1 | High income, highest spending | 39 | High-end marketing, premium products, loyalty rewards |
| 2 | Lowest income, high spending | 22 | Installment plans |
| 3 | Highest income, lowest spending ("Savers") | 35 | Bundles and discounts to encourage spending |
| 4 | Low income, low spending | — | Value-focused offers |

## Folder Structure

- `data/` — dataset files
- `notebooks/` — analysis and clustering notebook(s)
- `reports/` — saved charts and comparison outputs
