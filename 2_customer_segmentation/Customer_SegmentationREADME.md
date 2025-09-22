## K-Means vs DBSCAN: Final Comparison

Cluster Comparison

**Key Observations:**
- K-Means: 5 clusters, 0 outliers, spherical shapes
- DBSCAN: 4 clusters + 15 outliers, flexible shapes
- **Disagreement Rate: 46%** of customers assigned to different clusters

**Analysis:**
- Both methods revealed meaningful segments.
- K-Means produced cleaner, more interpretable clusters with no noise.
- DBSCAN identified 15 outliers, potentially high-value or risky customers worth investigating.

**Decision:** Chose K-Means for final segmentation because:
- No outliers, easier to explain to stakeholders
- Spherical clusters align with business logic
- Simpler to deploy and communicate

