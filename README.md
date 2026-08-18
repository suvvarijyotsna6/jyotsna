# Retail Multi-Segment Profiler & High-Value Customer Classifier — Prototype

A working Flask app that reproduces the internship case study end-to-end on the
real `Mall_Customers.csv` dataset (n=200):

- **K-Means segmentation** (K=5, chosen via elbow + silhouette) into five personas,
  built from standardised Annual Income + Spending Score.
- **Random Forest classifier** predicting a "High-Value Customer" label from
  **Age, Gender and Annual Income only** (spending score is deliberately excluded
  to avoid leakage and to mirror scoring a brand-new sign-up with no purchase history).
- A **live dashboard** ("Floor Board") showing EDA, cluster diagnostics, persona
  cards with recommended strategy, and model performance (accuracy, precision,
  recall, F1, ROC-AUC, confusion matrix, feature importance).
- A **scoring tool** ("Score a Customer") where you enter age/gender/income (and
  optionally a spending score) and get a live high-value probability + persona
  placement, computed by the actual trained models — not hard-coded.

## Project layout

```
prototype/
├── Mall_Customers.csv       # source dataset
├── train.py                 # trains KMeans + RandomForest + LogisticRegression, writes artifacts.json + *.joblib
├── generate_charts.py       # renders all dashboard charts from artifacts.json into static/charts/
├── app.py                   # Flask application (routes: / and /predict)
├── templates/                # Jinja2 HTML templates
├── static/style.css         # visual design (ledger / price-tag theme)
├── static/charts/           # generated PNG charts (pre-built, ready to serve)
├── artifacts.json           # all computed stats/metrics consumed by the templates
├── customers_scored.csv     # original data + Cluster/Persona/High_Value columns
└── model_*.joblib, scaler_cluster.joblib   # trained model files used by app.py
```

## Run it

```bash
pip install flask scikit-learn pandas numpy joblib matplotlib
python3 train.py              # optional — artifacts are already generated
python3 generate_charts.py    # optional — charts are already generated
python3 app.py
```

Then open **http://127.0.0.1:5000**.

## Notes / how it maps to the report

- Segmentation features: `Annual Income (k$)`, `Spending Score (1-100)`, standardised, K=5.
- High-Value label: top 30% by `0.5*normalised income + 0.5*normalised spending`.
- Classifier features: `Age`, `Gender_Code`, `Annual Income (k$)` (class-balanced Random Forest, 300 trees, max depth 6), benchmarked against a Logistic Regression baseline.
- Because this build trains on the *actual* uploaded CSV rather than the report's
  screenshots, the exact metric values differ slightly from the PDF (e.g. accuracy
  here is ~78% vs. 84% in the report) — the persona sizes/averages match closely
  since clustering is deterministic given the same data and `random_state`.
- This is a local prototype (Flask dev server) — not hardened for production
  deployment, authentication, or a real CRM integration (see the report's
  "Limitations & Future Work" section for next steps).
