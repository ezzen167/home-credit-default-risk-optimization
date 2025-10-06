# Home Credit Default Risk — Project README

## Project Overview
This repository/notebook implements a **Loan Default Risk prediction** pipeline using the *Home Credit Default Risk* dataset. The goal is to predict whether an applicant will default and to **optimize a decision threshold based on business cost** (balancing costs of false positives and false negatives).

Key components:
- Data loading and cleaning
- Feature preprocessing (numeric imputation, categorical encoding)
- Baseline model: Logistic Regression
- Advanced model: CatBoostClassifier (handles categoricals natively)
- Threshold optimization to minimize total business cost (scalar or amount-based using `AMT_CREDIT`)
- Model evaluation (confusion matrix, precision/recall/F1, ROC curves)
- CatBoost feature importance

---

## Files
- `application_train.csv` — (not included) main dataset (from Kaggle: Home Credit Default Risk)
- `notebook.ipynb` — (your uploaded Jupyter notebook) contains the full pipeline and results
- `catboost_model.cbm` — (optional) saved CatBoost model file if exported
- `lr_pipeline.pkl` — (optional) saved Logistic Regression pipeline
- `catboost_feature_importance.csv` — (optional) exported feature importance table

> Note: The large dataset files are not included in the repo. Download from Kaggle: https://www.kaggle.com/competitions/home-credit-default-risk

---

## Environment & Requirements
Use a Python 3.9+ environment. Recommended to run in Google Colab or local Jupyter Lab.

Install required packages (one-liner):

```bash
pip install -r requirements.txt
# or manually:
pip install pandas numpy scikit-learn matplotlib seaborn catboost joblib
```

Example `requirements.txt` (suggested):
```
pandas
numpy
scikit-learn
matplotlib
seaborn
catboost
joblib
```

---

## How to Run (quick start)
1. Place `application_train.csv` in the same directory as the notebook (or update the path in the notebook).
2. Open `notebook.ipynb` in Jupyter/Colab.
3. Run the notebook cells in order. The notebook is split into sections:
   - Data load & EDA
   - Data cleaning (drop columns with high missingness)
   - Train/validation split
   - Preprocessing pipelines
   - Logistic Regression training (baseline)
   - CatBoost training
   - Threshold optimization (business cost sweep)
   - Model evaluation & ROC
   - CatBoost feature importance
   - Save models & final summary

### Running in Google Colab (recommended for large dataset):
- Upload `kaggle.json` (Kaggle API token) and use the Kaggle API to download the dataset directly to the Colab VM (faster for big files).
- Or upload `application_train.csv` manually via the Colab file browser.

---

## Business Cost Optimization
The notebook implements two ways to quantify business cost:
- **Scalar costs**: fixed cost per false positive and false negative (e.g., `cost_fp = 500`, `cost_fn = 20000`).
- **Amount-based costs**: cost proportional to `AMT_CREDIT` (recommended). Example: `cost_fn = LGD * AMT_CREDIT`, `cost_fp = profit_rate * AMT_CREDIT`.

A threshold sweep calculates total business cost across thresholds and selects the threshold that minimizes expected cost. This threshold is stored and used to convert probabilities to decisions.

---

## Results (example summary)
- Logistic Regression ROC-AUC: ~0.749
- CatBoost ROC-AUC: ~0.760
- Optimal thresholds found during the run (example):
  - Logistic Regression best threshold: 0.655 (approx)
  - CatBoost best threshold: 0.125 (approx)
- CatBoost produced slightly lower expected total business cost and better AUC; recommended model for deployment.

Include your exact numbers in the notebook output cells for reproducible reporting.

---

## Tips & Notes
- The dataset is large (~300k rows). Preprocessing with OneHotEncoder can produce many columns — CatBoost often handles this more efficiently.
- Avoid feature leakage: do not include columns that contain future repayment behavior when predicting default at application time.
- Use cross-validation for more stable estimates of optimal threshold; the notebook uses a holdout validation split.
- Consider per-segment thresholds (by income buckets or product types) for better business outcomes.

---

## Citation
Kaggle Competition: *Home Credit Default Risk* (https://www.kaggle.com/competitions/home-credit-default-risk)

---

## License
This project content is provided for educational purposes. You can reuse and adapt it. If you include this in a public repository, consider adding an appropriate license (e.g., MIT).

---

## Next steps I can help with
- Generate a `requirements.txt` file and add it to the repo (done here).
- Create a cleaned `README.md` file saved to your workspace (this file).
- Produce a short printable project summary paragraph for inclusion in your report.
