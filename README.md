# Delhi Air Quality - AQI Category Classification

BITS Pilani WILP | M.Tech (AI/ML) | Machine Learning Assignment

- **GitHub Repository:** https://github.com/amitdeswal26-hub/delhi-aqi-classifier
- **Live Streamlit App:** https://delhi-aqi-classifier-bits.streamlit.app

---

## 1. Problem Statement

Classify a day's air quality in Delhi into one of six CPCB categories -
Good, Satisfactory, Moderate, Poor, Very Poor, Severe - using the
measured pollutant concentrations for that day.

Five classification models are trained on the same dataset, compared
across six evaluation metrics, and served through a Streamlit web app.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| Source | [Air Quality Data in India (2015-2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india), Kaggle (CPCB data) |
| File used | `city_day.csv`, filtered to Delhi |
| Instances | 1,999 |
| Features | 15 |
| Target | `AQI_Bucket` (6 classes) |
| Task | Multi-class classification |

**Features (15):** 12 measured pollutants - PM2.5, PM10, NO, NO2, NOx,
NH3, CO, SO2, O3, Benzene, Toluene, Xylene - plus 3 features derived
from the date: Month, DayOfWeek and Season.

**Class distribution:** Poor 542, Very Poor 520, Moderate 519,
Severe 239, Satisfactory 158, Good 21. The classes are imbalanced, so
all splits are stratified and all metrics are weighted averages.

**Note on leakage:** the raw `AQI` column was excluded from the
features. `AQI_Bucket` is derived directly from `AQI` by a fixed CPCB
banding rule, so including it would let every model score near 100% and
make the comparison meaningless.

**Preprocessing:** missing values imputed with the column median,
features standardised with `StandardScaler`, target integer-encoded,
80/20 stratified train-test split with `random_state=42`.

---

## 3. GitHub Repository Link

https://github.com/amitdeswal26-hub/delhi-aqi-classifier

---

## 4. Models Used

| Model | Parameters |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=10` |
| kNN | `n_neighbors=7` |
| Naive Bayes | Gaussian |
| Random Forest (Ensemble) | `n_estimators=300` |

Gaussian Naive Bayes was used rather than Multinomial because the
features are continuous and standardised.

### Comparison of evaluation metrics

AUC is one-vs-rest; Precision, Recall and F1 are weighted averages.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7350 | 0.9425 | 0.7371 | 0.7350 | 0.7349 | 0.6560 |
| Decision Tree | 0.7175 | 0.8181 | 0.7193 | 0.7175 | 0.7161 | 0.6331 |
| kNN | 0.7000 | 0.8994 | 0.7020 | 0.7000 | 0.7002 | 0.6104 |
| Naive Bayes | 0.6725 | 0.9143 | 0.6940 | 0.6725 | 0.6743 | 0.5875 |
| **Random Forest (Ensemble)** | **0.7800** | **0.9463** | **0.7826** | **0.7800** | **0.7803** | **0.7138** |

---

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best overall, with an AUC (0.9425) almost matching Random Forest. The AQI categories are ordered bands that rise with pollutant concentration, so linear boundaries separate them well. |
| Decision Tree | Acceptable accuracy but the worst AUC by a wide margin (0.8181). A single tree gets its probabilities from leaf class counts, which are coarse and poorly calibrated, so its ranking quality suffers even where its predictions are correct. |
| kNN | Weakest accuracy after Naive Bayes (0.7000). With 15 features, distances become less discriminative, and the neighbourhood vote is dominated by the three large classes. |
| Naive Bayes | Lowest accuracy (0.6725) and lowest MCC. Its independence assumption is badly violated here - NO, NO2 and NOx are near-deterministically related, as are PM2.5 and PM10 - so correlated evidence is double-counted. |
| Random Forest (Ensemble) | Best on all six metrics. Compared against the single Decision Tree, averaging 300 trees lifts AUC from 0.8181 to 0.9463, which is the clearest demonstration of what ensembling adds. |
| **Overall Winner** | **Random Forest (Ensemble)** - highest on every metric, with an MCC of 0.7138. MCC is the most reliable single figure here because it is not inflated by the dominant classes. |

---
