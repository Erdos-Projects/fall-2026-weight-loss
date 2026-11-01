# Weight-Loss Attempts, Behaviors, and Weight Change Among U.S. Adults

**Erdős Institute Data Science Boot Camp — Fall 2026**<br>
**Team:** Thomas Duong · Maggie Ha · William Coleman

## Research Question

> Which behavioral characteristics best predict one-year percentage weight change among U.S. adults who tried to lose weight, **beyond demographics and starting BMI**?

Trying to lose weight does not guarantee losing it. In our sample, only 28% of adults who tried lost 5% or more of their body weight, and about 34% gained weight. We ask whether everyday behaviors (physical activity, sitting time, sleep, mood, smoking, alcohol, and household food roles) help explain who succeeds.

## Data

- **Source:** [NHANES](https://www.cdc.gov/nchs/nhanes/), August 2021 – August 2023 (CDC/NCHS), public-use files.
- **Files merged on respondent ID (`SEQN`):** demographics (`DEMO_L`), weight history (`WHQ_L`), physical activity (`PAQ_L`), diet behavior (`DBQ_L`), sleep (`SLQ_L`), smoking (`SMQ_L`), alcohol (`ALQ_L`), depression screener (`DPQ_L`), and body measures (`BMX_L`). Together these cover 11,933 participants.
- **Outcome:** `pct_wt_change` = (current weight − weight 1 year ago) / weight 1 year ago × 100, from self-reported weights. Negative values mean weight was lost. A binary version, `lost_5pct` (lost ≥ 5%), is also included.

### Analytic sample

| Step | Remaining | Excluded |
|---|---:|---|
| All NHANES participants | 11,933 | |
| Tried to lose weight in the past 12 months | 3,855 | 8,078 did not try or did not answer |
| Aged 20+ | 3,598 | 257 under age 20 |
| Not pregnant | 3,583 | 15 pregnant |
| Both weights valid | 3,480 | 103 missing a weight |
| Change within ±50% | 3,474 | 6 implausible changes |
| **Attended the MEC exam** | **2,794** | 680 interview only |

### Cleaning highlights

- 442 refused or don't-know codes (7/9, 77/99, 777/999, …) were recoded to missing, matched to each question's answer length, so real values such as 7 hours of sleep were kept.
- Skip patterns were filled in rather than imputed: an answer of "none" became 0 or *Never* (for example, vigorous activity went from 51.5% missing to 0.5%).
- Derived variables: activity minutes/week, sedentary hours/day, PHQ-9 score, drinks/week, smoking status, sleep hours/night, and BMI one year ago.
- Measured body-size columns were dropped to avoid outcome leakage. About 12% of values are still missing for PHQ-9, drinks/week and income. Each model handles these itself (median imputation with missingness flags in Model 1; native missing-value handling in Models 2 and 3).

### Predictors (27)

| Group | Variables |
|---|---|
| **Covariates (15)** | Age, sex, race/ethnicity (5 indicators), education, income-to-poverty ratio, marital status (2 indicators), household size, U.S.-born, veteran status, BMI one year ago |
| **Behaviors (12)** | Moderate and vigorous activity (min/week), sedentary hours/day, PHQ-9 depressive symptoms, drinks/week, sleep hours, former/current smoking, four meal-planning and food-shopping roles |

## Approach

Every model is fit **twice**: once on covariates only, and once on covariates + behaviors. The difference in R² (ΔR²) answers the research question.

| Model | Method | Owner | Details |
|---|---|---|---|
| Model 1 | Penalized linear regression (LASSO, Ridge, Elastic Net) with bootstrap selection stability | Thomas | [README](Thomas_Model1_Penalized_Linear_Regression/README.md) |
| Model 2 | Random Forest with permutation importance, SHAP, and Boruta-style feature selection | William | [README](William_Model2_Random_Forest/README.md) |
| Model 3 | Gradient boosting (XGBoost) with permutation importance and SHAP | Maggie | [README](Maggie_Model3_XGBoost/README.md) |

**Shared design**
- Same 80/20 train/test split (2,235 / 559), stratified on outcome quintiles. The test set is used only once.
- All tuning and preprocessing are done on the training set only, inside pipelines, so no information leaks from the test set.
- ΔR² is assessed with a 2,000-resample paired bootstrap on the test set and with repeated 5-fold CV (5 × 5 = 25 folds) on the training set.
- Metrics: R², RMSE, MAE (in percentage points of weight change).

## Results

### Model performance (held-out test set)

| Model | Covariates only R² | + Behaviors R² | + Behaviors RMSE | Test ΔR² [95% CI] |
|---|---:|---:|---:|---|
| Mean-only baseline | −0.002 | — | 9.09 | — |
| LASSO | 0.074 | 0.080 | 8.71 | 0.005 [−0.010, 0.021] |
| Ridge | 0.072 | 0.079 | 8.72 | 0.008 [−0.011, 0.028] |
| Elastic Net | 0.075 | 0.080 | 8.71 | 0.005 [−0.010, 0.021] |
| Random Forest | 0.071 | 0.074 | 8.74 | 0.003 [−0.015, 0.021] |
| XGBoost | 0.065 | 0.075 | 8.74 | 0.010 [−0.015, 0.033] |

### Key findings

1. **Behaviors add a small but consistent gain.** Every model improves when behaviors are added. In repeated CV, R² rises by about 0.006–0.011, and the model with behaviors does better in 19–22 of 25 folds. However, every test-set 95% CI includes zero.
2. **Starting BMI dominates.** In the linear models it is worth about −3.0 percentage points per SD, roughly six times any other predictor; in the tree models, shuffling it alone drops test R² by about 0.20. Higher starting BMI goes with greater percentage loss.
3. **The models broadly agree on which behaviors matter:**
   - **More weight loss:** moderate and vigorous physical activity, being the main meal planner, current smoking.
   - **Less weight loss:** more sedentary time, higher PHQ-9 depressive symptoms, short sleep.
   - **Moderate activity is the most robust behavior.** In both tree models it is the only behavior whose test-set permutation importance is clearly above zero. In the random forest, a model using only BMI, age, income and moderate activity matches the full 27-predictor model.
4. **Non-linear patterns (XGBoost SHAP):** the benefit of moderate activity levels off after about 150–300 min/week. Sedentary time acts like a step, with smaller loss above about 5 h/day. Sleep under about 6.5 h and PHQ-9 scores of 5 or more go with smaller loss. Because the tree models did not beat the linear models, these shapes are descriptive only.
5. **No advantage for flexible models.** All models perform within about 0.015 R² of each other, and overall explanatory power is modest (test R² ≈ 0.07–0.08).

### Weight change in the sample (EDA)

Survey-weighted, 28% of adults who tried to lose weight lost 5% or more, 22% reported no change, and 34% gained. The spike at exactly 0% partly reflects heaping: people often report the same round number for both weights. Among all predictors, starting BMI has by far the strongest correlation with weight change (Spearman r = −0.30). No single behavior exceeds |r| = 0.06 on its own.

## Repository Structure

```
├── datasets/                                    # Raw NHANES .xpt files (Aug 2021 – Aug 2023)
├── clean_up_data/                               # Cleaning notebooks → nhanes_model_ready.csv
├── data_exploration/                            # EDA notebook
├── Thomas_Model1_Penalized_Linear_Regression/   # Model 1: LASSO / Ridge / Elastic Net
├── William_Model2_Random_Forest/                # Model 2: Random Forest + SHAP + Boruta
├── Maggie_Model3_XGBoost/                       # Model 3: XGBoost + SHAP
├── Executive Summary/                           # Executive summary (PDF) of Models 1–3
└── nhanes_model_ready.csv                       # Final modeling dataset (2,794 rows × 33 columns)
```

## How to Reproduce

**1. Install dependencies** (Python 3.10+; Model 2 needs scikit-learn ≥ 1.4 for native missing-value support in random forests):

```bash
pip install pandas numpy scipy "scikit-learn>=1.4" xgboost shap matplotlib seaborn jupyter
```

**2. Clean the data.** Run the notebooks in `clean_up_data/` in order:

| Step | Notebook | Input → Output |
|---|---|---|
| 1 | `weight_loss_data_cleaned.ipynb` | `datasets/*.xpt` → `nhanes_84_col.csv` |
| 2 | `weight_loss_data_cleaning_from_84_to_64.ipynb` | `nhanes_84_col.csv` → `nhanes_64_col.csv` |
| 3 | `weight_loss_cleaning_data_from_64col.ipynb` | `nhanes_64_col.csv` → `nhanes_cleaned.csv` |
| 4 | `data_ready.ipynb` | `nhanes_cleaned.csv` → `nhanes_model_ready.csv` |

You can skip this step: `nhanes_model_ready.csv` is already included at the top level.

**3. Explore and model.** Each notebook reads `nhanes_model_ready.csv`:
- `data_exploration/data_exploration.ipynb`
- `Thomas_Model1_Penalized_Linear_Regression/model1_penalized_regression.ipynb`
- `William_Model2_Random_Forest/model2_random_forest.ipynb`
- `Maggie_Model3_XGBoost/model3_gradient_boosting.ipynb`

All splits and searches use fixed random seeds, and each tuning step takes a few minutes. The Model 3 notebook looks for the CSV in its own folder, so copy `nhanes_model_ready.csv` into `Maggie_Model3_XGBoost/` or set `DATA_PATH = "../nhanes_model_ready.csv"`.

## Limitations

- **Not causal.** NHANES is cross-sectional, and importance measures show what models rely on, not causal effects.
- **Timing.** Behaviors were measured at the exam, at the *end* of the weight-change period, so the direction of influence is unclear.
- **Self-report.** Both weights used in the outcome are self-reported, and many people report round numbers (heaping), which inflates "no change."
- **Unweighted.** Results describe this sample, not the U.S. population. Survey weights can be turned on with `USE_SURVEY_WEIGHTS = True` in the Model 2 and Model 3 notebooks.
- **Modest R².** Most of the variation in one-year weight change remains unexplained.

## Data Citation

National Center for Health Statistics (NCHS). *National Health and Nutrition Examination Survey (NHANES), August 2021–August 2023.* Hyattsville, MD: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/
