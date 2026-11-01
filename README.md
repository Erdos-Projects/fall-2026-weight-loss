# Weight-Loss Attempts, Behaviors, and Weight Change Among U.S. Adults

**Erdős Institute Data Science Boot Camp — Fall 2026**
**Team:** Thomas Duong · Maggie Ha · William Coleman

## Research Question

> Which behavioral characteristics best predict one-year percentage weight change among U.S. adults who tried to lose weight, **beyond demographics and starting BMI**?

Trying to lose weight does not guarantee losing it — about 34% of people in our sample gained weight despite trying. We ask whether everyday behaviors (physical activity, sitting time, sleep, mood, smoking, alcohol, and household food roles) help explain who succeeds.

## Data

- **Source:** [NHANES](https://www.cdc.gov/nchs/nhanes/), August 2021 – August 2023 (CDC/NCHS), public-use files.
- **Files merged on respondent ID (`SEQN`):** demographics (`DEMO_L`), weight history (`WHQ_L`), physical activity (`PAQ_L`), diet behavior (`DBQ_L`), sleep (`SLQ_L`), smoking (`SMQ_L`), alcohol (`ALQ_L`), depression screener (`DPQ_L`), and body measures (`BMX_L`) — 11,933 participants in total.
- **Analytic sample:** 2,794 adults aged 20+ who reported trying to lose weight in the past 12 months and attended the MEC exam. Pregnant women and 6 people with changes beyond ±50% were excluded.
- **Outcome:** `pct_wt_change` = (current weight − weight 1 year ago) / weight 1 year ago × 100, from self-reported weights. Negative = weight lost.

### Predictors (27)

| Group | Variables |
|---|---|
| **Covariates (15)** | Age, sex, race/ethnicity (5 indicators), education, income-to-poverty ratio, marital status (2 indicators), household size, U.S.-born, veteran status, BMI one year ago |
| **Behaviors (12)** | Moderate and vigorous activity (min/week), sedentary hours/day, PHQ-9 depressive symptoms, drinks/week, sleep hours, former/current smoking, four meal-planning and food-shopping roles |

## Approach

Every model is fit **twice** — once on covariates only, once on covariates + behaviors. The difference in R² (ΔR²) answers the research question.

| Model | Method | Owner |
|---|---|---|
| Model 1 | Penalized linear regression (LASSO, Ridge, Elastic Net) | Thomas |
| Model 2 | Gradient boosting (XGBoost) with SHAP | Maggie |
| Models 3–4 | OLS linear regression and tuned Random Forest | William |

**Shared design**
- Same 80/20 train/test split (2,235 / 559), stratified on outcome quintiles; the test set is used once.
- All tuning and preprocessing done on the training set only, inside pipelines (no leakage).
- ΔR² assessed with a 2,000-resample paired bootstrap on the test set and repeated 5-fold CV (25 folds) on the training set.
- Metrics: R², RMSE, MAE (percentage points of weight change).

## Results

### Model performance (held-out test set)

| Model | Covariates only R² | + Behaviors R² | + Behaviors RMSE | Test ΔR² [95% CI] |
|---|---:|---:|---:|---|
| Mean-only baseline | −0.002 | — | 9.09 | — |
| LASSO | 0.074 | 0.080 | 8.71 | 0.005 [−0.010, 0.021] |
| Ridge | 0.072 | 0.079 | 8.72 | 0.008 [−0.011, 0.028] |
| Elastic Net | 0.075 | 0.080 | 8.71 | 0.005 [−0.010, 0.021] |
| XGBoost | 0.065 | 0.075 | 8.74 | 0.010 [−0.015, 0.033] |
| OLS | 0.068 | 0.085 | 8.69 | 0.017 [−0.003, 0.037] |
| Random Forest | 0.073 | 0.081 | 8.71 | 0.008 [−0.008, 0.027] |

### Key findings

1. **Behaviors add a small but consistent gain.** Every model improves when behaviors are added (CV R² up ~0.005–0.011; improved in 18–22 of 25 folds), but all test-set 95% CIs include zero.
2. **Starting BMI dominates.** About −3.0 percentage points per SD in the linear models — roughly six times any other predictor. Higher starting BMI goes with greater percentage loss.
3. **The models agree on which behaviors matter:**
   - **More weight loss:** being the main meal planner, moderate and vigorous physical activity, current smoking.
   - **Less weight loss:** more sedentary time, higher PHQ-9 depressive symptoms, short sleep.
4. **Non-linear patterns (XGBoost/SHAP):** the benefit of moderate activity levels off after ~150–300 min/week; less loss above ~5 sedentary hours/day and below ~6.5 hours of sleep. Because XGBoost did not beat the linear models, these shapes are descriptive.
5. **No clear advantage for flexible models.** All models perform within ~0.02 R² of each other; overall explanatory power is modest (test R² ≈ 0.07–0.09).

## Repository Structure

```
├── datasets/                         # Raw NHANES .xpt files (2021–2023)
├── clean_up_data/                    # Cleaning pipeline → nhanes_model_ready.csv
├── data_exploration/                 # EDA notebook
├── Thomas_Model1_Penalized_Linear_Regression/   # Model 1: LASSO / Ridge / Elastic Net + report
├── Maggie_Model2_XGBoost/            # Model 2: XGBoost + SHAP + report
├── william_models_3_4/               # Models 3–4: OLS and Random Forest
├── william/                          # Early exploration, evaluation plan, preliminary results
├── Executive Summary/                # Executive summary (PDF) and Models 3–4 additions
├── Presentation/                     # Slides: data cleaning, Models 1–4
├── model proposal.pdf                # Original modeling proposal
└── nhanes_model_ready.csv            # Final modeling dataset (2,794 × 33)
```

## How to Reproduce

**1. Install dependencies** (Python 3.10+):

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

**2. Clean the data** — run the notebooks in `clean_up_data/` in order:

| Step | Notebook | Input → Output |
|---|---|---|
| 1 | `weight_loss_data_cleaned.ipynb` | `datasets/*.xpt` → `nhanes_84_col.csv` |
| 2 | `weight_loss_data_cleaning_from_84_to_64.ipynb` | `nhanes_84_col.csv` → `nhanes_64_col.csv` |
| 3 | `weight_loss_cleaning_data_from_64col.ipynb` | `nhanes_64_col.csv` → `nhanes_cleaned.csv` |
| 4 | `data_ready.ipynb` | `nhanes_cleaned.csv` → `nhanes_model_ready.csv` |

Cleaning recoded refused/don't-know answers to missing, filled in implied skip patterns, and dropped measured body-size columns to avoid outcome leakage.

**3. Run the models** — each notebook reads `nhanes_model_ready.csv`:
- `Thomas_Model1_Penalized_Linear_Regression/model1_penalized_regression.ipynb`
- `Maggie_Model2_XGBoost/model2_gradient_boosting.ipynb`
- `william_models_3_4/models3_4_linear_random_forest.ipynb`

## Limitations

- **Not causal.** NHANES is cross-sectional; importance measures show what models rely on, not effects.
- **Timing.** Behaviors were measured at the exam, at the *end* of the weight-change period.
- **Self-report.** Both weights used in the outcome are self-reported.
- **Unweighted.** Results describe this sample, not the U.S. population (survey weights can be turned on with `USE_SURVEY_WEIGHTS = True` in the Model 2 notebook).
- **Modest R².** Most of the variation in one-year weight change remains unexplained.

## Data Citation

National Center for Health Statistics (NCHS). *National Health and Nutrition Examination Survey (NHANES), August 2021–August 2023.* Hyattsville, MD: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

