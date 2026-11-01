# Model 3: Gradient Boosting (XGBoost)

## Overview

This analysis asks whether behavioral characteristics improve the prediction of one-year percentage weight change among U.S. adults who tried to lose weight, beyond demographics and starting BMI.

Gradient boosting builds many small decision trees in sequence, each correcting the errors of the ones before. Like the random forest in Model 2, it can capture **non-linear** relationships and **interactions** (for example, activity mattering more at higher starting BMI) without specifying them in advance.

The primary objectives are to:

1. Test whether behaviors improve predictive performance beyond demographics and baseline BMI.
2. Identify which behaviors the model relies on, using permutation importance and SHAP values.
3. Describe the *shape* of each behavior's relationship with weight change: thresholds, plateaus and interactions that a linear model cannot represent.

**Notebook:** `model3_gradient_boosting.ipynb`

## Dataset

- **Source:** NHANES, August 2021–August 2023 (`nhanes_model_ready.csv`)
- **Sample size:** 2,794 U.S. adults who tried to lose weight
- **Outcome:** One-year percentage weight change (`pct_wt_change`), restricted to ±50%. The mean is −1.22% (SD 8.85%).
- **Training set:** 2,235 participants (80%)
- **Test set:** 559 participants (20%)

Negative values of `pct_wt_change` mean weight was lost; positive values mean weight was gained.

## Predictor Sets

XGBoost is fitted twice:

- **Covariates only (15):** age, sex, race/ethnicity (5 indicators), education, income-to-poverty ratio, marital status (2 indicators), household size, U.S.-born, veteran status, BMI one year ago.
- **Covariates + behaviors (27):** the 15 above plus moderate and vigorous activity (min/week), sedentary hours/day, PHQ-9 depressive symptoms, drinks/week, sleep hours, former/current smoking, and four meal-planning and food-shopping roles.

## Methodology

### 1. Missing data

About 12% of values are missing for drinks/week, PHQ-9 and income-to-poverty ratio, and less than 1.5% for everything else. XGBoost handles missing values natively: at each split it learns which branch missing values should go down. No imputation is needed.

### 2. Train/test split

An 80/20 split stratified on outcome quintiles (`random_state = 42`). It is the same split the other models use. The test set is used only for final evaluation.

### 3. Hyperparameter tuning

Randomized search (60 candidates × 5-fold CV) on the training set, run separately for each predictor set. With a noisy outcome and only about 2,200 training rows, boosting overfits easily. The search therefore covers shallow trees, small learning rates and strong regularization:

| Hyperparameter | Search range | Covariates only | Covariates + behaviors |
|---|---|---:|---:|
| `n_estimators` | 100–800 | 666 | 507 |
| `learning_rate` | 0.005–0.1 (log) | 0.0066 | 0.0122 |
| `max_depth` | 2–5 | 3 | 3 |
| `min_child_weight` | 5–60 | 55 | 39 |
| `subsample` | 0.5–1.0 | 0.93 | 0.75 |
| `colsample_bytree` | 0.4–1.0 | 0.98 | 0.83 |
| `reg_lambda` (L2) | 0.1–50 (log) | 1.82 | 2.40 |
| `reg_alpha` (L1) | 0.001–10 (log) | 8.57 | 4.50 |
| `gamma` | 0–2 | 0.47 | 1.11 |
| **Best CV RMSE** | | 8.18 | 8.12 |

### 4. Evaluation

- Metrics on the test set: R², RMSE, MAE (in percentage points of weight change).
- **Paired bootstrap** (2,000 resamples) on the test set for 95% CIs of ΔR² and ΔRMSE.
- **Repeated 5-fold CV** (5 repeats, 25 folds) on the training set, holding each set's tuned hyperparameters fixed.

### 5. Interpretation

- **Permutation importance** on the test set: the drop in R² when one predictor is shuffled.
- **Grouped permutation:** all 12 behaviors shuffled together vs. all 15 covariates shuffled together.
- **SHAP values** (TreeSHAP): a summary (beeswarm) plot, dependence plots for the top four behaviors, and SHAP interaction values.

## Results

### Model performance (test set)

| Predictor set | Test R² | Test RMSE | Test MAE | CV R² (mean ± SD) |
|---|---:|---:|---:|---:|
| Mean-only baseline | −0.002 | 9.09 | 6.28 | — |
| Covariates only | 0.065 | 8.78 | 6.11 | 0.127 ± 0.030 |
| Covariates + behaviors | **0.075** | **8.74** | 6.11 | **0.138 ± 0.030** |

### Do behaviors add predictive value?

| Comparison | Estimate | 95% CI |
|---|---:|---|
| Test ΔR² | +0.010 | [−0.015, 0.033] |
| Test ΔRMSE | −0.047 | [−0.155, 0.068] |
| Repeated-CV ΔR² (mean of 25 folds) | +0.011 | behaviors better in 22 of 25 folds |

### Key Findings

**1. Behaviors add a small, consistent gain that is not statistically certain.**

- Adding behaviors raised test R² by 0.010 and mean CV R² by 0.011. This is the largest gain among the team's models.
- The model with behaviors did better in 22 of 25 CV folds, but the test-set 95% CI for ΔR² includes zero.
- Shuffling all 12 behaviors together drops test R² by 0.024 (95% CI −0.002 to 0.047). Shuffling all 15 covariates together drops it by 0.246.

**2. Baseline BMI dominates.**

- Shuffling BMI one year ago alone drops test R² by 0.22, about 15 times the next predictor (income-to-poverty ratio, 0.015).
- Its mean |SHAP| is 2.5 percentage points, more than five times that of any other predictor. Higher starting BMI pushes predictions toward greater percentage loss.
- The strongest interactions involving a behavior are with starting BMI: moderate activity × BMI and PHQ-9 × BMI.

**3. Which behaviors matter, and in which direction.**

| Behavior | Permutation ΔR² (test) | Mean \|SHAP\| | Direction (from SHAP) |
|---|---:|---:|---|
| Moderate activity (min/week) | **0.014** [0.004, 0.025] | 0.41 | More activity → more loss, up to a point |
| PHQ-9 depressive symptoms | −0.004 | 0.33 | Higher scores (≥ 5) → less loss |
| Sedentary hours/day | 0.007 [−0.000, 0.013] | 0.28 | More than about 5 h/day → less loss |
| Sleep hours | −0.001 | 0.25 | Under about 6.5 h → less loss |
| Vigorous activity (min/week) | 0.006 [−0.002, 0.013] | 0.24 | More activity → more loss |
| Drinks/week | −0.002 | 0.18 | More drinks → slightly less loss |
| Current smoking | 0.001 | 0.18 | Current smokers → more loss |
| Main meal planner | −0.000 | 0.17 | Main planner → slightly more loss |

Moderate activity is the only behavior whose permutation-importance CI excludes zero. Sedentary time and vigorous activity come next. PHQ-9 and sleep have large SHAP values but do not hurt test performance when shuffled, so their patterns should be read with caution.

**4. Non-linear shapes (SHAP dependence plots).**

- **Moderate activity:** the benefit grows up to roughly 150–300 min/week and then levels off. Very high values add nothing more.
- **Sedentary time:** a step rather than a slope. Predicted loss is greater below about 5 h/day and smaller above it.
- **Sleep:** short sleep (under about 6.5 h) is linked to clearly less loss. Above about 7 h, the effect is flat or slightly favorable.
- **PHQ-9:** scores of 1–4 are linked to slightly more loss, while scores of 5 or more (mild symptoms and above) are linked to less loss.

Because XGBoost did not beat the linear models overall, these shapes are **descriptive** and not confirmed effects.

**5. Overall predictive performance is modest.**

Test R² is about 0.075, close to the penalized linear models (0.080) and the random forest (0.074). The flexibility of boosting does not buy extra accuracy here.

## Limitations

- **Not causal.** Permutation and SHAP importance describe what the *model* relies on, not effects of the behaviors.
- **Correlated predictors share credit**, for example meal planning and food shopping. The grouped permutation results help with this.
- **Timing.** Behaviors were measured at the exam, at the *end* of the weight-change period, so the direction of influence is unclear. For example, people may sleep less *because* they gained weight.
- **Self-report.** Both weights in the outcome are self-reported.
- **Unweighted.** Results describe this sample. Set `USE_SURVEY_WEIGHTS = True` at the top of the notebook to fit and score with the MEC exam weights.

## How to Run

```bash
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib jupyter
jupyter notebook model3_gradient_boosting.ipynb
```

- The notebook reads `nhanes_model_ready.csv` from its own folder (`DATA_PATH`), so copy the file from the repository root or change the path.
- The random seed is fixed (`RANDOM_STATE = 42`). Tuning takes a few minutes.
- **To use LightGBM instead:** replace `xgb.XGBRegressor(...)` with `lightgbm.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)`, and rename `max_depth`/`min_child_weight` in `param_dist` to `num_leaves`/`min_child_samples`. Everything downstream, including SHAP, works unchanged.

## Data Source

National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey (NHANES), August 2021–August 2023.

https://www.cdc.gov/nchs/nhanes/
