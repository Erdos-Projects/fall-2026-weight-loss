# Model 2: Random Forest

## Overview

This analysis asks whether behavioral characteristics improve the prediction of one-year percentage weight change among U.S. adults who tried to lose weight, beyond demographics and starting BMI.

A random forest grows many decision trees, each on a random sample of people and a random subset of predictors at each split, and averages their predictions. Unlike the linear models in Model 1, it can capture **non-linear** relationships and **interactions** without specifying them in advance.

The primary objectives are to:

1. Test whether behaviors improve predictive performance beyond demographics and baseline BMI.
2. Identify which behaviors the forest relies on, using permutation importance and SHAP values.
3. Check which predictors stand out from noise, using a Boruta-style shadow-feature test.

**Notebook:** `model2_random_forest.ipynb`

## Dataset

- **Source:** NHANES, August 2021–August 2023 (`../nhanes_model_ready.csv`)
- **Sample size:** 2,794 U.S. adults who tried to lose weight
- **Outcome:** One-year percentage weight change (`pct_wt_change`). The mean is −1.22% (SD 8.85%) and values range from −40.1% to +50.0%.
- **Training set:** 2,235 participants (80%)
- **Test set:** 559 participants (20%)

Negative values of `pct_wt_change` mean weight was lost; positive values mean weight was gained.

## Predictor Sets

The forest is fitted twice:

- **Covariates only (15):** age, sex, race/ethnicity (5 indicators), education, income-to-poverty ratio, marital status (2 indicators), household size, U.S.-born, veteran status, BMI one year ago.
- **Covariates + behaviors (27):** the 15 above plus moderate and vigorous activity (min/week), sedentary hours/day, PHQ-9 depressive symptoms, drinks/week, sleep hours, former/current smoking, and four meal-planning and food-shopping roles.

## Methodology

### 1. Missing data

About 12% of values are missing for drinks/week, PHQ-9 and income-to-poverty ratio, and less than 1.5% for everything else. scikit-learn's random forest (version ≥ 1.4) handles missing values natively: at each split it learns which side missing values should go to. No imputation is needed.

### 2. Train/test split

An 80/20 split stratified on outcome quintiles, with a fixed random seed. It is the same split the other models use. The test set is used only for final evaluation.

### 3. Hyperparameter tuning

- Randomized search (30 candidates × 5-fold CV, 300 trees each) on the training set, run separately for each predictor set.
- Tuned: `max_depth`, `min_samples_leaf`, `max_features`, `max_samples`.
- The final forests use 1,000 trees with the chosen settings.

| Predictor set | `max_depth` | `min_samples_leaf` | `max_features` | `max_samples` | Best CV RMSE |
|---|---:|---:|---:|---:|---:|
| Covariates only | 4 | 8 | 0.7 | 0.59 | 8.20 |
| Covariates + behaviors | 6 | 19 | 0.7 | 0.99 | 8.16 |

### 4. Evaluation

- Metrics on the test set: R², RMSE, MAE (in percentage points of weight change).
- **Paired bootstrap** (2,000 resamples) on the test set for 95% CIs of ΔR² and ΔRMSE.
- **Repeated 5-fold CV** (5 repeats, 25 folds) on the training set, holding each set's tuned hyperparameters fixed.

### 5. Importance and feature selection

- **Permutation importance** on the test set: the drop in R² when one predictor is shuffled.
- **Grouped permutation:** all 12 behaviors shuffled together vs. all 15 covariates shuffled together.
- **SHAP values** (TreeSHAP): the size and direction of each predictor's contribution for each person, plus dependence plots and SHAP interaction values.
- **Boruta-style shadow-feature test:** 100 rounds, each comparing every predictor's impurity importance to the best of its shuffled "shadow" copies. Hits are tested against a 50% chance rate (binomial test, Bonferroni-corrected across 27 predictors).

## Results

### Model performance (test set)

| Predictor set | Test R² | Test RMSE | Test MAE | CV R² (mean ± SD) |
|---|---:|---:|---:|---:|
| Mean-only baseline | −0.002 | 9.09 | 6.28 | — |
| Covariates only | 0.071 | 8.75 | 6.06 | 0.122 ± 0.028 |
| Covariates + behaviors | **0.074** | **8.74** | 6.09 | **0.128 ± 0.029** |

### Do behaviors add predictive value?

| Comparison | Estimate | 95% CI |
|---|---:|---|
| Test ΔR² | +0.003 | [−0.015, 0.021] |
| Test ΔRMSE | −0.012 | [−0.099, 0.069] |
| Repeated-CV ΔR² (mean of 25 folds) | +0.006 | behaviors better in 19 of 25 folds |

### Key Findings

**1. Behaviors add a small gain that is not statistically certain.**

- Adding behaviors raised test R² by about 0.003 and mean CV R² by about 0.006.
- The model with behaviors did better in 19 of 25 CV folds, but the test-set 95% CI for ΔR² includes zero.
- Shuffling all 12 behaviors together drops test R² by 0.011 (95% CI −0.007 to 0.028). Shuffling all 15 covariates together drops it by 0.225.

**2. Baseline BMI dominates.**

- Shuffling BMI one year ago alone drops test R² by about 0.20, roughly 12 times the next predictor (age, 0.016).
- Its mean |SHAP| is 2.5 percentage points, five times that of any other predictor.
- Most of the strongest SHAP interactions involve starting BMI (with income, age, PHQ-9 and moderate activity).

**3. Moderate physical activity is the most reliable behavior.**

| Behavior | Permutation ΔR² (test) | Mean \|SHAP\| | Boruta |
|---|---:|---:|---|
| Moderate activity (min/week) | **0.010** [0.005, 0.015] | 0.23 | Confirmed |
| Vigorous activity (min/week) | 0.004 [0.001, 0.008] | 0.13 | Rejected |
| PHQ-9 depressive symptoms | −0.004 | 0.30 | Confirmed |
| Sleep hours | −0.000 | 0.13 | Confirmed |
| Sedentary hours/day | 0.001 | 0.08 | Rejected |
| Drinks/week | −0.000 | 0.10 | Rejected |

Moderate activity is the only behavior that the Boruta test confirms *and* that improves test-set performance. PHQ-9 and sleep are used heavily during training (high SHAP, confirmed by Boruta), but shuffling them does not hurt test performance. That suggests the forest partly fits noise with them.

**4. Feature selection.**

The Boruta test confirmed 6 predictors: BMI one year ago, age, income-to-poverty ratio, PHQ-9, moderate activity, and sleep. It rejected the other 21. A forest built on just the 4 predictors that are both confirmed and helpful on the test set (BMI, age, income, moderate activity) reached test R² = 0.079, slightly above the full 27-predictor model's 0.074 (though its CV R² was a little lower, 0.120 vs. 0.128). Nearly all of the forest's predictive power comes from a handful of variables.

**5. Overall predictive performance is modest.**

Test R² is about 0.07. This is in line with the linear models in Model 1, so the forest's flexibility does not buy extra accuracy here.

## Limitations

- **Not causal.** Permutation and SHAP importance describe what the *model* relies on, not effects of the behaviors.
- **Correlated predictors share credit**, for example meal planning and food shopping. The grouped permutation results help with this.
- **Boruta favors continuous predictors.** Impurity importance makes 0/1 indicators (smoking, meal-planner roles) hard to confirm, so a rejection means "not detectably better than noise by this measure," not "no effect."
- **Pulled toward the middle.** Forest predictions are averages over trees and rarely reach extreme weight changes.
- **Timing.** Behaviors were measured at the exam, at the *end* of the weight-change period.
- **Self-report.** Both weights in the outcome are self-reported.
- **Unweighted.** Results describe this sample. Set `USE_SURVEY_WEIGHTS = True` at the top of the notebook for a fit weighted by the MEC exam weights.

## How to Run

```bash
pip install pandas numpy scipy "scikit-learn>=1.4" shap matplotlib seaborn jupyter
jupyter notebook model2_random_forest.ipynb
```

The notebook reads `nhanes_model_ready.csv` from the repository root. The random seeds are fixed. Tuning and the Boruta test take a few minutes; set `N_JOBS` at the top of the notebook to control parallelism.

## Data Source

National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey (NHANES), August 2021–August 2023.

https://www.cdc.gov/nchs/nhanes/
