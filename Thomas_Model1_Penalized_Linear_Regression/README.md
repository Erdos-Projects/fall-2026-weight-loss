# Model 1: Penalized Linear Regression (LASSO, Ridge, and Elastic Net)

## Overview

This analysis investigates whether behavioral characteristics improve the prediction of one-year percentage weight change among U.S. adults who attempted to lose weight.

Using data from the **National Health and Nutrition Examination Survey (NHANES), August 2021–August 2023**, we compare three penalized regression methods: LASSO, Ridge, and Elastic Net.

The primary objectives are to:

1. Evaluate whether behavioral factors improve predictive performance beyond demographic characteristics and baseline BMI.
2. Compare the predictive performance of LASSO, Ridge, and Elastic Net regression.
3. Identify the behavioral predictors most consistently associated with percentage weight change.

## Dataset

- **Source:** NHANES, August 2021–August 2023
- **Sample size:** 2,794 U.S. adults who attempted to lose weight
- **Outcome:** One-year percentage weight change (`pct_wt_change`)
- **Training set:** 2,235 participants (80%)
- **Testing set:** 559 participants (20%)

Negative values of `pct_wt_change` indicate weight loss, while positive values indicate weight gain.

## Predictor Variables

Each penalized regression method was evaluated using two predictor sets.

### Covariates Only (15 Predictors)

Includes demographic and baseline characteristics, such as:

- Age and sex
- Race/ethnicity
- Educational attainment
- Income-to-poverty ratio
- Marital status
- Household size
- U.S.-born status
- Veteran status
- BMI one year ago

### Covariates + Behaviors (27 Predictors)

Includes the 15 covariates plus 12 behavioral predictors:

- Moderate and vigorous physical activity (minutes/week)
- Sedentary time (hours/day)
- Depressive symptoms (PHQ-9)
- Alcohol consumption (drinks/week)
- Sleep duration
- Smoking status
- Household meal-planning responsibilities
- Household food-shopping responsibilities

Comparing these predictor sets allows us to evaluate the additional predictive value of behavioral characteristics.

## Methodology

### 1. Data Preprocessing

- Performed an 80/20 train-test split, balanced on the outcome.
- Imputed missing values using training-set medians.
- Added missingness indicators for income, PHQ-9, and alcohol consumption.
- Standardized predictors to mean 0 and standard deviation 1.
- Applied preprocessing within model pipelines to prevent data leakage.

### 2. Penalized Regression Methods

**LASSO Regression (L1 Penalty)**

- Shrinks coefficients toward zero.
- Performs variable selection by setting some coefficients to exactly zero.

**Ridge Regression (L2 Penalty)**

- Shrinks coefficients toward zero while retaining all predictors.
- Helps manage correlated predictors.

**Elastic Net Regression (L1 + L2 Penalties)**

- Combines LASSO and Ridge penalties.
- Balances coefficient shrinkage and variable selection.

Hyperparameters were selected using **10-fold cross-validation** on the training set.

### 3. Model Evaluation

Predictive performance was evaluated using:

- R-squared (R²)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Additional analyses included:

- Paired bootstrap with 2,000 resamples to construct 95% confidence intervals for performance differences.
- Repeated 5-fold cross-validation with 5 repetitions (25 folds).
- LASSO coefficient-path analysis.
- Bootstrap variable-selection stability using 200 resamples.
- Residual diagnostics and observed-versus-predicted plots.

## Results

### Model Performance

| Regression Method | Predictor Set | Test R² | Test RMSE |
|---|---|---:|---:|
| LASSO | Covariates Only | 0.074 | 8.74 |
| LASSO | Covariates + Behaviors | **0.080** | **8.71** |
| Ridge | Covariates Only | 0.072 | 8.75 |
| Ridge | Covariates + Behaviors | 0.079 | 8.72 |
| Elastic Net | Covariates Only | 0.075 | 8.74 |
| Elastic Net | Covariates + Behaviors | **0.080** | **8.71** |

### Key Findings

**1. Behavioral characteristics provided modest predictive improvements.**

- Adding behavioral predictors increased test R² by approximately 0.005–0.008.
- Repeated cross-validation showed an average R² improvement of approximately 0.008.
- The additional behavioral variables improved predictive performance in 20–21 of 25 cross-validation folds.
- However, the 95% bootstrap confidence intervals included zero, indicating that the improvement was not statistically certain.

**2. Baseline BMI was the strongest predictor.**

- Baseline BMI had the largest standardized regression coefficient (approximately −3.0).
- Higher baseline BMI was associated with greater percentage weight loss.

**3. Several behavioral characteristics showed stable associations with weight change.**

Predictors associated with **greater weight loss** included:

- Being the primary household meal planner
- Current smoking status
- Greater vigorous physical activity

Predictors associated with **less weight loss** included:

- Greater sedentary time
- Higher depressive symptom scores (PHQ-9)
- Missing alcohol-consumption information

These predictors were selected in at least 88% of LASSO bootstrap resamples with consistent coefficient directions.

**4. Overall predictive performance was modest.**

The models explained a relatively small proportion of variation in one-year percentage weight change, with test R² values of approximately 0.07–0.08.

## Limitations

- The observational design does not support causal conclusions.
- Behavioral characteristics were measured at the end of the weight-change period, making temporal relationships unclear.
- Self-reported weight measurements may introduce measurement error.
- The analyses were unweighted, so the results describe the study sample rather than nationally representative estimates.
- The modest R² values indicate that substantial variation in weight change remains unexplained.

## Conclusion

Adding behavioral characteristics to demographic and baseline BMI predictors provided small but relatively consistent improvements in predicting one-year percentage weight change among adults attempting to lose weight.

Baseline BMI was the strongest predictor, while vigorous physical activity, sedentary behavior, depressive symptoms, and household meal-planning responsibilities demonstrated relatively stable associations with weight change.

Overall, behavioral information may contribute additional predictive value, although the magnitude of improvement was modest and statistically uncertain.

## Data Source

National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey (NHANES), August 2021–August 2023.

https://www.cdc.gov/nchs/nhanes/
