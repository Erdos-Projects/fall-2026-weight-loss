
# NHANES 2021–2023: Model Evaluation Plan

## 1. Project Objective

The primary objective is to evaluate how accurately demographic, socioeconomic, behavioral, and health-related characteristics predict self-reported percentage weight change among adults aged 20 years and older who attempted to lose weight during the previous 12 months.

The project uses NHANES 2021–2023 data and supervised regression models. Because the data are cross-sectional, the analysis focuses on retrospective prediction of reported weight change rather than forecasting future weight loss.

From a goal-striving and self-regulation perspective, attempting to lose weight does not necessarily translate into successful weight loss. We are interested in identifying characteristics that help predict differences in weight-change outcomes among individuals who share a weight-loss goal.

Preliminary models were initially evaluated among all adults aged 20 and older. The primary research question was subsequently refined to focus specifically on weight-loss attempters.

## 2. Unit of Analysis and Outcome

Each observation represents one NHANES participant, uniquely identified by `SEQN`.

The revised analytic sample includes 3,493 adults aged 20 years and older who reported attempting to lose weight during the previous 12 months (`WHQ070 = 1`) and had valid percentage weight-change data.

The outcome variable is continuous percentage weight change:

Percentage Weight Change = [(Current Weight − Weight One Year Ago) / Weight One Year Ago] × 100

Negative values indicate weight loss, positive values indicate weight gain, and zero indicates no reported weight change.

The initial model contains 16 candidate predictors. Weight-loss attempt status is excluded as a predictor because it is now an inclusion criterion.

### Initial Predictor Domains

- Demographics: Age, sex, education, marital status, race/ethnicity.
- Socioeconomic: Family income-to-poverty ratio, employment status.
- Physical activity: Moderate and vigorous leisure-time physical activity, sitting time.
- Sleep: Average sleep duration.
- Smoking: Smoking status.
- Medical: Medical condition count, hypertension history, high cholesterol history.
- Weight history: Estimated baseline BMI.

## 3. Training and Test Split

The revised analytic sample was randomly divided into:

- Training set: 2,794 participants (80%).
- Test set: 699 participants (20%).

The split was generated using scikit-learn's `train_test_split()` with `test_size=0.20`, `random_state=440`, and `shuffle=True`.

Five-fold cross-validation is performed exclusively within the training dataset.

The 699 test participants remain held aside during model development, feature selection, and hyperparameter tuning. The final selected model will be evaluated once on the held-out test dataset.

Because the research question was refined after preliminary exploration, the revised test set is an internal holdout rather than fully independent external validation.

### NHANES Sampling Considerations

NHANES uses a complex, stratified, multistage probability sampling design.

The preliminary random split does not explicitly account for survey stratification, clustering, or unequal selection probabilities.

Relevant survey-design variables include:

- `WTINT2YR`: Interview sampling weights.
- `WTMEC2YR`: Examination sampling weights.
- `SDMVSTRA`: Masked variance stratum.
- `SDMVPSU`: Masked primary sampling unit.

The appropriate weighting and validation strategy will be determined based on the final analytic variables and intended generalization population.

Population-level predictive performance will not be claimed from the preliminary unweighted results.

## 4. Cross-Validation and Preprocessing

Five-fold cross-validation is used within the training dataset.

In each fold, approximately 80% of training participants are used to fit the model, and the remaining 20% are used for validation.

Preprocessing is performed within scikit-learn pipelines:

- Numerical predictors: Median imputation and standardization.
- Categorical predictors: Most-frequent imputation and one-hot encoding.
- Unknown categories: Handled using `OneHotEncoder(handle_unknown="ignore")`.

Preprocessing parameters are estimated using only the training portion of each cross-validation fold to prevent leakage into validation data.

Model comparisons and feature-selection analyses are conducted using training data only.

## 5. Evaluation Metrics

Three regression metrics are used:

**Mean Absolute Error (MAE):** The average absolute difference between predicted and observed percentage weight change, expressed in percentage points.

**Root Mean Squared Error (RMSE):** A prediction-error metric that places greater emphasis on large deviations, also expressed in percentage points.

**R²:** Predictive performance relative to predicting the mean outcome.

Lower MAE and RMSE indicate better predictive performance, while higher R² indicates better performance.

MAE and RMSE are the primary metrics, with R² used as a supplementary measure.

## 6. Preliminary Baseline Models

### Primary Analysis: Weight-Loss Attempters

Three regression models were evaluated using five-fold cross-validation on the revised training dataset (N = 2,794).

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 6.463 | 9.457 | -0.001 |
| Linear Regression | 6.189 | 8.781 | 0.137 |
| Random Forest | 6.178 | 8.769 | 0.138 |

Both Linear Regression and Random Forest improved performance relative to the dummy baseline.

Random Forest achieved the lowest prediction errors, although its improvement over Linear Regression was minimal.

Overall predictive performance remained modest.

### Preliminary Exploratory Analysis: All Adults Aged 20+

Before refining the research question, the same three modeling approaches were evaluated among all adults aged 20 and older (training N = 6,071).

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 5.408 | 8.646 | -0.001 |
| Linear Regression | 5.425 | 8.177 | 0.105 |
| Random Forest | 5.363 | 8.171 | 0.106 |

These results are retained as preliminary exploratory analyses. Because the populations and predictor sets differ, performance metrics should not be interpreted as a direct comparison of model quality.

All reported results are unweighted training-set cross-validation estimates.

## 7. Feature Selection and Predictor Evaluation

Feature selection combined domain knowledge, permutation importance, and comparisons of alternative predictor sets.

### 7.1 Permutation Importance

Permutation importance was evaluated for the 16-predictor Random Forest using a separate validation subset of the training data.

The largest importance values were:

| Predictor | Importance |
|---|---:|
| Baseline BMI | 1.516 |
| Income-to-poverty ratio | 0.199 |
| Age | 0.169 |
| Vigorous LTPA | 0.094 |
| Smoking status | 0.069 |
| Moderate LTPA | 0.034 |

Importance represents the average increase in validation RMSE when a predictor was shuffled.

Baseline BMI was substantially more influential than the other predictors.

Permutation importance reflects model reliance rather than causal effects. Correlated predictors and validation-sample variability may affect importance estimates.

### 7.2 Predictor Subset Comparisons

Four Random Forest models were compared using five-fold cross-validation.

| Feature Set | N Predictors | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Baseline BMI only | 1 | 6.550 | 9.124 | 0.068 |
| BMI + Age + Poverty Ratio | 3 | 6.381 | 8.932 | 0.107 |
| BMI + Age + Poverty + LTPA + Smoking | 6 | 6.273 | 8.845 | 0.123 |
| All 16 predictors | 16 | 6.178 | 8.769 | 0.138 |

Predictive performance improved as additional predictor groups were included.

The full 16-predictor model achieved the best performance, although the six-predictor model performed reasonably similarly with substantially fewer variables.

### 7.3 Additional Predictor Evaluation

Several previously excluded predictors were evaluated for incremental predictive value.

| Additional Predictor | MAE | RMSE | R² |
|---|---:|---:|---:|
| Original 16 predictors | 6.178 | 8.769 | 0.138 |
| + PHQ-9 depressive symptoms | 6.191 | 8.770 | 0.138 |
| + Currently dieting | 6.174 | 8.769 | 0.138 |
| + Both dieting variables | 6.177 | 8.775 | 0.137 |
| + Alcohol consumption | 6.201 | 8.784 | 0.136 |
| + Weekly working hours | 6.184 | 8.776 | 0.137 |

None of the additional predictors meaningfully improved predictive performance.

PHQ-9, dieting status, and alcohol consumption also had substantial missingness, approximately 29–33% among training participants.

Working hours were reconstructed to distinguish structural questionnaire skips from genuine missing responses.

### Provisional Feature-Selection Decision

Retain the original 16 predictors as the current benchmark.

PHQ-9, dieting behavior, alcohol consumption, and working hours will be provisionally excluded from the primary model because they did not provide meaningful incremental predictive benefit.

These decisions are exploratory and do not imply that the excluded variables are unrelated to weight change.

Because multiple feature comparisons were conducted using the same training folds, further validation may be needed before finalizing the predictor set.

## 8. Data Leakage Considerations

### Baseline BMI and Shared Measurement

Estimated baseline BMI incorporates weight one year earlier, which also appears in the percentage weight-change calculation.

This shared measurement creates potential mathematical coupling and measurement-error concerns.

Baseline BMI will therefore be examined through sensitivity analyses rather than interpreted as a causal predictor.

### Post-Outcome Information

Several predictors were measured around the NHANES interview rather than at the beginning of the weight-change period.

The models are therefore intended for retrospective prediction, not prospective forecasting.

### Preprocessing Leakage

Imputation, scaling, and categorical encoding are fitted within cross-validation training folds.

### Group-Level and Survey-Design Considerations

NHANES uses a clustered, stratified sampling design. Random participant-level splitting may place individuals from the same sampling clusters in both training and validation data.

Survey-design-aware evaluation will be considered before making population-level generalization claims.

### Overfitting the Test Set

The held-out test dataset will not be used for feature selection, model tuning, or repeated evaluation.

## 9. Robustness and Sensitivity Analyses

### 9.1 Baseline BMI Sensitivity

Random Forest models were compared with and without baseline BMI among weight-loss attempters.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| With Baseline BMI | 6.178 | 8.769 | 0.138 |
| Without Baseline BMI | 6.506 | 9.471 | -0.005 |

Removing baseline BMI substantially reduced predictive performance.

This suggests that baseline BMI contributes substantial predictive information beyond the other included characteristics, although its relationship with the outcome may partly reflect shared mathematical structure.

### 9.2 Outlier Sensitivity

Extreme weight change was defined descriptively as an absolute percentage weight change greater than 20%.

Among 2,794 training participants:

- 135 participants (4.83%) reported extreme weight changes.
- These participants accounted for 39.59% of total squared prediction error.

Out-of-fold prediction errors were examined separately for extreme and non-extreme outcomes.

| Group | N | MAE | RMSE | Mean Residual |
|---|---:|---:|---:|---:|
| All Participants | 2,794 | 6.178 | 8.777 | 0.001 |
| Non-Extreme (≤20%) | 2,659 | 5.311 | 6.993 | 0.061 |
| Extreme (>20%) | 135 | 23.257 | 25.124 | -1.194 |

Prediction errors were substantially larger among participants reporting extreme weight changes.

The model tended to underpredict the magnitude of large weight losses and gains.

All participants will be retained in the primary analysis. Restricted-group metrics are reported as diagnostics, not as evidence of improved overall performance.

## 10. Planned Additional Evaluation

The following checks remain planned:

### Subgroup Performance

Compare prediction errors across age, sex, and race/ethnicity groups.

### Additional Robustness Checks

Investigate sensitivity to unusually large predictor values and consider alternative modeling approaches for extreme weight-change outcomes.

### Target-Shuffling Check

Evaluate whether model performance deteriorates when outcome values are randomly shuffled, as expected.

### Cross-Validation Stability

Examine fold-level performance variability and consider repeated or nested cross-validation for more rigorous model selection.

### Survey-Design Evaluation

Investigate appropriate use of NHANES sampling weights, strata, and primary sampling units in model evaluation.

## 11. Documentation and Reproducibility

The project uses Python, Jupyter notebooks, scikit-learn, and GitHub.

The current analysis is maintained in `nhanes_exploration.ipynb`.

### Saved Results

- `results/cv_scores_random80_20_2026-10-15.csv`
- `results/cv_scores_attempters_random80_20_2026-10-15.csv`
- `results/feature_sets_attempters_2026-10-15.csv`
- `results/permutation_importance_attempters_2026-10-15.csv`
- `results/bmi_sensitivity_attempters_2026-10-15.csv`
- `results/outlier_errors_attempters_2026-10-15.csv`

These files contain baseline model comparisons, feature-selection results, permutation importance, and robustness-analysis summaries.

Additional evaluation artifacts will be organized into dedicated notebooks and scripts as the project develops.

## 12. Limitations

- Weight measurements and many predictors are self-reported.
- The data are cross-sectional, limiting prospective prediction and causal interpretation.
- Several predictors were measured around the interview rather than throughout the preceding year.
- Some variables have substantial missingness or questionnaire eligibility restrictions.
- Extreme observations contribute disproportionately to prediction error.
- Baseline BMI and percentage weight change share a previous-weight measurement.
- Preliminary evaluation does not yet account for NHANES survey weights, stratification, or clustering.
- Model performance may vary across demographic groups.
- Feature-selection comparisons were exploratory and may be optimistic because the same training data were used repeatedly.
- The revised target population was selected after preliminary exploration, limiting the independence of subsequent internal validation.

## 13. Next Steps

The next stage will focus on:

1. Reviewing cross-validation stability and subgroup performance.
2. Conducting remaining robustness checks.
3. Evaluating appropriate incorporation of NHANES survey-design information.
4. Refining the final predictor set and model-selection strategy.
5. Organizing evaluation outputs into reproducible repository artifacts.
6. Evaluating the final selected model once on the held-out test dataset.

At this checkpoint, baseline models have been evaluated using multiple metrics and cross-validation. Initial feature-selection and robustness analyses have also been completed. Further modeling and evaluation will continue during subsequent project development.
