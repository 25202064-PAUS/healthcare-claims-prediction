# PAUS Project Summary
## Predictive Analytics for Healthcare Insurance Claims: Claim Amount Estimation and Fraud Detection

**Purpose:** Project documentation and presentation hand-off. This document consolidates the work completed in SAS Studio, SAS Viya, and the accompanying Streamlit application. It distinguishes reported SAS results from the independently packaged app model.

---

## 1. Business problem and objectives

Healthcare insurers need to assess submitted claims efficiently while identifying claims that may require additional investigation. This project applies predictive analytics to two related tasks:

1. **Regression:** estimate the submitted claim amount (`Claim_Amount`).
2. **Classification:** identify potentially fraudulent claims (`Is_Fraud`).

The intended users are insurance claim analysts and fraud-review teams. Model outputs are decision-support signals; they do not replace claim adjudication or investigation.

## 2. Dataset

- **Dataset:** Healthcare Fraud Detection Dataset, attributed to Nudrat Abbas on Kaggle.
- **Original file:** `healthcare_fraud_detection.csv`.
- **Size:** 10,000 observations and 20 columns.
- **Data nature:** simulated/synthetic healthcare claims data; it is not a real patient or insurer transaction dataset.
- **Date range:** 6 January 2021 to 26 January 2025.
- **Identifiers:** `Claim_ID` is unique; 300 distinct `Provider_ID` values were identified.
- **Fraud target distribution:** 829 fraud records (8.29%) and 9,171 genuine records (91.71%). The target is imbalanced.
- **Claim amount:** mean 572.80; median 461.23; minimum 60.21; maximum 6,590.70; standard deviation 406.20. The distribution is positively skewed.

### Variables in the original file

`Provider_ID`, `Claim_ID`, `Patient_Age`, `Patient_Gender`, `Diagnosis_Code`, `Procedure_Code`, `Claim_Amount`, `Approved_Amount`, `Insurance_Type`, `Claim_Submission_Date`, `Days_Between_Service_and_Claim`, `Number_of_Claims_Per_Provider_Monthly`, `Provider_Specialty`, `Patient_State`, `Claim_Status`, `Is_Fraud`, `Length_of_Stay`, `Visit_Type`, `Chronic_Condition_Flag`, and `Prior_Visits_12m`.

## 3. Data preparation

### Missing values

The original data contained missing values in three fields:

| Variable | Missing observations |
|---|---:|
| `Insurance_Type` | 350 |
| `Provider_Specialty` | 350 |
| `Prior_Visits_12m` | 350 |
| **Total missing cells** | **1,050** |

The SAS Studio preparation used:

- Mode imputation for `Insurance_Type`.
- Mode imputation for `Provider_Specialty`.
- Median imputation for `Prior_Visits_12m`.

The SAS Describe Missing Data output was checked after imputation and showed 10,000 non-missing observations for each of the 20 variables in the imputed table.

### Other preparation and exclusions

- `Claim_ID` and `Provider_ID` were treated as identifiers rather than predictive business attributes.
- `Approved_Amount` was excluded from claim amount estimation because it is strongly correlated with `Claim_Amount` (approximately 0.937) and may not be available at the point when a submitted amount is estimated.
- `Claim_Status` was excluded from fraud prediction because it may reflect a post-adjudication outcome and create target leakage.
- The original date field was not used as a raw predictor in the final classification input list.
- `Procedure_Code` was treated as categorical/nominal for classification even though it is stored numerically.
- For the app model, `Days_Between_Service` is mapped from the original field `Days_Between_Service_and_Claim`; `Number_of_Claims_Per_Provider_Mo` is the shortened name used for `Number_of_Claims_Per_Provider_Monthly` in the SAS modelling export.

**Methodology note:** The SAS imputation was performed before the regression partition in the work completed. This sequencing should be acknowledged as a limitation because preprocessing before partitioning can allow information from the full dataset to influence imputation values.

## 4. Regression analysis — SAS Studio

### Target and modelling approach

- **Target:** `Claim_Amount`.
- **Platform:** SAS Studio.
- **Workflow:** Predictive Regression Models task, with a 70% training / 30% test partition and no separate validation partition.
- **Selection methods run:** Forward Selection, Backward Elimination, Stepwise Regression, LASSO, Adaptive LASSO, Elastic Net, and Least Angle Regression (LAR).
- **Additional diagnostics:** linear regression diagnostics, collinearity assessment, VIF/tolerance, residual plots, Cook's distance, and DFFITS/studentized residual diagnostics.

### Forward selection results captured in the project work

| Step | Variable entered | Adjusted R² | Training ASE | Test ASE |
|---:|---|---:|---:|---:|
| 0 | Intercept only | 0.0000 | 77,894.7992 | 75,154.4979 |
| 1 | Days_Between_Service | 0.0121 | 76,939.0601 | 73,879.8485 |
| 2 | Patient_Gender | 0.0138 | 76,796.6426 | 73,768.1699 |
| 3 | Number_of_Claims_Per | 0.0155 | 76,650.3774 | 73,753.0895 |
| 4 | Insurance_Type | 0.0158 | 76,594.0937 | 73,772.8927 |
| 5 | Provider_Specialty | 0.0161 | 76,515.6033 | 73,697.7453 |
| 6 | Visit_Type | 0.0163 | 76,475.2003 | 73,663.3723 |
| 7 | Patient_State | 0.0164 | 76,384.1038 | 73,681.4803 |

The highest adjusted R² shown in this forward-selection table is **0.0164 (1.64%) at Step 7**. The test ASE is lowest at Step 6 (73,663.3723), so the model choice depends on the selection criterion used. Do not describe 1.64% as 1.94%.

### Diagnostics

- The VIF values reported for the assessed regression predictors were all below 3; no serious multicollinearity was indicated by this check.
- Cook's distance screening used the common reference cutoff `4/n`. For 10,000 observations, this is 0.0004.
- 482 observations were above that screening cutoff and 9,518 were below it in the diagnostic output reviewed. This is a flag for investigation, not automatic proof that those observations should be deleted. The project discussion should state that influential observations were identified and assessed, rather than claiming they were all removed.

### Regression scoring equation supplied by SAS

The supplied SAS-generated GLMSELECT scoring logic contains the following coefficients and indicator effects:

| Term | Coefficient |
|---|---:|
| Intercept | 522.602238691581 |
| Patient_Gender = Female | +20.476514735386 |
| Insurance_Type = Private | -12.9555790325701 |
| Provider_Specialty = Internal Medicine | +13.5467511953848 |
| Visit_Type = Inpatient | -4.01611408761764 |
| Days_Between_Service_and_Claim | -3.55207938214922 |
| Number_of_Claims_Per_Provider_Monthly | +0.39264475395121 |

The app implements this equation using the exact categorical baseline/dummy logic in the supplied scoring code. The full regression model comparison statistics for all seven selection methods were not reliably extractable from the available PDF exports; do not invent a winning method or missing regression metrics. The app labels this component as the SAS-supplied regression scoring equation.

## 5. Classification analysis — SAS Viya

### Target, inputs and partition

- **Target:** `Is_Fraud`, binary, event level 1.
- **Partition:** 70% training and 30% test, stratified by target; the comparison report shows 2,855 test observations.
- **Candidate models:** Full Decision Tree, Pruned Decision Tree, Gradient Boosted Decision Tree, Random Forest (Forest), and Logistic Regression.
- **Model comparison criterion used for champion selection:** KS (Youden).

### Classification inputs used in SAS Viya

1. `Claim_Amount`
2. `Days_Between_Service`
3. `Number_of_Claims_Per_Provider_Mo`
4. `Patient_Age`
5. `Chronic_Condition_Flag`
6. `Patient_Gender`
7. `Diagnosis_Code`
8. `Insurance_Type`
9. `Length_of_Stay`
10. `Patient_State`
11. `Prior_Visits_12m`
12. `Procedure_Code`
13. `Provider_Specialty`
14. `Visit_Type`

`Approved_Amount`, `Claim_Status`, IDs, raw submission date, and diagnostic artifacts were excluded from the classification inputs. `Number_of_Claims_Per_Provider_Mo` remains a classification input even though the variable was not retained in the final regression scoring equation; the two targets have separate models and separate predictor selection.

### SAS Viya Model Comparison results

| Model | Accuracy | AUC-ROC | F1 score | KS (Youden) | Average Squared Error |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 94.89% | 0.9632 | 0.5521 | **0.8388** | 0.0358 |
| Pruned Decision Tree | **95.24%** | 0.9469 | 0.5310 | 0.8260 | 0.0367 |
| Full Decision Tree | 94.71% | 0.9545 | **0.5546** | 0.8172 | 0.0377 |
| Random Forest | 95.17% | **0.9649** | 0.5141 | 0.8233 | **0.0339** |
| Logistic Regression | 92.99% | 0.5000 | 0.0000 | 0.0000 | 0.0651 |

**SAS Viya champion:** Gradient Boosting, because it achieved the highest KS statistic (0.8388). Random Forest had the highest AUC-ROC (0.9649), the Pruned Decision Tree had the highest accuracy (95.24%), and the Full Decision Tree had the highest F1 score (0.5546). Because only 8.29% of the full dataset is labelled fraud, accuracy should not be used alone to assess performance. Precision, recall, F1 and AUC-ROC are important when evaluating the minority fraud class.

### Gradient Boosting configuration from exported SAS training code

The SAS-generated training code records these settings:

- 100 trees (`ntrees=100`)
- Learning rate 0.1
- Sampling rate 0.5
- Maximum depth 4
- Minimum leaf size 5
- 50 bins
- Ridge regularization 1; LASSO 0
- Seed 12345
- Partition variable `_PartInd_`, with training role 1 and test role 2

## 6. Web application

A Streamlit application is included in the deliverable ZIP. It contains four sections:

1. **Claim amount estimate:** a user enters the six inputs required by the SAS-supplied regression equation and receives an estimated amount, contribution chart and calculation breakdown.
2. **Fraud risk screening:** a user enters the 14 classification inputs and receives an estimated fraud probability, a 0.50-threshold classification, and a review recommendation.
3. **Model performance:** displays the SAS Viya comparison table and separately reports the metrics of the locally packaged app model.
4. **Methodology & notes:** explains the dataset, predictors, scoring logic and limitations.

### Important model provenance distinction

The registered SAS Viya model's exported Python Score API was a **batch-scoring API for a CAS table**, not a direct single-record scoring endpoint. The app therefore does not expose SAS URLs, CAS table URIs, output library names or access tokens to end users.

To provide a working individual fraud-screening form with the files available, the ZIP includes a separate local scikit-learn Gradient Boosting pipeline trained on the original 10,000-row dataset. It uses the 14 classification predictors, a stratified 70:30 split, median imputation for numeric inputs, most-frequent imputation and one-hot encoding for categorical inputs. The Gradient Boosting settings are aligned to the exported SAS training settings where corresponding scikit-learn parameters exist (100 estimators, learning rate 0.1, subsample 0.5, maximum depth 4, minimum leaf size 5). This local model is **not the exact registered SAS Viya model binary**; its metrics are shown separately in the app and must not be presented as the SAS Viya model comparison results.

### Local app model test results

The packaged local model was fitted with a stratified 70:30 split (`random_state=12345`) and evaluated on 3,000 held-out observations. Its current test metrics are:

| Metric | Local app model result |
|---|---:|
| Accuracy | 95.07% |
| AUC-ROC | 0.9704 |
| F1 score | 0.6769 |
| Precision | 0.7416 |
| Recall | 62.25% |
| Test observations | 3,000 |

These metrics belong only to the local app model and are not the figures in the SAS Viya Model Comparison report. Differences can arise from the split implementation, preprocessing and software implementation.

## 7. Limitations and points to state in the presentation

- The dataset is simulated; results should not be generalized to real insurer operations without validation on real, representative data.
- Fraud is a minority class (8.29%); false negatives and false positives have different business costs. A production threshold should be selected with business stakeholders.
- `Approved_Amount` and `Claim_Status` were excluded to reduce target leakage and reflect information available at claim submission.
- The regression forward-selection adjusted R² is low (1.64%), indicating that the selected predictors explain only a small proportion of variation in claim amount. This should be discussed candidly.
- Imputation before partitioning can introduce information leakage; a production workflow should estimate imputation values on training data only and apply them to test/new data.
- The individual fraud-screening feature in the app uses a locally refitted model, not the registered SAS Viya model. The SAS Viya comparison results and local app model results are kept separate.
- Model outputs are for prioritizing review, not automatic rejection, payment or fraud determination.

## 8. Suggested presentation storyline (8–10 minutes, under 15 slides)

1. Title and team/project objective.
2. Business problem: claim assessment and fraud risk.
3. Dataset profile and fraud class imbalance.
4. Data preparation, missing values and leakage controls.
5. End-to-end workflow and tool split (SAS Studio regression; SAS Viya classification; Streamlit app).
6. Regression target, model selection methods and partitioning.
7. Regression selection results and diagnostics (include forward-selection table highlights and limitations).
8. Regression scoring equation and interpretation of selected coefficients.
9. Classification target, predictors and five candidate algorithms.
10. SAS Viya model comparison table and Gradient Boosting champion decision.
11. App architecture and model-provenance note (SAS-supplied regression equation + locally packaged fraud classifier).
12. Live app walkthrough: amount estimate and fraud risk screening.
13. Business implications, limitations and conclusion.

**Presentation accuracy rule:** Do not say the app calls the registered SAS Viya model for each individual claim. The current packaged app uses a local model for individual fraud scoring. Also do not describe the local app metrics as SAS Viya metrics.
