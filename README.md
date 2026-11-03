# Bank Marketing Classification

## Problem Statement
Predict whether a bank client will subscribe to a term deposit based on customer, campaign, and economic information.

## Dataset Description
- Source: UCI Machine Learning Repository - Bank Marketing
- Instances: 41,188
- Original input features: 20
- Target: `y` (`no` = 0, `yes` = 1)
- Split: 80% training and 20% testing
- Feature decision: `duration` was excluded because it is known after the marketing call and could cause data leakage.

## Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.8009 | 0.3679 | 0.6466 | 0.4689 | 0.4011 |
| Decision Tree | 0.8265 | 0.7777 | 0.3488 | 0.6228 | 0.4472 | 0.3748 |
| KNN | 0.8949 | 0.7553 | 0.5708 | 0.2694 | 0.3660 | 0.3434 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| SVM | 0.8577 | 0.7829 | 0.4140 | 0.6325 | 0.5004 | 0.4346 |

## Observations
-Logistic Regression: Highest AUC and recall, but lower precision means more false-positive "Yes" predictions.
-Decision Tree: moderate performance across metrics; lower F1 and MCC than SVM.
-KNN: Highest accuracy and precision, but very low recall, so it misses many actual "Yes" cases
-Naive Bayes: Lower F1 and MCC than the stronger models.
-SVM: Best F1 and MCC; strongest overall balance for the minority"Yes" class.

## Overall Winner
SVM was selected as the best overall model because it achieved the highest F1-score and MCC, which are more informative than accuracy for this imbalanced dataset.

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

## Links
GitHub Repository : https://github.com/2026aj05006-hue/bank-marketing-classification
Streamlit App : https://bank-marketing-classification-poy4g4hx8q6vwjyjfogssd.streamlit.app
