# Banking-Churn-Prediction-Model

## Dataset

The dataset contains synthetic customer data for a bank, with 10,000 records and 14 features:

* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary
* Exited (target)

## Exploratory Data Analysis (EDA)

* Checked the target distribution (`Exited`): 20% churners, 80% non-churners.
* Dropped unnecessary columns: `RowNumber`, `CustomerId`, `Surname`.
* Analyzed binary features vs churn:

  * `IsActiveMember`: inactive members are more likely to churn (~27% vs ~14%).
  * `HasCrCard`: no significant effect.
* Analyzed categorical features vs churn:

  * `Geography`: German customers churn more (~32%) than French or Spanish (~16%).
  * `Gender`: females are slightly more likely to churn (~25% vs ~16% for males).
* Explored numerical features vs churn, checking distributions and correlations with the target.
* Confirmed class imbalance and overall data readiness for modeling.

## Modeling

### Logistic Regression (LR)

* Split data into train (80%) and test (20%), stratified by target.
* Applied `class_weight='balanced'` to give churners more importance.
* Tuned threshold to 0.45 to achieve ~77% recall for churners.
* Evaluation metrics:

  * Precision: 0.36, Recall: 0.77, F1-score: 0.49
  * ROC-AUC: 0.777
* Observations: LR caught most churners but had low precision (many false positives).

### Random Forest (RF)

* Encoded categorical features using one-hot encoding.
* Re-split data after encoding.
* Trained `RandomForestClassifier` with `class_weight='balanced'`.
* Tuned threshold to 0.21 to achieve ~77% recall.
* Evaluation metrics:

  * Precision: 0.48, Recall: 0.77, F1-score: 0.60
  * ROC-AUC: 0.855
* Observations: RF improved precision while maintaining recall, providing a more reliable model for identifying potential churners.

## Conclusion

* Logistic Regression is simple and catches most churners, but produces many false positives.
* Random Forest is more robust, improving precision without sacrificing recall.
* Threshold tuning is essential for balancing recall and precision according to business needs.
