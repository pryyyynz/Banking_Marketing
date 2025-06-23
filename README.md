## Project Overview
This project analyzes the Bank Marketing dataset to predict whether a customer will subscribe to a term deposit based on demographic, financial, and campaign-related features.

## Analysis & Methods
- **Exploratory Data Analysis (EDA):** Explored feature distributions, class imbalance, and relationships between variables using visualizations and summary statistics.
- **Feature Engineering:** Applied label, ordinal, one-hot, and cyclical encoding. Created new features such as campaign efficiency ratio, economic sentiment, and contact history.
- **Preprocessing:** Encoded and normalized all features, checked for missing/infinite values, and ensured reproducibility.
- **Class Imbalance Handling:** Compared SMOTE, under-sampling, and SMOTEENN techniques to address class imbalance.
- **Modeling:** Trained and evaluated multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes, MLP) using accuracy, precision, recall, F1, and ROC AUC metrics. Selected the best model based on F1 score.
- **Interpretability:** Analyzed feature importance for the best model to identify key drivers of subscription.

## Insights & Results
- The dataset is highly imbalanced, with a small proportion of customers subscribing.
- Economic indicators, contact history, and campaign features are strong predictors of subscription.
- Feature engineering and proper encoding significantly improved model performance.
- The best model (Random Forest) achieved strong performance:
  - **Accuracy:** ~91%
  - **Precision:** ~65%
  - **Recall:** ~60%
  - **F1 Score:** ~62%
  - **ROC AUC:** ~0.90
- Class imbalance handling (SMOTEENN) further improved recall and F1 score.

## Conclusion
The project demonstrates a robust workflow for data preprocessing, feature engineering, model selection, and evaluation. The final model provides actionable insights for marketing strategies and can be deployed for real-time predictions.