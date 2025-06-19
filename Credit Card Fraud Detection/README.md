# Credit Card Fraud Detection

This project detects fraudulent credit card transactions using machine learning models. It includes data preprocessing, visualization, model training, and evaluation.

## Features
- Data normalization and visualization
- Handles class imbalance with SMOTE
- Trains Logistic Regression, Decision Tree, KNN, and SVM models
- Evaluates models with confusion matrix, classification report, ROC, and PR curves

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Usage
1. Download the dataset `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) or another source.
2. Place `creditcard.csv` in the `Credit Card Fraud Detection` directory (not included in this repo due to size).
3. Run the script:
   ```bash
   python fraud_detection.py
   ```

## Note
- Large data files are excluded from the repository. Please download them manually. 