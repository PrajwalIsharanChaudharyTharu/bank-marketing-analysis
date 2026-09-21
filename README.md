# Bank Marketing Data Analysis, Machine Learning and PySpark

This project analyses the UCI Bank Marketing dataset to investigate customer characteristics and predict whether a customer will subscribe to a term deposit.

## Dataset

The dataset contains 45,211 customer records and 17 attributes. It includes demographic, financial, contact and campaign-related information.

Source: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## Research Problem

The main objective is to predict whether a bank customer will subscribe to a term deposit using machine learning techniques.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PySpark
- Google Colab

## Machine Learning Models

Four machine learning algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The models were evaluated using Accuracy, Precision, Recall, F1-Score and ROC-AUC.

## Big Data Analysis

Apache Spark (PySpark) was used to demonstrate distributed data processing and aggregation techniques on the Bank Marketing dataset.

## Project Structure

```text
bank-marketing-analysis/
├── data/
│   └── bank-full.csv
├── notebooks/
│   └── bank_marketing_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── visualisation.py
│   └── models.py
├── results/
│   ├── figures/
│   └── model_results.csv
├── README.md
└── requirements.txt


## Author

Prajwal Isharan Chaudhary Tharu

MSc in Advanced Computer Science
