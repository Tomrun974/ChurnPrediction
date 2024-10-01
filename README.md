# Customer Churn Prediction using PySpark

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Model Building and Evaluation](#model-building-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Customer churn, the phenomenon of customers discontinuing a service, poses a significant challenge for businesses across various industries. Predicting churn enables companies to proactively engage at-risk customers, implement retention strategies, and ultimately enhance profitability.

This project presents an end-to-end machine learning solution for predicting customer churn using **PySpark**, a powerful tool for large-scale distributed data processing. By leveraging PySpark, the project efficiently handles and analyzes large datasets, making it scalable and suitable for real-world applications.

## Project Overview

The goal of this project is to develop a predictive model that accurately identifies customers at risk of churning. The process involves data cleaning, exploratory data analysis, feature engineering, model training with hyperparameter tuning, and evaluation of multiple machine learning algorithms.

The best-performing model is selected based on key evaluation metrics pertinent to churn prediction, such as recall, precision, F1 score, and AUC. The final model can help businesses focus their retention efforts on the most vulnerable customers.

## Objectives

- **Build a robust machine learning model to predict customer churn.**
- **Perform data cleaning and preprocessing to prepare the dataset for modeling.**
- **Conduct exploratory data analysis to uncover insights and patterns.**
- **Engineer features to improve model performance.**
- **Evaluate and compare multiple machine learning algorithms.**
- **Select the best model based on relevant performance metrics.**

## Dataset Description

The dataset used in this project contains customer information, behavioral data, and churn risk scores. Key features include:

- **Demographics:** Age, gender, region category.
- **Account Information:** Membership category, joining date, customer seniority.
- **Usage Patterns:** Average transaction value, average time spent, average frequency of login days.
- **Engagement Metrics:** Points in wallet, preferred offer types, medium of operation.
- **Churn Risk Score:** The target variable indicating the likelihood of a customer churning.

The dataset is assumed to be large and is processed using PySpark DataFrames.

## Project Structure

- `notebooks/` - Jupyter notebooks containing code and analysis.
- `data/` - Folder for dataset files.
- `models/` - Saved machine learning models.
- `images/` - Visualizations and plots generated during EDA.
- `README.md` - Project documentation.
- `requirements.txt` - Python dependencies.

## Methodology

### Data Preprocessing

- **Handling Missing Values:** Identified and imputed missing values using appropriate strategies (e.g., mean imputation, mode for categorical variables).
- **Data Type Conversion:** Ensured all columns have the correct data types for processing.
- **Outlier Detection:** Detected and treated outliers to reduce their impact on the model.
- **Data Cleaning:** Removed duplicates and corrected inconsistent data entries.

### Exploratory Data Analysis (EDA)

- **Descriptive Statistics:** Computed summary statistics to understand feature distributions.
- **Correlation Analysis:** Examined relationships between features and the target variable.
- **Visualization:** Created histograms, box plots, and heatmaps to visualize data patterns.

### Feature Engineering

- **Date Transformation:** Calculated customer seniority based on the joining date.
- **Categorical Encoding:** Applied one-hot encoding to categorical variables.
- **Scaling Numerical Features:** Used `StandardScaler` and `MinMaxScaler` to normalize numerical features.
- **Feature Selection:** Identified and retained features that contribute most to predicting churn.

### Model Building and Evaluation

Trained and evaluated the following machine learning models using PySpark's MLlib:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient-Boosted Trees**
5. **XGBoost**

#### Hyperparameter Tuning

- Employed cross-validation and grid search to fine-tune hyperparameters for each model.
- Used `ParamGridBuilder` and `CrossValidator` for systematic hyperparameter optimization.

#### Evaluation Metrics

- **Area Under ROC Curve (AUC)**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

These metrics provide a comprehensive assessment of model performance, especially focusing on recall and precision, which are critical in churn prediction.

## Results

The models were evaluated based on the aforementioned metrics, with the following results:

| Model                   |    AUC    | Accuracy | Precision |  Recall  | F1 Score |
|-------------------------|-----------|----------|-----------|----------|----------|
| Logistic Regression     | 0.965112  | 0.896603 | 0.900938  | 0.896603 | 0.896773 |
| Decision Tree           | 0.872440  | 0.923476 | 0.923468  | 0.923476 | 0.923471 |
| Random Forest           | 0.972063  | 0.926340 | 0.926514  | 0.926340 | 0.926239 |
| Gradient-Boosted Trees  | 0.973143  | 0.925522 | 0.925562  | 0.925522 | 0.925536 |
| **XGBoost**             | **0.974938** | **0.928659** | **0.928867**  | **0.928659** | **0.928555** |

**Analysis:**

- **XGBoost** outperformed all other models across all key metrics.
- The high recall of XGBoost indicates it is effective at identifying customers who are likely to churn.
- The precision is also high, meaning fewer false positives (customers predicted to churn who actually don't).
- The F1 score, balancing precision and recall, is the highest for XGBoost, making it the optimal model.

## Conclusion

- The XGBoost model is the best performer for predicting customer churn in this project.
- High recall is particularly valuable, ensuring that most at-risk customers are identified.
- The model can be utilized by businesses to implement targeted retention strategies.
- Future work could involve exploring additional features, advanced hyperparameter tuning, or incorporating more complex models.

## Getting Started

### Prerequisites

- **Python 3.7 or higher**
- **Apache Spark with PySpark**
- **HDFS (Hadoop Distributed File System)** for storing large datasets (optional but recommended)
- **Jupyter Notebook or JupyterLab**
- **Java Development Kit (JDK) 8 or higher**

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/tomrun974/CustomerChurnPrediction.git
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up PySpark:**

   Ensure that PySpark is properly installed and configured. Set the `SPARK_HOME` environment variable if necessary.

### Working with HDFS

If your dataset is stored in HDFS, ensure that your Spark environment is properly connected to your HDFS cluster:

1. **Configure HDFS** by setting up Hadoop and adjusting the `core-site.xml` and `hdfs-site.xml` files as needed.
2. **Upload your dataset to HDFS**:

   ```bash
   hdfs dfs -put /path/to/local/dataset.csv /user/your_hdfs_username/dataset.csv
   ```

3. **Load the Dataset from HDFS in PySpark**:

   ```python
   file_path = "hdfs://localhost:9000/user/your_hdfs_username/dataset.csv"
   df = spark.read.csv(file_path, header=True, inferSchema=True)
   ```

### Running the Project

1. **Start Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

2. **Open the Notebook:**

   Navigate to the `notebooks/` directory and open the main notebook (e.g., `ChurnPrediction.ipynb`).

3. **Run the Cells:**

   Execute the notebook cells sequentially to reproduce the analysis and results.

## Acknowledgments

- **Apache Spark Community:** For providing an excellent platform for big data processing.
- **PySpark Documentation:** For comprehensive guides and references.
- **Machine Learning Communities:** For valuable resources and discussions on churn prediction.
- **Dataset Providers:** Gratitude to the providers of the dataset used in this project.

---

*Note: This project is intended for educational purposes. The dataset used is hypothetical and may not represent real customer data.*

