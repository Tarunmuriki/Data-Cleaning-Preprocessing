# Task 1 - Data Cleaning & Preprocessing

## AI & ML Internship Task

### Objective

The objective of this task is to learn how to clean and preprocess raw data before using it in Machine Learning models. Data preprocessing is one of the most important steps in Machine Learning because real-world datasets usually contain missing values, categorical data, inconsistent formats, and outliers.

In this project, the Titanic Dataset was used to perform different preprocessing techniques such as handling missing values, encoding categorical features, feature scaling, and outlier detection/removal.

# Tools & Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* VS Code

# Dataset Used

Dataset Name: Titanic Dataset

The dataset contains passenger information such as:

* Passenger ID
* Name
* Gender
* Age
* Ticket Fare
* Passenger Class
* Survival Status

# Project Structure

```text
Task1_Data_Preprocessing
│
├── dataset
│   ├── titanic.csv
│   └── cleaned_titanic.csv
│
├── screenshots
│   ├── outlier_boxplot.png
│   └── correlation_heatmap.png
│
├── data Cleaning.py
├── README.md
└── requirements.txt
```

# Installation Steps

## Step 1: Install Python

Download and install Python from:
https://www.python.org/downloads/

## Step 2: Install Required Libraries

Open terminal and run:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn
```

# Steps Performed in the Project

## 1. Importing Libraries

The following libraries were imported:

* Pandas for data handling
* NumPy for numerical operations
* Matplotlib and Seaborn for visualization
* Scikit-learn for preprocessing techniques

## 2. Loading the Dataset

The Titanic dataset was loaded using Pandas:

```python
df = pd.read_csv("dataset/titanic.csv")
```

The first few rows of the dataset were displayed using:

```python
df.head()
```

## 3. Exploring Dataset Information

Basic dataset information was checked using:

* `df.info()`
* `df.isnull().sum()`

This helped identify:

* Missing values
* Data types
* Total number of rows and columns

# Missing Value Handling

The dataset contained missing values in:

* Age column
* Embarked column
* Cabin column

## Techniques Used

### Age Column

Missing values were replaced using the median value:

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
```

### Embarked Column

Missing values were replaced using mode:

```python
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

### Cabin Column

The Cabin column had too many missing values, so it was removed:

```python
df.drop(columns=['Cabin'], inplace=True)
```

# Encoding Categorical Variables

Machine Learning models cannot directly understand text data. Therefore, categorical values were converted into numerical format using Label Encoding.

## Encoded Columns

* Sex
* Embarked

Example:

* Male → 1
* Female → 0

Code used:

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
```

# Feature Scaling

Feature scaling helps improve Machine Learning model performance by bringing all numerical features to a similar scale.

Standardization was used in this project.

## Scaled Columns

* Age
* Fare

Code used:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```

# Outlier Detection

Outliers are extreme values that are significantly different from normal observations.

A boxplot was used to visualize outliers in the Fare column.

Code used:

```python
sns.boxplot(x=df['Fare'])
```

# Outlier Removal Using IQR Method

The Interquartile Range (IQR) method was used to remove outliers.

Formula:

```text
IQR = Q3 - Q1
```

Outliers were removed using lower and upper bounds.

Code used:

```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
```

# Data Visualization

Two visualizations were created:

1. Boxplot for outlier detection
2. Correlation Heatmap

## Correlation Heatmap

The heatmap shows relationships between numerical columns.

Code used:

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
```

# Output Files Generated

The following files were generated:

* cleaned_titanic.csv
* outlier_boxplot.png
* correlation_heatmap.png

# Interview Questions & Answers

## 1. What are missing values?

Missing values are empty or null entries in a dataset.

## 2. Why is preprocessing important?

Preprocessing improves data quality and helps Machine Learning models perform better.

## 3. What is Label Encoding?

Label Encoding converts categorical text values into numerical values.

## 4. Difference between normalization and standardization?

### Normalization

Scales values between 0 and 1.

### Standardization

Transforms data so that:

* Mean = 0
* Standard Deviation = 1

## 5. What are outliers?

Outliers are abnormal or extreme values in a dataset.

## 6. How are outliers detected?

Outliers can be detected using:

* Boxplots
* IQR Method
* Z-score Method

# Conclusion

In this project, the Titanic dataset was successfully cleaned and preprocessed using Python. Missing values were handled, categorical variables were encoded, numerical features were scaled, and outliers were removed. Proper preprocessing improves data quality and increases the efficiency and accuracy of Machine Learning models.

This task helped in understanding the importance of preprocessing techniques in real-world Machine Learning workflows.
