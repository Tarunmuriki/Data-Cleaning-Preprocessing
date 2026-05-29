# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# STEP 1 - LOAD DATASET
print("Loading Dataset...\n")
df = pd.read_csv("dataset/titanic.csv")
print("First 5 Rows:")
print(df.head())

# STEP 2 - BASIC INFORMATION
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# STEP 3 - HANDLE MISSING VALUES
# Fill Age missing values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked missing values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because too many missing values
df.drop(columns=['Cabin'], inplace=True)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# STEP 4 - ENCODE CATEGORICAL VARIABLES
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
print("\nEncoded Data:")
print(df[['Sex', 'Embarked']].head())

# STEP 5 - FEATURE SCALING
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
print("\nScaled Features:")
print(df[['Age', 'Fare']].head())

# STEP 6 - OUTLIER DETECTION
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot for Fare")
plt.savefig("screenshots/outlier_boxplot.png")
plt.show()

# STEP 7 - REMOVE OUTLIERS USING IQR
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
print("\nDataset Shape After Removing Outliers:")
print(df.shape)

# STEP 8 - SAVE CLEANED DATASET
df.to_csv("dataset/cleaned_titanic.csv", index=False)
print("\nCleaned dataset saved successfully!")

# STEP 9 - CORRELATION HEATMAP
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("screenshots/correlation_heatmap.png")
plt.show()
print("\nTask Completed Successfully!")