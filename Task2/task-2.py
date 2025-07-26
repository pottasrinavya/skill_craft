import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load Titanic dataset from seaborn (you can also load from a CSV if you have it)
df = sns.load_dataset('titanic')
# Display the first few rows
print(df.head())
print("\n🔍 Dataset Info:")
print(df.info())
print("\n🧼 Missing Values:")
print(df.isnull().sum())
# Drop columns with too many missing values or not useful
df = df.drop(['deck', 'embark_town', 'alive'], axis=1)
# Fill missing age with median
df['age'] = df['age'].fillna(df['age'].median())
# Fill embarked with the mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
# Drop any remaining rows with missing values
df.dropna(inplace=True)
print("\n✅ After Cleaning - Missing Values:")
print(df.isnull().sum())
print("\n📊 Summary Statistics:")
print(df.describe())
# 1. Gender distribution
sns.countplot(data=df, x='sex')
plt.title('Gender Distribution')
plt.show()
# 2. Survival count by gender
sns.countplot(data=df, x='sex', hue='survived')
plt.title('Survival by Gender')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
# 3. Class distribution
sns.countplot(data=df, x='pclass')
plt.title('Passenger Class Distribution')
plt.show()
# 4. Age distribution
sns.histplot(df['age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()
# 5. Survival rate by class
sns.barplot(data=df, x='pclass', y='survived')
plt.title('Survival Rate by Passenger Class')
plt.show()
# 6. Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Survival by Age Group
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 40, 60, 80], labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])
sns.countplot(data=df, x='age_group', hue='survived')
plt.title('Survival by Age Group')
plt.show()
# Survival by Embarked Location
sns.countplot(data=df, x='embarked', hue='survived')
plt.title('Survival by Embarkation Port')
plt.show()
