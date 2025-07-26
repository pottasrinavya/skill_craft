import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Correctly load the dataset with ';' as the separator
df = pd.read_csv("bank-additional.csv", sep=';')

# -----------------------------------
# Step 1: Data Preprocessing
# -----------------------------------
print("📋 Dataset Info:")
print(df.info())
print("\n🔍 First few rows:")
print(df.head())

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('y', axis=1)
y = df['y']

# -----------------------------------
# Step 2: Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# Step 3: Train Decision Tree Model
# -----------------------------------
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------
# Step 4: Evaluate Model
# -----------------------------------
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------------
# Step 5: Visualize Decision Tree
# -----------------------------------
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
