import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, RocCurveDisplay)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("constituents-financials_csv.csv")
print("Initial shape:", df.shape)
print(df.columns)

numeric_cols = ['Price', 'Price/Earnings', 'Dividend Yield', 'Earnings/Share', '52 Week Low', '52 Week High', 'Market Cap', 'EBITDA', 'Price/Sales', 'Price/Book']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Target: P/E
df = df.dropna(subset=['Price/Earnings'])

df = df.dropna()
print("Shape after cleaning:", df.shape)

# Features
df["Price_Range"] = df["52 Week High"] - df["52 Week Low"]
df["Distance_From_High"] = (df["52 Week High"] - df["Price"]) / df["52 Week High"]
df["Distance_From_Low"] = (df["Price"] - df["52 Week Low"]) / df["52 Week Low"]

# High Valuation = P/E >= 75th percentile
pe_75 = df["Price/Earnings"].quantile(0.75)
df["HighValuation"] = (df["Price/Earnings"] >= pe_75).astype(int)

features = ["Price", "Dividend Yield", "Earnings/Share", "EBITDA", "Price/Sales", "Price/Book", "Price_Range", "Distance_From_High", "Distance_From_Low", "Sector"]
X = df[features]
y = df["HighValuation"]

numeric_features = [col for col in features if col != "Sector"]
categorical_features = ["Sector"]

train_test_ratio = 0.80  #80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - train_test_ratio, random_state=42, stratify=y)

preprocess = ColumnTransformer(
    transformers=[("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler())]), 
        numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Decision Tree
tree_clf = Pipeline([
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42))])
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
y_prob_tree = tree_clf.predict_proba(X_test)[:, 1]

print("\n=== DECISION TREE PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall:", recall_score(y_test, y_pred_tree))
print("ROC AUC:", roc_auc_score(y_test, y_prob_tree))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Random Forest
rf_clf = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        random_state=42))])

rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

print("\n=== RANDOM FOREST PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Reds')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
encoder = rf_clf.named_steps["preprocess"].transformers_[1][1]
cat_names = encoder.get_feature_names_out(categorical_features)

all_features = numeric_features + list(cat_names)
importances = rf_clf.named_steps["model"].feature_importances_

feat_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10,7))
sns.barplot(data=feat_df, x="Importance", y="Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# Misclassified
misclassified = X_test.copy()
misclassified['Company'] = df.loc[X_test.index, 'Name']
misclassified['Actual'] = y_test.values
misclassified['Predicted'] = y_pred_rf
wrong = misclassified[misclassified['Actual'] != misclassified['Predicted']]

print("\n=== 5 MISCLASSIFIED SAMPLES ===")
print(wrong.head(5))