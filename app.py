
# ============================================================
# CASE STUDY: CART ABANDONMENT PREDICTION USING RANDOM FOREST
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# ------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ------------------------------------------------------------

# Download your Google Sheets dataset as a CSV file.
# In Google Colab, upload the CSV file first.

df = pd.read_csv("cart_abandonment.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 2: PREPARE THE TARGET VARIABLE
# ------------------------------------------------------------

# Assumed target:
# PurchaseCompleted = 1 means purchase completed
# PurchaseCompleted = 0 means purchase not completed
#
# Change this column name to match your actual dataset.

target = "PurchaseCompleted"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Remove rows where the target is missing
df = df.dropna(subset=[target]).copy()

# Convert target labels to 0 and 1 if necessary
if not set(df[target].unique()).issubset({0, 1, False, True}):
    mapping = {
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "completed": 1, "abandoned": 0,
        "purchase": 1, "no purchase": 0
    }

    df[target] = (
        df[target].astype(str).str.strip().str.lower().map(mapping)
    )

    df = df.dropna(subset=[target]).copy()

df[target] = df[target].astype(int)

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Remove identifiers that should not be used as predictive features
id_columns = [
    col for col in X.columns
    if col.lower() in ["customerid", "sessionid", "userid"]
]

X = X.drop(columns=id_columns, errors="ignore")

print("\nTarget distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe(include="all"))

# Target distribution
sns.countplot(x=y)
plt.title("Purchase Completion Distribution")
plt.xlabel("Purchase Completed (0 = No, 1 = Yes)")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.show()

# Correlation heatmap for numeric columns
numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] > 1:
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# STEP 4: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("Categorical features:", categorical_features)


# ------------------------------------------------------------
# STEP 5: PREPROCESS THE DATA
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ------------------------------------------------------------
# STEP 6: SPLIT THE DATASET
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# STEP 7: BUILD THE RANDOM FOREST MODEL
# ------------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])


# ------------------------------------------------------------
# STEP 8: TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# ------------------------------------------------------------
# STEP 9: MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# STEP 10: EVALUATE MODEL PERFORMANCE
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

if len(np.unique(y_test)) == 2:
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Purchase", "Purchase"]
)

disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 11: FEATURE IMPORTANCE
# ------------------------------------------------------------

classifier = model.named_steps["classifier"]
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)
plt.title("Top 10 Features Influencing Purchase Prediction")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 12: PREDICT FOR A NEW CUSTOMER
# ------------------------------------------------------------

# Example: use the first test sample as an example customer.
# Replace this with actual customer/session information.

sample_customer = X_test.iloc[[0]]

purchase_probability = model.predict_proba(
    sample_customer
)[0, 1]

predicted_class = model.predict(sample_customer)[0]

print("\n========== CUSTOMER PREDICTION ==========")
print(f"Purchase probability: {purchase_probability:.2%}")

if predicted_class == 1:
    print("Prediction: Customer is likely to complete the purchase.")
else:
    print("Prediction: Customer is likely to abandon the cart.")

# Potential campaign audience:
# A low purchase probability can indicate a potential abandonment risk.
# The threshold should be selected using validation data and campaign costs.

abandonment_risk = 1 - purchase_probability

print(f"Estimated abandonment risk: {abandonment_risk:.2%}")

if abandonment_risk >= 0.70:
    print("Campaign suggestion: Consider a targeted reminder or discount.")
else:
    print("Campaign suggestion: A discount may not be necessary.")


# ------------------------------------------------------------
# STEP 13: SAVE THE TRAINED MODEL
# ------------------------------------------------------------

import joblib

joblib.dump(model, "cart_abandonment_model.pkl")

print("\nModel saved as cart_abandonment_model.pkl")

