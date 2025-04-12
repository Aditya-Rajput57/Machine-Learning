from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ------------------------------
# Part 1: Synthetic Dataset
# ------------------------------

# Generate synthetic classification dataset
X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

# Visualize first 2 features
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, marker="*")
plt.title("Synthetic Data (2 Features)")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predict one sample
predicted = model.predict([X_test[6]])
print("Actual Value:", y_test[6])
print("Predicted Value:", predicted[0])

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Synthetic Data")
plt.show()

# ------------------------------
# Part 2: Used Cars Dataset
# ------------------------------

# CSV path
csv_path = 'C:/Users/Icon/Downloads/ML--LAB-main/ML-5/used_cars_data.csv'

# Check if file exists
if not os.path.exists(csv_path):
    print(f" File not found: {csv_path}")
else:
    # Load dataset
    df = pd.read_csv(csv_path)
    print("\n Dataset Loaded Successfully.")
    print("First 5 rows:")
    print(df.head())

    # Show column names
    print("\nColumns in dataset:", df.columns.tolist())

    # Visualize 'purpose' vs 'not.fully.paid' if the columns exist
    if 'purpose' in df.columns and 'not.fully.paid' in df.columns:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x='purpose', hue='not.fully.paid')
        plt.xticks(rotation=45, ha='right')
        plt.title("Loan Purpose vs Not Fully Paid")
        plt.show()
    else:
        print("Required columns 'purpose' and/or 'not.fully.paid' not found.")

    # Preprocessing: Convert categorical to dummy vars
    if 'purpose' in df.columns:
        pre_df = pd.get_dummies(df, columns=['purpose'], drop_first=True)
    else:
        pre_df = df.copy()

    if 'not.fully.paid' in pre_df.columns:
        X = pre_df.drop('not.fully.paid', axis=1)
        y = pre_df['not.fully.paid']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=125
        )

        # Train model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Predict and Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("\n--- Used Cars Dataset ---")
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix - Used Cars Data")
        plt.show()
    else:
        print("Target column 'not.fully.paid' not found.")