#################################################################################
#
#  Project Name  : Diabetes Detection System
#  Description   : Loads the Pima Indians Diabetes dataset, performs
#                   exploratory data analysis, cleans biologically
#                   impossible zero values (treated as missing data),
#                   visualizes feature relationships, then trains and
#                   compares Logistic Regression, KNN, Decision Tree, and
#                   Random Forest classifiers to predict diabetes Outcome.
#  Date          : 23-Aug-2026
#  Author        : Shubham Namdev Shitole
#
#################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Border = "-"*50

# NOTE : In this dataset, a value of 0 in these columns is not a real
# measurement — it's biologically impossible to have 0 Glucose, 0 Blood
# Pressure, 0 Skin Thickness, 0 Insulin, or 0 BMI in a living patient.
# These zeros are actually missing data recorded as 0 rather than NaN.
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the Diabetes CSV, prints an EDA summary (head, info,
#                 stats, missing value counts), and reports how many
#                 biologically impossible zero values exist per column
# Return Value :  Loaded DataFrame
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def load_data(data_path):
    df = pd.read_csv(data_path)

    print("Shape of dataset : ", df.shape)
    print("First few records : ")
    print(df.head())

    print(Border)
    print("Column Info : ")
    df.info()

    print(Border)
    print("Statistical Summary : ")
    print(df.describe())

    print(Border)
    print("Missing Values (NaN) Per Column : ")
    print(df.isnull().sum())

    print(Border)
    print("Zero Values Per Column (some of these are actually missing data) : ")
    for col in ZERO_AS_MISSING_COLS:
        zero_count = (df[col] == 0).sum()
        print(f"{col} : {zero_count} zero values ({zero_count / len(df) * 100:.2f} %)")

    print(Border)
    print("Class distribution (Outcome) : ")
    print(df["Outcome"].value_counts())

    return df


#################################################################################
#
# Function Name : handle_missing_data
# Input :         Data
# Description :   Replaces biologically impossible zero values in Glucose,
#                 BloodPressure, SkinThickness, Insulin, and BMI with the
#                 median value for that column, grouped by Outcome, since
#                 diabetic and non-diabetic patients tend to have different
#                 typical ranges for these measurements
# Return Value :  Cleaned DataFrame
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def handle_missing_data(Data):
    cleaned_data = Data.copy()

    print(Border)
    print("Cleaning zero-as-missing values...")
    print(Border)

    for col in ZERO_AS_MISSING_COLS:
        # Treat 0 as missing by converting it to NaN first
        cleaned_data[col] = cleaned_data[col].replace(0, pd.NA)
        cleaned_data[col] = pd.to_numeric(cleaned_data[col])

        missing_before = cleaned_data[col].isnull().sum()

        # Fill with the median for that column, computed separately for
        # diabetic (Outcome=1) and non-diabetic (Outcome=0) patients
        median_by_outcome = cleaned_data.groupby("Outcome")[col].transform("median")
        cleaned_data[col] = cleaned_data[col].fillna(median_by_outcome)

        print(f"{col} : filled {missing_before} missing values (median per Outcome group)")

    return cleaned_data


#################################################################################
#
# Function Name : DisplayPlots
# Input :         Data
# Description :   Visualizes Outcome distribution, feature histograms,
#                 Glucose levels by Outcome, and feature correlation matrix
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def DisplayPlots(Data):
    sns.countplot(x="Outcome", data=Data)
    plt.title("Distribution of Diabetes Outcome (0 = No, 1 = Yes)")
    plt.show()

    Data.hist(figsize=(15, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.show()

    sns.boxplot(x="Outcome", y="Glucose", data=Data)
    plt.title("Glucose Level by Diabetes Outcome")
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(Data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()


#################################################################################
#
# Function Name : prepare_model_data
# Input :         Data
# Description :   Splits Data into features (X) and label (Y), splits into
#                 train/test sets, then scales features using StandardScaler.
#                 The scaler is fit ONLY on the training set and applied to
#                 the test set, to avoid data leakage.
# Return Value :  X_train, X_test, Y_train, Y_test (all scaled)
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def prepare_model_data(Data):
    X = Data.drop("Outcome", axis=1)
    Y = Data["Outcome"]

    print("X Shape : ", X.shape)
    print("Y Shape : ", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    # Fit the scaler ONLY on the training set...
    X_train = scaler.fit_transform(X_train)
    # ...then reuse those same fitted parameters on the test set.
    X_test = scaler.transform(X_test)

    print("X_train : ", X_train.shape)
    print("X_test : ", X_test.shape)
    print("Y_train : ", Y_train.shape)
    print("Y_test : ", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


#################################################################################
#
# Function Name : evaluate_model
# Input :         model_name, model, X_train, X_test, Y_train, Y_test
# Description :   Trains the given classifier and prints accuracy, confusion
#                 matrix, and classification report
# Return Value :  Accuracy score of the model on the test set
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def evaluate_model(model_name, model, X_train, X_test, Y_train, Y_test):
    print(Border)
    print(f"Training : {model_name}")
    print(Border)

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"{model_name} Accuracy : {accuracy * 100:.2f} %")

    print("Confusion Matrix : ")
    print(confusion_matrix(Y_test, Y_pred))

    print("Classification Report : ")
    print(classification_report(Y_test, Y_pred))

    return accuracy


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full Diabetes Detection pipeline end-to-end:
#                 load, clean, visualize, split/scale, train and compare
#                 Logistic Regression, KNN, Decision Tree, and Random Forest
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    df = load_data("diabetes.csv")

    print(Border)
    print("Step 2 : Handle Missing (Zero) Values")
    print(Border)

    clean_df = handle_missing_data(df)

    print(Border)
    print("Step 3 : Visualization of DataSet")
    print(Border)

    DisplayPlots(clean_df)

    print(Border)
    print("Step 4 : Prepare Train/Test Data")
    print(Border)

    X_train, X_test, Y_train, Y_test = prepare_model_data(clean_df)

    print(Border)
    print("Step 5 : Build, Train and Evaluate Models")
    print(Border)

    results = {}

    results["Logistic Regression"] = evaluate_model(
        "Logistic Regression", LogisticRegression(max_iter=1000),
        X_train, X_test, Y_train, Y_test
    )

    results["KNN"] = evaluate_model(
        "K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5),
        X_train, X_test, Y_train, Y_test
    )

    results["Decision Tree"] = evaluate_model(
        "Decision Tree", DecisionTreeClassifier(random_state=42),
        X_train, X_test, Y_train, Y_test
    )

    results["Random Forest"] = evaluate_model(
        "Random Forest", RandomForestClassifier(n_estimators=100, random_state=42),
        X_train, X_test, Y_train, Y_test
    )

    print(Border)
    print("Step 6 : Model Comparison")
    print(Border)

    for name, acc in results.items():
        print(f"{name} : {acc * 100:.2f} %")

    best_model_name = max(results, key=results.get)
    print(f"\nBest Model : {best_model_name} with {results[best_model_name] * 100:.2f} % accuracy")


if __name__ == "__main__":
    main()
