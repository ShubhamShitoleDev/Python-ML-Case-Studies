#################################################################################
#
#  Project Name  : Ad Click Prediction System
#  Description   : Loads the online advertising dataset, performs
#                   exploratory data analysis on user behaviour signals
#                   (time on site, internet usage, area income, age),
#                   engineers an Hour feature from Timestamp, drops
#                   high-cardinality text columns, then trains and compares
#                   Logistic Regression, KNN, Decision Tree, and Random
#                   Forest classifiers to predict whether a user clicks on
#                   an ad.
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


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the Advertising CSV and prints an EDA summary
#                 (head, info, stats, missing value counts, class balance)
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
    print("Missing Values Per Column : ")
    print(df.isnull().sum())

    print(Border)
    print("Class distribution (Clicked on Ad) : ")
    print(df["Clicked on Ad"].value_counts())

    return df


#################################################################################
#
# Function Name : engineer_features
# Input :         Data
# Description :   Extracts Hour from Timestamp (click behaviour can vary by
#                 time of day), then drops high-cardinality / free-text
#                 columns (Ad Topic Line, City, Country, Timestamp) that add
#                 noise rather than signal for a tabular classifier
# Return Value :  DataFrame with engineered features, ready for modeling
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def engineer_features(Data):
    cleaned_data = Data.copy()

    print(Border)
    print("Extracting 'Hour' from Timestamp...")
    cleaned_data["Timestamp"] = pd.to_datetime(cleaned_data["Timestamp"])
    cleaned_data["Hour"] = cleaned_data["Timestamp"].dt.hour
    print("Hour value counts : ")
    print(cleaned_data["Hour"].value_counts().sort_index())

    print(Border)
    print("Dropping high-cardinality / free-text columns : Ad Topic Line, City, Country, Timestamp")
    cleaned_data = cleaned_data.drop(columns=["Ad Topic Line", "City", "Country", "Timestamp"])
    print(f"Remaining columns : {list(cleaned_data.columns)}")

    return cleaned_data


#################################################################################
#
# Function Name : DisplayPlots
# Input :         Data
# Description :   Visualizes click distribution and how user behaviour
#                 signals (time on site, internet usage, income, age)
#                 relate to whether the user clicked the ad
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def DisplayPlots(Data):
    sns.countplot(x="Clicked on Ad", data=Data)
    plt.title("Distribution of Ad Clicks (0 = No, 1 = Yes)")
    plt.show()

    Data.hist(figsize=(15, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.show()

    sns.boxplot(x="Clicked on Ad", y="Daily Time Spent on Site", data=Data)
    plt.title("Time Spent on Site by Click Outcome")
    plt.show()

    sns.boxplot(x="Clicked on Ad", y="Daily Internet Usage", data=Data)
    plt.title("Daily Internet Usage by Click Outcome")
    plt.show()

    sns.boxplot(x="Clicked on Ad", y="Area Income", data=Data)
    plt.title("Area Income by Click Outcome")
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
    X = Data.drop("Clicked on Ad", axis=1)
    Y = Data["Clicked on Ad"]

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
# Description :   Runs the full Ad Click Prediction pipeline end-to-end:
#                 load, engineer features, visualize, split/scale, train
#                 and compare Logistic Regression, KNN, Decision Tree, and
#                 Random Forest
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    df = load_data("advertising.csv")

    print(Border)
    print("Step 2 : Feature Engineering")
    print(Border)

    engineered_df = engineer_features(df)

    print(Border)
    print("Step 3 : Visualization of DataSet")
    print(Border)

    DisplayPlots(engineered_df)

    print(Border)
    print("Step 4 : Prepare Train/Test Data")
    print(Border)

    X_train, X_test, Y_train, Y_test = prepare_model_data(engineered_df)

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
