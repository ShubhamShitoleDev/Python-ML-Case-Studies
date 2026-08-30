#################################################################################
#
#  Project Name  : Marvellous Breast Cancer Detection System
#  Description   : Loads the Breast Cancer dataset, scales features, then
#                   trains and compares seven classification approaches —
#                   Decision Tree, Logistic Regression, Random Forest,
#                   Bagging, AdaBoost, and Voting (hard & soft) — evaluating
#                   each using accuracy, confusion matrix, and classification
#                   report.
#  Date          : 23-Aug-2026
#  Author        : Shubham Shitole
#
#################################################################################

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the Breast Cancer CSV and prints its shape and
#                 first few records
# Return Value :  Loaded DataFrame
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_data(data_path):
    df = pd.read_csv(data_path)

    print("Shape of dataset : ", df.shape)
    print("First few records : ")
    print(df.head())

    return df


#################################################################################
#
# Function Name : prepare_model_data
# Input :         Data
# Description :   Splits Data into features (X) and target (Y), splits into
#                 train/test sets, then scales features using StandardScaler.
#                 The scaler is fit ONLY on the training set and applied to
#                 the test set, to avoid data leakage.
#
#                 NOTE : The original 7 scripts each called
#                 scaler.fit_transform() on BOTH X_train and X_test. That
#                 re-fits the scaler from scratch on the test set, so
#                 training data and test data end up scaled using two
#                 completely different mean/std values. The model trains on
#                 one scale and gets evaluated on a different scale, which
#                 silently distorts the accuracy reported. Fixed here by
#                 fitting the scaler once on X_train and reusing those same
#                 parameters (via .transform, not .fit_transform) on X_test.
# Return Value :  X_train, X_test, Y_train, Y_test (all scaled)
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def prepare_model_data(Data):
    X = Data.drop("target", axis=1)
    Y = Data["target"]

    print("X shape : ", X.shape)
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
# Author :        Shubham Shitole
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
# Description :   Runs the full Breast Cancer Detection pipeline end-to-end:
#                 load, split/scale, train and compare 7 classifiers
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    df = load_data("breast_cancer.csv")

    print(Border)
    print("Step 2 : Separate Features/Labels, Split, and Scale")
    print(Border)

    X_train, X_test, Y_train, Y_test = prepare_model_data(df)

    print(Border)
    print("Step 3 : Build, Train and Evaluate Models")
    print(Border)

    results = {}
    
    results["Decision Tree"] = evaluate_model(
        "Decision Tree", DecisionTreeClassifier(random_state=42),
        X_train, X_test, Y_train, Y_test
    )

    results["Logistic Regression"] = evaluate_model(
        "Logistic Regression", LogisticRegression(max_iter=1000),
        X_train, X_test, Y_train, Y_test
    )

    results["Random Forest"] = evaluate_model(
        "Random Forest", RandomForestClassifier(n_estimators=10, random_state=42),
        X_train, X_test, Y_train, Y_test
    )

    base_model = DecisionTreeClassifier(random_state=42)
    bagging_model = BaggingClassifier(estimator=base_model, n_estimators=10, random_state=42)
    results["Bagging"] = evaluate_model(
        "Bagging", bagging_model,
        X_train, X_test, Y_train, Y_test
    )

    results["AdaBoost"] = evaluate_model(
        "AdaBoost", AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
        X_train, X_test, Y_train, Y_test
    )

    model_log = LogisticRegression(max_iter=1000)
    model_dt = DecisionTreeClassifier(random_state=42)
    model_knn = KNeighborsClassifier(n_neighbors=5)

    voting_hard = VotingClassifier(
        estimators=[("logistic", model_log), ("decision_tree", model_dt), ("knn", model_knn)],
        voting="hard"
    )
    results["Voting (Hard)"] = evaluate_model(
        "Voting (Hard)", voting_hard,
        X_train, X_test, Y_train, Y_test
    )

    voting_soft = VotingClassifier(
        estimators=[("logistic", model_log), ("decision_tree", model_dt), ("knn", model_knn)],
        voting="soft"
    )
    results["Voting (Soft)"] = evaluate_model(
        "Voting (Soft)", voting_soft,
        X_train, X_test, Y_train, Y_test
    )

    print(Border)
    print("Step 4 : Model Comparison")
    print(Border)

    for name, acc in results.items():
        print(f"{name} : {acc * 100:.2f} %")

    best_model_name = max(results, key=results.get)
    print(f"\nBest Model : {best_model_name} with {results[best_model_name] * 100:.2f} % accuracy")


if __name__ == "__main__":
    main()
