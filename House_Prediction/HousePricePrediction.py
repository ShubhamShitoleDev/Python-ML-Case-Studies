#################################################################################
#
#  Project Name  : House Price Prediction System
#  Description   : Loads the California Housing dataset, trains and compares
#                   three regression approaches — a standalone Decision Tree,
#                   a Bagging ensemble of Decision Trees, and Gradient
#                   Boosting — evaluating each using MSE and R2 score.
#  Date          : 23-Aug-2026
#  Author        : Shubham Namdev Shitole
#
#################################################################################

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the California Housing CSV and prints its shape and
#                 first few records
# Return Value :  Loaded DataFrame
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def load_data(data_path):
    df = pd.read_csv(data_path)

    print("Shape of Dataset : ", df.shape)
    print("First few records : ")
    print(df.head())

    return df


#################################################################################
#
# Function Name : prepare_model_data
# Input :         Data
# Description :   Splits Data into features (X) and target (Y), then splits
#                 into training and testing sets
# Return Value :  X_train, X_test, Y_train, Y_test
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def prepare_model_data(Data):
    X = Data.drop("target", axis=1)
    Y = Data["target"]

    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("X_train : ", X_train.shape)
    print("X_test : ", X_test.shape)
    print("Y_train : ", Y_train.shape)
    print("Y_test : ", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


#################################################################################
#
# Function Name : evaluate_model
# Input :         model_name, model, X_train, X_test, Y_train, Y_test
# Description :   Trains the given regressor and prints MSE and R2 score
# Return Value :  Dictionary with the model's MSE and R2 score
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

    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"{model_name} MSE : {mse:.4f}")
    print(f"{model_name} R2  : {r2:.4f}")

    return {"MSE": mse, "R2": r2}


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full House Price Prediction pipeline end-to-end:
#                 load, split, train and compare Decision Tree, Bagging, and
#                 Gradient Boosting regressors
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    df = load_data("california_housing.csv")

    print(Border)
    print("Step 2 : Separate Features and Labels, Split Train/Test")
    print(Border)

    X_train, X_test, Y_train, Y_test = prepare_model_data(df)

    print(Border)
    print("Step 3 : Build, Train and Evaluate Models")
    print(Border)

    results = {}

    # Individual Decision Tree
    results["Decision Tree"] = evaluate_model(
        "Decision Tree", DecisionTreeRegressor(random_state=42),
        X_train, X_test, Y_train, Y_test
    )

    # Bagging - an ensemble of Decision Trees trained on random subsets
    base_model = DecisionTreeRegressor(random_state=42)
    bagging_model = BaggingRegressor(estimator=base_model, n_estimators=10, random_state=42)
    results["Bagging"] = evaluate_model(
        "Bagging", bagging_model,
        X_train, X_test, Y_train, Y_test
    )

    # Boosting - trees built sequentially, each correcting the previous one's errors
    boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    results["Gradient Boosting"] = evaluate_model(
        "Gradient Boosting", boosting_model,
        X_train, X_test, Y_train, Y_test
    )

    print(Border)
    print("Step 4 : Model Comparison")
    print(Border)

    for name, metrics in results.items():
        print(f"{name} : MSE = {metrics['MSE']:.4f} , R2 = {metrics['R2']:.4f}")

    # Higher R2 (closer to 1.0) indicates a better fit
    best_model_name = max(results, key=lambda name: results[name]["R2"])
    print(f"\nBest Model : {best_model_name} with R2 = {results[best_model_name]['R2']:.4f}")


if __name__ == "__main__":
    main()
