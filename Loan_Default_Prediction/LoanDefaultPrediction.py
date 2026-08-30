#################################################################################
#
#  Project Name  : Loan Default Prediction System
#  Description   : Loads the Loan Default dataset, performs exploratory
#                   data analysis (including a class balance check, since
#                   loan defaults are typically a minority class), encodes
#                   categorical features, then trains and compares Random
#                   Forest and Gradient Boosting classifiers to predict
#                   whether a borrower will default.
#  Date          : 23-Aug-2026
#  Author        : Shubham Namdev Shitole
#
#################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the Loan Default CSV and prints an EDA summary
#                 (head, info, stats, missing values, and class balance)
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
    print("Class Distribution (Default) : ")
    print(df["Default"].value_counts())
    print("Default Rate : %.2f %%" % (df["Default"].mean() * 100))

    return df


#################################################################################
#
# Function Name : DisplayPlots
# Input :         Data
# Description :   Visualizes default distribution and how key financial
#                 signals (Credit Score, Income, DTI Ratio, Interest Rate)
#                 relate to default outcome
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def DisplayPlots(Data):
    sns.countplot(x="Default", data=Data)
    plt.title("Distribution of Loan Default (0 = Repaid, 1 = Defaulted)")
    plt.show()

    sns.boxplot(x="Default", y="CreditScore", data=Data)
    plt.title("Credit Score by Default Outcome")
    plt.show()

    sns.boxplot(x="Default", y="Income", data=Data)
    plt.title("Income by Default Outcome")
    plt.show()

    sns.boxplot(x="Default", y="DTIRatio", data=Data)
    plt.title("Debt-to-Income Ratio by Default Outcome")
    plt.show()

    sns.boxplot(x="Default", y="InterestRate", data=Data)
    plt.title("Interest Rate by Default Outcome")
    plt.show()


#################################################################################
#
# Function Name : prepare_model_data
# Input :         Data
# Description :   Drops the LoanID identifier, maps Yes/No columns to
#                 binary, one-hot encodes remaining categorical columns,
#                 splits into features (X) and label (Y), splits into
#                 train/test sets, then scales features. The scaler is fit
#                 ONLY on the training set and applied to the test set, to
#                 avoid data leakage.
# Return Value :  X_train, X_test, Y_train, Y_test (all scaled)
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def prepare_model_data(Data):
    cleaned_data = Data.drop(columns=["LoanID"]).copy()

    # Yes/No columns -> binary
    yes_no_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
    for col in yes_no_cols:
        cleaned_data[col] = cleaned_data[col].map({"Yes": 1, "No": 0})

    # Remaining nominal categorical columns -> one-hot encoding
    categorical_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
    cleaned_data = pd.get_dummies(cleaned_data, columns=categorical_cols, drop_first=True)

    print("Columns after encoding : ")
    print(list(cleaned_data.columns))

    X = cleaned_data.drop("Default", axis=1)
    Y = cleaned_data["Default"]

    print("X Shape : ", X.shape)
    print("Y Shape : ", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

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
# Description :   Runs the full Loan Default Prediction pipeline end-to-end:
#                 load, visualize, encode/split/scale, train and compare
#                 Random Forest and Gradient Boosting
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    df = load_data("Loan_Default.csv")

    print(Border)
    print("Step 2 : Visualization of DataSet")
    print(Border)

    DisplayPlots(df)

    print(Border)
    print("Step 3 : Prepare Train/Test Data")
    print(Border)

    X_train, X_test, Y_train, Y_test = prepare_model_data(df)

    print(Border)
    print("Step 4 : Build, Train and Evaluate Models")
    print(Border)

    results = {}

    # class_weight="balanced" gives more weight to the minority (default)
    # class during training, since defaults are usually under-represented.
    results["Random Forest"] = evaluate_model(
        "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        X_train, X_test, Y_train, Y_test
    )

    results["Gradient Boosting"] = evaluate_model(
        "Gradient Boosting",
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        X_train, X_test, Y_train, Y_test
    )

    print(Border)
    print("Step 5 : Model Comparison")
    print(Border)

    for name, acc in results.items():
        print(f"{name} : {acc * 100:.2f} %")

    best_model_name = max(results, key=results.get)
    print(f"\nBest Model : {best_model_name} with {results[best_model_name] * 100:.2f} % accuracy")
    print("\nNOTE : Given class imbalance, also compare the 'Default' class")
    print("precision/recall in the classification reports above, not just accuracy.")


if __name__ == "__main__":
    main()
