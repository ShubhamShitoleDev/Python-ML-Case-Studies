#################################################################################
#
#  Project Name  : Wine Classification
#  Description   : Loads the Wine dataset, performs exploratory data
#                   analysis, visualizes feature distributions and
#                   correlations, trains and evaluates KNN, Decision Tree,
#                   and Random Forest classifiers to predict wine class.
#  Date          : 23-Aug-2026
#  Author        : Shubham Shitole
#
#################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the Wine CSV and prints an EDA summary (head, info,
#                 stats, missing value counts)
# Return Value :  Loaded DataFrame
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_data(data_path):
    df = pd.read_csv(data_path)

    print("="*50)
    print("1. FIRST 5 ENTRIES:")
    print(df.head())

    print("\n" + "="*50)
    print("2. Data INFO (Column & DataTypes) : ")
    df.info()

    print("\n" + "="*50)
    print("3. STATISTICAL SUMMARY : ")
    print(df.describe())

    print("\n" + "=" *50)
    print("4. NULL VALUE COUNTS (PER COLUMN) : ")
    print(df.isnull().sum())

    return df


#################################################################################
#
# Function Name : DisplayPlots
# Input :         Data
# Description :   Visualizes wine class distribution, feature histograms,
#                 alcohol content by class, and feature correlation matrix
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def DisplayPlots(Data):
    sns.countplot(x='Class',data=Data)
    plt.title("Distribution of wine classes")
    plt.show()

    Data.hist(figsize= (15,10),bins = 20)
    plt.suptitle("Feature Distributions")
    plt.show()

    sns.boxenplot(x='Class',y='Alcohol',data=Data)
    plt.title("Alcohol Distribution by Wine Class")
    plt.show()

    plt.figure(figsize=(12,8))
    sns.heatmap(Data.corr(),annot=True,cmap='coolwarm',fmt='.2f')
    plt.title("Feature Correaltion Matrix")
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
# Author :        Shubham Shitole
#
#################################################################################

def prepare_model_data(Data):
    print(Border)
    print("Step 3 : Describe Independent & Dependent Variables")
    print(Border)

    # X : Independent Variables / Features
    # Y : Dependent Variable / Label
    X = Data.drop(columns=['Class'])
    Y = Data['Class']

    print("X Shape : ", X.shape)
    print("Y Shape : ", Y.shape)

    print(Border)
    print("Step 4 : Split the dataset for training and testing")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("X_train : ", X_train.shape)
    print("X_test : ", X_test.shape)
    print("Y_train : ", Y_train.shape)
    print("Y_test : ", Y_test.shape)

    print(Border)
    print("Step 5 : Feature Scaling")
    print(Border)

    # Fit the scaler ONLY on the training set...
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("Feature scaling done using StandardScaler (fit on training set only)")

    return X_train, X_test, Y_train, Y_test


#################################################################################
#
# Function Name : evaluate_model
# Input :         model_name, model, X_train, X_test, Y_train, Y_test
# Description :   Trains the given model and prints accuracy, confusion
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
# Description :   Runs the full Wine classification pipeline end-to-end:
#                 load, visualize, split/scale, train and compare KNN,
#                 Decision Tree, and Random Forest models
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    clean_df = load_data("WinePredictor.csv")

    print(Border)
    print("Step 2 : Visualization of DataSet")
    print(Border)

    DisplayPlots(clean_df)

    X_train, X_test, Y_train, Y_test = prepare_model_data(clean_df)

    print(Border)
    print("Step 6 : Build, Train and Evaluate Models")
    print(Border)

    results = {}

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
    print("Step 7 : Model Comparison")
    print(Border)

    for name, acc in results.items():
        print(f"{name} : {acc * 100:.2f} %")

    best_model_name = max(results, key=results.get)
    print(f"\nBest Model : {best_model_name} with {results[best_model_name] * 100:.2f} % accuracy")


if __name__ == "__main__":
    main()
