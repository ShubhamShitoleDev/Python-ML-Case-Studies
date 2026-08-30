#################################################################################
#
#  Project Name  : Industrial ML Pipeline
#  Description   : A reusable, end-to-end machine learning pipeline that
#                   can be pointed at any tabular dataset. Automatically
#                   detects numeric and categorical columns, handles
#                   missing values, encodes and scales features, trains a
#                   model, evaluates it, visualizes results, and generates
#                   predictions on new raw data using the exact same
#                   preprocessing steps that were fit during training.
#                   Demonstrated end-to-end on the Diabetes dataset.
#  Date          : 23-Aug-2026
#  Author        : Shubham Namdev Shitole
#
#################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads any CSV file and prints a basic EDA summary
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
    print("Missing Values Per Column : ")
    print(df.isnull().sum())

    return df


#################################################################################
#
# Function Name : preprocess_data
# Input :         Data, target_column
# Description :   Generic preprocessing step that works on ANY dataset:
#                 automatically detects numeric vs categorical columns,
#                 imputes missing numeric values with the median and
#                 missing categorical values with the mode, one-hot encodes
#                 categorical columns, and scales numeric columns. Returns
#                 the fitted transformers alongside the processed data, so
#                 the exact same preprocessing can later be reapplied to
#                 brand new, unseen data during prediction.
# Return Value :  X (processed features), Y (target), pipeline_state (dict
#                 of fitted transformers + column info needed to
#                 preprocess new data identically)
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def preprocess_data(Data, target_column):
    cleaned_data = Data.copy()

    Y = cleaned_data[target_column]
    X = cleaned_data.drop(columns=[target_column])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Detected numeric columns : {numeric_cols}")
    print(f"Detected categorical columns : {categorical_cols}")

    # Impute missing numeric values with the median
    numeric_imputer = SimpleImputer(strategy="median")
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

    # Impute missing categorical values with the most frequent value
    encoder = None
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

        # One-hot encode categorical columns
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_array = encoder.fit_transform(X[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=X.index)

        X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
    else:
        categorical_imputer = None

    # Scale numeric columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print("Processed feature shape : ", X.shape)

    pipeline_state = {
        "target_column": target_column,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_imputer": numeric_imputer,
        "categorical_imputer": categorical_imputer,
        "encoder": encoder,
        "scaler": scaler,
        "final_columns": X.columns.tolist(),
    }

    return X, Y, pipeline_state


#################################################################################
#
# Function Name : train_model
# Input :         model, X_train, Y_train
# Description :   Trains the given model on the training data
# Return Value :  The trained model
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def train_model(model, X_train, Y_train):
    print(Border)
    print(f"Training model : {type(model).__name__}")
    print(Border)

    model.fit(X_train, Y_train)
    print("Model training complete")

    return model


#################################################################################
#
# Function Name : evaluate_model
# Input :         model, X_test, Y_test
# Description :   Evaluates the trained model on the test set and prints
#                 accuracy, confusion matrix, and classification report
# Return Value :  Y_pred (predictions on the test set), accuracy
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def evaluate_model(model, X_test, Y_test):
    print(Border)
    print("Evaluating model on test data")
    print(Border)

    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    print(f"Accuracy : {accuracy * 100:.2f} %")
    print("Confusion Matrix : ")
    print(confusion_matrix(Y_test, Y_pred))
    print("Classification Report : ")
    print(classification_report(Y_test, Y_pred))

    return Y_pred, accuracy


#################################################################################
#
# Function Name : visualize_results
# Input :         model, X_test, Y_test, Y_pred, feature_names
# Description :   Visualizes the confusion matrix as a heatmap, and the
#                 model's feature importances (if the model supports them)
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def visualize_results(model, Y_test, Y_pred, feature_names):
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)

        plt.figure(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=importance)
        plt.title("Top 10 Feature Importances")
        plt.show()
    else:
        print(f"{type(model).__name__} does not expose feature_importances_, skipping that plot")


#################################################################################
#
# Function Name : generate_predictions
# Input :         model, pipeline_state, new_data
# Description :   Applies the EXACT same preprocessing steps that were fit
#                 during training (same imputers, same encoder, same
#                 scaler) to brand new raw data, then returns the model's
#                 predictions. This is what makes the pipeline usable
#                 beyond a one-off script — new data goes through
#                 identical preprocessing before prediction, the way a
#                 deployed model would need to.
# Return Value :  Array of predictions for new_data
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def generate_predictions(model, pipeline_state, new_data):
    X_new = new_data.copy()

    numeric_cols = pipeline_state["numeric_cols"]
    categorical_cols = pipeline_state["categorical_cols"]

    X_new[numeric_cols] = pipeline_state["numeric_imputer"].transform(X_new[numeric_cols])

    if categorical_cols:
        X_new[categorical_cols] = pipeline_state["categorical_imputer"].transform(X_new[categorical_cols])
        encoded_array = pipeline_state["encoder"].transform(X_new[categorical_cols])
        encoded_cols = pipeline_state["encoder"].get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=X_new.index)
        X_new = pd.concat([X_new.drop(columns=categorical_cols), encoded_df], axis=1)

    X_new[numeric_cols] = pipeline_state["scaler"].transform(X_new[numeric_cols])

    # Ensure column order matches what the model was trained on
    X_new = X_new.reindex(columns=pipeline_state["final_columns"], fill_value=0)

    predictions = model.predict(X_new)
    return predictions


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Demonstrates the full reusable pipeline end-to-end on the
#                 Diabetes dataset: load, preprocess, split, train,
#                 evaluate, visualize, and generate predictions on new
#                 sample data
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
    print("Step 2 : Preprocess Data (Generic Pipeline Step)")
    print(Border)

    X, Y, pipeline_state = preprocess_data(df, target_column="Outcome")

    print(Border)
    print("Step 3 : Split Train/Test")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    print("X_train : ", X_train.shape)
    print("X_test : ", X_test.shape)

    print(Border)
    print("Step 4 : Train Model")
    print(Border)

    model = train_model(RandomForestClassifier(n_estimators=100, random_state=42), X_train, Y_train)

    print(Border)
    print("Step 5 : Evaluate Model")
    print(Border)

    Y_pred, accuracy = evaluate_model(model, X_test, Y_test)

    print(Border)
    print("Step 6 : Visualize Results")
    print(Border)

    visualize_results(model, Y_test, Y_pred, feature_names=X.columns.tolist())

    print(Border)
    print("Step 7 : Generate Predictions on New Data")
    print(Border)

    new_sample = pd.DataFrame([{
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 90,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35
    }])

    prediction = generate_predictions(model, pipeline_state, new_sample)
    print("New sample input : ")
    print(new_sample)
    print(f"Predicted Outcome : {prediction[0]} (0 = No Diabetes, 1 = Diabetes)")


if __name__ == "__main__":
    main()
