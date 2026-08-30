#################################################################################
#
#  Project Name  : Sentiment Analysis System
#  Description   : Loads the IMDB 50K Movie Reviews dataset, cleans review
#                   text (removing leftover HTML tags), converts text into
#                   numeric features using TF-IDF, then trains and compares
#                   two ensemble learning approaches — Bagging and
#                   AdaBoost (Boosting) — to classify review sentiment as
#                   positive or negative.
#  Date          : 23-Aug-2026
#  Author        : Shubham Namdev Shitole
#
#################################################################################

import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the IMDB reviews CSV and prints an EDA summary
#                 (shape, sample rows, sentiment class balance)
# Return Value :  Loaded DataFrame
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def load_data(data_path, sample_size=None):
    df = pd.read_csv(data_path)

    print("Full dataset shape : ", df.shape)

    if sample_size is not None and sample_size < len(df):
        df = (
            df.groupby("sentiment", group_keys=False)
            .apply(lambda g: g.sample(n=sample_size // 2, random_state=42))
            .reset_index(drop=True)
        )
        print(f"Using a sample of {len(df)} reviews (sample_size = {sample_size})")

    print("Shape of dataset used : ", df.shape)
    print("First few records : ")
    print(df.head())

    print(Border)
    print("Missing Values Per Column : ")
    print(df.isnull().sum())

    print(Border)
    print("Sentiment Class Distribution : ")
    print(df["sentiment"].value_counts())

    return df


#################################################################################
#
# Function Name : clean_text
# Input :         Data
# Description :   Removes leftover HTML tags (e.g. "<br />") that are a
#                 known artifact of how this dataset was originally
#                 scraped, and collapses extra whitespace
# Return Value :  DataFrame with a cleaned "review" column
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def clean_text(Data):
    cleaned_data = Data.copy()

    def _clean(text):
        text = re.sub(r"<.*?>", " ", text)       # strip HTML tags
        text = re.sub(r"\s+", " ", text).strip() # collapse whitespace
        return text

    print(Border)
    print("Cleaning review text (removing HTML tags)...")
    cleaned_data["review"] = cleaned_data["review"].apply(_clean)

    print("Sample cleaned review : ")
    print(cleaned_data["review"].iloc[0][:200], "...")

    return cleaned_data


#################################################################################
#
# Function Name : prepare_model_data
# Input :         Data, max_features
# Description :   Converts review text into numeric TF-IDF features
#                 (capped at max_features most frequent words, English
#                 stop words removed), encodes sentiment as 0/1, then
#                 splits into stratified train/test sets
# Return Value :  X_train, X_test, Y_train, Y_test (TF-IDF sparse matrices)
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def prepare_model_data(Data, max_features=5000):
    Y = Data["sentiment"].map({"positive": 1, "negative": 0})

    print(Border)
    print(f"Vectorizing text using TF-IDF (max_features = {max_features})...")

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(Data["review"])

    print("TF-IDF feature matrix shape : ", X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

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
# Description :   Runs the full Sentiment Analysis pipeline end-to-end:
#                 load, clean text, vectorize with TF-IDF, train and
#                 compare Bagging and AdaBoost ensemble classifiers on the
#                 full 50,000-review IMDB dataset
#
#                 NOTE : Running on the full dataset (rather than a
#                 subset) means this can take a while to complete,
#                 especially the Bagging step, since it trains multiple
#                 Decision Trees on a large TF-IDF feature matrix. This is
#                 expected — let it run.
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)
    
    df = load_data("IMDB Dataset.csv", sample_size=10000)

    print(Border)
    print("Step 2 : Clean Review Text")
    print(Border)

    clean_df = clean_text(df)

    print(Border)
    print("Step 3 : Vectorize Text and Split Train/Test")
    print(Border)

    X_train, X_test, Y_train, Y_test = prepare_model_data(clean_df, max_features=5000)

    print(Border)
    print("Step 4 : Build, Train and Evaluate Models")
    print(Border)

    results = {}

    # Bagging - multiple Decision Trees trained on random subsets of the
    # data, then majority-voted
    base_model = DecisionTreeClassifier(random_state=42)
    bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42, n_jobs=-1)
    results["Bagging"] = evaluate_model(
        "Bagging", bagging_model,
        X_train, X_test, Y_train, Y_test
    )

    # AdaBoost - trees built sequentially, each one focusing more on the
    # examples the previous tree got wrong
    boosting_model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
    results["AdaBoost"] = evaluate_model(
        "AdaBoost (Boosting)", boosting_model,
        X_train, X_test, Y_train, Y_test
    )

    print(Border)
    print("Step 5 : Model Comparison")
    print(Border)

    for name, acc in results.items():
        print(f"{name} : {acc * 100:.2f} %")

    best_model_name = max(results, key=results.get)
    print(f"\nBest Model : {best_model_name} with {results[best_model_name] * 100:.2f} % accuracy")


if __name__ == "__main__":
    main()
