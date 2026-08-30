#################################################################################
#
#  Project Name  : Titanic Survival Predictor
#  Description   : Loads the Titanic dataset, performs exploratory data
#                   analysis, handles missing data, engineers new features,
#                   trains and compares Logistic Regression, Decision Tree,
#                   and Random Forest classifiers, then tunes the Random
#                   Forest model using GridSearchCV.
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the Titanic CSV, prints an EDA summary (head, info,
#                 stats, missing value counts), and drops irrelevant columns
# Return Value :  Cleaned DataFrame with PassengerId and Ticket removed
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def load_data(data_path):
    df = pd.read_csv(data_path)
    
    print("="*50)
    print("1. FIRST 5 ENTRIES:")
    print(df.head())
    
    print("\n" + "="*50)
    print("2. DATA INFO (Columns & Dtypes):")
    
    df.info()
    
    print("\n" + "="*50)
    print("3. STATISTICAL SUMMARY:")
    print(df.describe())
    
    print("\n" + "="*50)
    print("4. NULL VALUE COUNTS (PER COLUMN):")
    print(df.isnull().sum())
    
    print("\n" + "="*50)
    print("5. PERCENTAGE OF MISSING DATA:")
    age_missing_avg = df['Age'].isna().mean() * 100
    cabin_missing_avg = df['Cabin'].isna().mean() * 100
    embarked_missing_avg = df['Embarked'].isna().mean() * 100
    print(f"Age: {age_missing_avg:.2f}%")
    print(f"Cabin: {cabin_missing_avg:.2f}%")
    print(f"Embarked: {embarked_missing_avg:.2f}%")
    
    print("\n" + "="*50)
    print("6. DROPPING IRRELEVANT COLUMNS (PassengerId, Ticket):")
    rData = df.drop(columns=["PassengerId", "Ticket"])
    rData.info()
    print(f"New Shape: {rData.shape}")
    
    return rData


#################################################################################
#
# Function Name : DisplayPlots
# Input :         Data
# Description :   Visualizes survival distribution against gender, passenger
#                 class, age, and fare
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def DisplayPlots(Data):
    sns.countplot(x='Survived', data=Data)
    plt.show()

    sns.countplot(x='Sex', hue='Survived', data=Data)
    plt.show()

    print("\n=== Survival Rate by Gender ===")
    print(Data.groupby('Sex')['Survived'].mean())

    sns.countplot(x="Pclass", hue='Survived', data=Data)
    plt.title("Passenger Class (Pclass)")
    plt.show()

    print("\n=== Survival Rate by Passenger Class ===")
    print(Data.groupby('Pclass')['Survived'].mean())

    plt.figure(figsize=(10,6))
    survived = Data[Data['Survived'] == 1]['Age'].dropna()
    died = Data[Data['Survived'] == 0]['Age'].dropna()
    plt.hist(survived, bins=20, alpha=0.6, label='Survived', color='green', edgecolor='black')
    plt.hist(died, bins=20, alpha=0.6, label='Died', color='red', edgecolor='black')
    plt.title("Numeric Distribution – Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    print("\n=== Average Age by Survival ===")
    print(Data.groupby('Survived')['Age'].mean())

    sns.boxplot(x='Pclass', y='Fare', data=Data)
    plt.title("Fare Distribution by Passenger Class")
    plt.show()


#################################################################################
#
# Function Name : handle_missing_data
# Input :         Data
# Description :   Fills missing Embarked values with the mode, missing Age
#                 values with the median age per passenger class, and
#                 converts Cabin into a binary Has_Cabin indicator
# Return Value :  Cleaned DataFrame with no missing values in the handled
#                 columns
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def handle_missing_data(Data):
    cleaned_data = Data.copy()
    
    print("\n" + "="*50)
    print("PHASE 3: HANDLING MISSING DATA")
    print("="*50)
    
    print("\n1. Fixing 'Embarked'...")
    print(f"   Missing before: {cleaned_data['Embarked'].isnull().sum()}")
    mode_embarked = cleaned_data['Embarked'].mode()[0]

    cleaned_data['Embarked'] = cleaned_data['Embarked'].fillna(mode_embarked)
    print(f"   Missing after: {cleaned_data['Embarked'].isnull().sum()}")
    print(f"   Filled with mode: '{mode_embarked}'")
    
    print("\n2. Fixing 'Age'...")
    print(f"   Missing before: {cleaned_data['Age'].isnull().sum()}")
    median_ages = cleaned_data.groupby('Pclass')['Age'].transform('median')
    cleaned_data['Age'] = cleaned_data['Age'].fillna(median_ages)
    print(f"   Missing after: {cleaned_data['Age'].isnull().sum()}")
    print("   Filled with median age per passenger class")
    
    print("\n3. Handling 'Cabin'...")
    print(f"   Missing before: {cleaned_data['Cabin'].isnull().sum()}")
    cleaned_data['Has_Cabin'] = cleaned_data['Cabin'].notna().astype(int)
    cleaned_data.drop(columns=['Cabin'], inplace=True)
    print(f"   Created 'Has_Cabin' (1 if cabin present, 0 otherwise)")
    print(f"   Original 'Cabin' dropped.")
    
    print("\n" + "="*50)
    print("VERIFICATION: Missing values after Phase 3:")
    print(cleaned_data.isnull().sum())
    print("="*50)
    
    return cleaned_data


#################################################################################
#
# Function Name : engineer_features
# Input :         Data
# Description :   Creates FamilySize, IsAlone, and Title features, grouping
#                 rare titles together, then drops the original Name column
# Return Value :  DataFrame with engineered features added
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def engineer_features(Data):
    cleaned_data = Data.copy()
    
    print("\n" + "="*50)
    print("PHASE 4: FEATURE ENGINEERING")
    print("="*50)
    
    print("\n1. Creating 'FamilySize'...")
    cleaned_data['FamilySize'] = cleaned_data['SibSp'] + cleaned_data['Parch'] + 1
    print(f"   FamilySize range: {cleaned_data['FamilySize'].min()} to {cleaned_data['FamilySize'].max()}")
    
    print("\n2. Creating 'IsAlone'...")
    cleaned_data['IsAlone'] = (cleaned_data['FamilySize'] == 1).astype(int)
    print(f"   Passengers traveling alone: {cleaned_data['IsAlone'].sum()}")
    print(f"   Passengers with family: {len(cleaned_data) - cleaned_data['IsAlone'].sum()}")
    
    print("\n3. Extracting 'Title' from Name...")
    cleaned_data['Title'] = cleaned_data['Name'].str.extract(r',\s*([A-Za-z]+)\.', expand=False)
    print("\n   All titles found:")
    print(cleaned_data['Title'].value_counts())
    
    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Don', 'Dona', 
                   'Lady', 'Capt', 'Countess', 'Jonkheer', 'Ms', 'Sir']
    cleaned_data['Title'] = cleaned_data['Title'].replace(rare_titles, 'Rare')
    print("\n   Titles after grouping rare ones:")
    print(cleaned_data['Title'].value_counts())
    
    print("\n4. Dropping 'Name' column...")
    cleaned_data.drop(columns=['Name'], inplace=True)
    print(f"   Remaining columns: {list(cleaned_data.columns)}")
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETE!")
    print(f"Final shape: {cleaned_data.shape}")
    print("="*50)
    
    return cleaned_data


#################################################################################
#
# Function Name : prepare_model_data
# Input :         Data
# Description :   Encodes categorical variables, splits the dataset into
#                 train/test sets, then scales Age and Fare. The scaler is
#                 fit ONLY on the training set and applied to the test set,
#                 to avoid data leakage.
# Return Value :  X_train, X_test, y_train, y_test
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def prepare_model_data(Data):
    data_frame = Data.copy()
    
    data_frame['Sex'] = data_frame['Sex'].map({'male': 0, 'female': 1})
    print("Sex after encoding:")
    print(data_frame['Sex'].unique())
    print(data_frame['Sex'].value_counts())
    
    data_frame = pd.get_dummies(data_frame, columns=['Embarked', 'Title'], drop_first=True)
    
    X = data_frame.drop('Survived', axis=1)
    y = data_frame['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    scaler = StandardScaler()
    # Fit the scaler ONLY on the training set...
    X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
    # ...then use those same fitted parameters to transform the test set.
    # The scaler never "sees" the test set's own mean/std.
    X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])
    
    print("X_train[Age] Mean:", X_train['Age'].mean())
    print("X_train[Age] Standard deviation:", X_train['Age'].std())
    print("X_train[Fare] Mean:", X_train['Fare'].mean())
    print("X_train[Fare] Standard deviation:", X_train['Fare'].std())
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


#################################################################################
#
# Function Name : tune_random_forest
# Input :         X_train, X_test, y_train, y_test
# Description :   Runs GridSearchCV over a Random Forest parameter grid to
#                 find the best-performing hyperparameter combination, then
#                 evaluates the tuned model on the test set
# Return Value :  best_model, test_accuracy
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def tune_random_forest(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("PHASE 7: HYPERPARAMETER TUNING (Random Forest)")
    print("="*50)
    
    param_grid = {
        'n_estimators': [50, 100, 200],           
        'max_depth': [10, 20, 30, None],         
        'min_samples_split': [2, 5, 10],         
        'min_samples_leaf': [1, 2, 4]             
    }
    
    print("\nParameter grid to search:")
    print(param_grid)
    print(f"\nTotal combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])}")
    
    rf = RandomForestClassifier(random_state=42)
    
    print("\nStarting GridSearchCV.")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,                     
        scoring='accuracy',
        n_jobs=-1,                
        verbose=1                 
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "-"*50)
    print("GRID SEARCH COMPLETE!")
    print("-"*50)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_tuned)
    
    print(f"\nTest Accuracy with Tuned Model: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_tuned))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_tuned))
    
    print("\nTop 5 Most Important Features (Tuned Model):")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(5)
    print(feature_importance)
    
    print("\n" + "="*50)
    print("PHASE 7 COMPLETE!")
    print("="*50)
    
    return best_model, test_accuracy


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full Titanic pipeline end-to-end: load, visualize,
#                 clean, engineer features, split/scale, compare default
#                 models, tune the Random Forest, and report final results
# Date :          23-Aug-2026
# Author :        Shubham Namdev Shitole
#
#################################################################################

def main():
    clean_df = load_data("Titanic.csv")
    
    DisplayPlots(clean_df)
    
    clean_df = handle_missing_data(clean_df)
    
    engineered_df = engineer_features(clean_df)
    
    X_train, X_test, y_train, y_test = prepare_model_data(engineered_df)
    print("\n Phase 5 Complete! Ready for Phase 6.")
    
    print("\n" + "="*50)
    print("PHASE 6: DEFAULT MODEL COMPARISON")
    print("="*50)
    
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, y_pred_log)
    print(f"\n Logistic Regression Accuracy: {log_acc:.4f}")
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Accuracy: {dt_acc:.4f}")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    best_default = max(log_acc, dt_acc, rf_acc)
    best_model_name = ['Logistic Regression', 'Decision Tree', 'Random Forest'][[log_acc, dt_acc, rf_acc].index(best_default)]
    print(f"\n Best Default Model: {best_model_name} with {best_default*100:.2f}%")
    
    print("\n" + "-"*50)
    print("Proceeding with Phase 7: Tuning Random Forest...")
    print("-"*50)
    
    tuned_model, tuned_accuracy = tune_random_forest(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*50)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*50)
    print(f"Default Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"Tuned Random Forest Accuracy:  {tuned_accuracy:.4f} ({tuned_accuracy*100:.2f}%)")
    print(f"Improvement: {tuned_accuracy - rf_acc:.4f} ({ (tuned_accuracy - rf_acc)*100:.2f}%)")
    
    print("\n Project Complete! Built and optimized a classification model on the Titanic dataset.")
    print("="*50)


if __name__ == "__main__":
    main()
