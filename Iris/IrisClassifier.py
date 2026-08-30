#################################################################################
#
#  Project Name  : Iris Flower Classification
#  Description   : Loads the Iris dataset, performs exploratory data analysis,
#                   visualizes feature relationships, trains a Decision Tree
#                   classifier, and evaluates its performance.
#  Date          : 23-Aug-2026
#  Author        : Shubham Namdev Shitole
#
#################################################################################

import pandas as pd  # Used for EDA

# Used for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Used for Split the dataset for training and testing
from sklearn.model_selection import train_test_split

# Used for Build the model
from sklearn.tree import DecisionTreeClassifier

# Used for calculate the accuracy,display confusion matrix
from sklearn.metrics import(accuracy_score,confusion_matrix,classification_report)

Border = "-"*30

############################################################
# Step 1 : Load the Data Set
############################################################

print(Border)
print("Step 1 : Load the Data Set")
print(Border)

DataPath = "iris.csv"

df = pd.read_csv(DataPath)

print("DataSet Loaded Successfully")
print("Initial entries from dataSets are : ")
print(df.head())

############################################################
# Step 2 : Data Analysis (EDA)
############################################################

print(Border)
print("Step 2 : Data Analysis (EDA)")
print(Border)

print("Shape of dataset : ",df.shape)

print("Column names : ")
print(list(df.columns))
print("Missing values per column : ")
print(df.isnull().sum())
print("Class distribution (Species Count) : ")
print(df["species"].value_counts())
print("Statistical report of dataset : ")
print(df.describe())

############################################################
# Step 3 : Describe Independent & Dependent Variables
############################################################

print(Border)
print("Step 3 : Describe Independent & Dependent Variables")
print(Border)

# X :  Independent Variable / Feature
# Y :  Dependent  Variable / Lable

feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X = df[feature_cols]
Y = df["species"]

print("X Shape : ",X.shape)
print("Y Shape : ",Y.shape)

############################################################
# Step 4 : Visualization of DataSet
############################################################

print(Border)
print("Step 4 : Visualization of DataSet")
print(Border)

# Scatter plot

plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal length (cm)"],temp["petal width (cm)"],label = sp)

plt.title("Marvellous Iris Case Study")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid()
plt.show()


############################################################
# Step 5 : Split the dataset for training and testing
############################################################

print(Border)
print("Step 5 : Split the dataset for training and testing")
print(Border)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.5,random_state=42)

print("Dataset splitting activity done")

print("X : ",X.shape)   #(150,4)
print("Y : ",Y.shape)   #(150,)

print("X_train : ",X_train.shape) #(75,4)
print("X_test : ",X_test.shape) #(75,4)

print("Y_train : ",Y_train.shape) #(75,)
print("Y_test : ",Y_test.shape)   #(75,)


############################################################
# Step 6 : Build the model
############################################################

print(Border)
print("Step 6 : Build the model")
print(Border)

model = DecisionTreeClassifier(max_depth=5)
print("Model gets created successfully")

############################################################
# Step 7 : Train the model
############################################################

print(Border)
print("Step 7 : Train the model")
print(Border)

model.fit(X_train,Y_train)
print("Model trained succefully")

############################################################
# Step 8 : Evaluate the model
############################################################

print(Border)
print("Step 8 : Evaluate the model")
print(Border)

Y_pred = model.predict(X_test)

print("Model Evaluating done")

print("Expected answers : ")
print(Y_test)

print("Predicted answers : ")
print(Y_pred)

############################################################
# Step 9 : Evaluate the model performance
############################################################

print(Border)
print("Step 9 : Evaluate the model performance")
print(Border)

accurcy = accuracy_score(Y_test,Y_pred)
print("Accuarcy model is : ",accurcy * 100 ,"%")

print("Confusion Matrix : ")
cm = confusion_matrix(Y_test,Y_pred)
print(cm)

print("Classifiction Report : ")
print(classification_report(Y_test,Y_pred))
