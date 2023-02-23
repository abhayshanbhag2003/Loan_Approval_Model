import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


df = pd.read_csv('Loan Prediction Dataset.csv')




"""First replace all the None with either mean or mde of that variable throughout the dataset."""

df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
# df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
# df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())



"""Label Encoding to convert objects to float value"""

# Label Encoding for better analysis
from sklearn.preprocessing import LabelEncoder

Encoder = LabelEncoder()

# this is a list of all features
cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area", "Loan_Status"]

# Looping through all the features
for c in cols:
    df[c] = Encoder.fit_transform(df[c])



X = ["Gender","Married", "Education", "Dependents","LoanAmount", "Credit_History",   "Property_Area"]
Y = ["Loan_Status"]

# This is a generalised code to check which algorithm gives best possible result
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import warnings
import pickle
warnings.filterwarnings("ignore")

# def classify(x, y,model):  # x and y are input and output variable,Model is an object of any type of classifiaction technique
#     x_train, x_test, y_train, y_test = train_test_split(df[X], df["Loan_Status"], test_size=0.25, random_state=42)
#     model.fit(x_train, y_train)
#     pickle.dump(model, open('model.pkl', 'wb'))
#     model = pickle.load(open('model.pkl', 'rb'))

x_train, x_test, y_train, y_test = train_test_split(df[X], df["Loan_Status"], test_size=0.25, random_state=42)
LR = LogisticRegression()
LR.fit(x_train, y_train)
print(LR.predict([[0,1,0,133,3,1,2]]))
print(LR.predict([[0,1,3,133,3,1,2]]))

pickle.dump(LR, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))


# Fitting that model
# classify(X, Y, LR)


