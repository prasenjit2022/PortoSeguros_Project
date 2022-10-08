import warnings
from fileinput import filename
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
st.title("Porto Seguro’s Safe Driver Prediction") # Title of the page
st.write("This app Predicts whether the driver will file an insurance Claim or not next year")
image=Image.open('PortoSeguro.jpg')
st.image(image)
uploaded_file=st.file_uploader("Waiting for uploading a file")
# Once file is uploaded by user we are showing the data frame
if uploaded_file is not None:
    dataframe=pd.read_csv(uploaded_file)
    test_id=dataframe['id']
    y=dataframe['target']
    X=dataframe.drop(['target','id'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    st.write(f"Shape of Train Dataset = {X_train.shape}")
    st.write(f"Shape of Test Dataset = {y_test.shape}")
else:
    st.write("Please upload file for Prediction")
#Function for calculating gini coefficient
def gini(actual, predictions):
    assert (len(actual) == len(predictions))
    all = np.asarray(np.c_[actual, predictions, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
def gini_normalized(actual, predictions):
    return gini(actual, predictions) / gini(actual, actual)
# Side bar for selecting the different classifier
classfier_name=st.sidebar.selectbox("Classifier_Name",("Adaboost","XGBoost","Random Forest","SGDClassifier"))
if classfier_name=="XGBoost":  # code for loading the different models saved.
    with open('finalmodel.sav','rb') as f:
        model=pickle.load(f)
elif classfier_name=="Adaboost":
    with open('Adaboost_model.sav','rb') as m:
        model=pickle.load(m)
elif classfier_name=="SGDClassifier":
    with open('SGDClassifier_model.sav','rb') as s:
        model=pickle.load(s)
else:
    with open('RandomForest_model.sav','rb') as r:
        model=pickle.load(r)

# Calculate gini score once Predict button is pressed.
if st.button('Predict'):
    pred=model.predict_proba(X_test)[:,1]
    gini_score=round(gini_normalized(y_test,pred),2)
    Prediction=pd.DataFrame(pred)
    Prediction['target']= dataframe['target'].map({0:'Claim not filed',1:'claimed filed'})
    st.write("Predicted Data for Drivers Filed a Claim or not")
    st.write(Prediction['target'])
    st.write(f"gini_score = {gini_score}")
    st.write(f"Model_Name = {classfier_name}")

