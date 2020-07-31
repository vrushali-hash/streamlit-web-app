
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from PIL import Image

st.title("Banknote Analysis Prediction")

variance=st.number_input("Variance")

skewness=st.number_input("Skewness")
                   
curtosis=st.number_input("curtosis")

Entropy=st.number_input("Entropy")                   

test_data=([[variance,skewness,curtosis,Entropy]])
if st.button("Predict"):
    from sklearn.model_selection import train_test_split
    data=pd.read_csv("C:/Users/Admin/Downloads/BankNote_Authentication.csv")
    df=pd.DataFrame(data)
    x=df.drop(['class'],axis=1)
    y=df['class']
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    sc.fit_transform(x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    from sklearn.linear_model import LogisticRegression
    model1=LogisticRegression()
    model1.fit(x_train,y_train)
    y_pred=model1.predict(x_test)
    from sklearn.metrics import accuracy_score
    accu=accuracy_score(y_pred,y_test)
    output=model1.predict(test_data)

    st.write("Class :",output)
    
