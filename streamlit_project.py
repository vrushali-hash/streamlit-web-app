
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier


st.title(" Data Analysis Web App")
option=st.sidebar.selectbox("choose option:",['EDA','visualization','model','about'])
def act():
    if option=='EDA':
        st.subheader("exploratory data analysis")
        data=st.file_uploader("data loaded:",type=['csv','txt','xlsx','json'])
        st.success("data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df)
            if st.checkbox("Display columns"):
                st.write(df.columns)
                if st.checkbox("Display null values"):
                    
                    st.write(df.isnull().sum())
                    if st.checkbox("Display type"):
                        st.write(df.dtypes)
                        if st.checkbox("Display shape"):
                            st.write(df.shape)
                            if st.checkbox("Display summary"):
                                st.write(df.describe())
                                if st.checkbox("select multicolumns"):
                                    select_col=st.multiselect("select multicolumns:",df.columns)
                                    df1=df[select_col]
                                    st.dataframe(df1)
                                    if st.button("Head"):
                                        st.write(df.head())
                                        
                                        
                                        
    elif option=='visualization':
        st.subheader("visulzation part")
        data=st.file_uploader("data loaded:",type=['csv','txt','xlsx','json'])
        st.success("data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df)
            if st.checkbox("Display multicolumns"):
                select_col=st.multiselect("selected columns:",df.columns)
                df1=df[select_col]
                st.dataframe(df1)
                if st.checkbox("Display pairplot"):
                    st.write(sns.pairplot(df,diag_kind='kde'))
                    st.pyplot()
                    if st.checkbox("simple bar plot"):
                        df.plot(kind='bar')
                        st.pyplot()
                        if st.checkbox("bar plot of group with count"):
                            v_count=df.groupby('class')
                            st.bar_chart(v_count)
                            if st.checkbox("data histogram"):
                                df.hist(bins=20)
                                st.pyplot()
                                if st.checkbox("distribution of varible"):
                                    st.write(sns.distplot(df['class']))
                                    st.pyplot()
                                
                            
                
        
    elif option=='model':
        st.subheader("modeling")
        data=st.file_uploader("data loaded:",type=['csv','txt','xlsx','json'])
        st.success("data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df)
        algorithum=['Logistic reg','KNN','SVM','Desion tree']
        classifier_name=st.sidebar.selectbox("select option:",algorithum)
        seed=st.sidebar.slider("seed",1,200)
                                                                  
        x=df.iloc[:,0:-1]
        y=df.iloc[:,-1]
        
        ##split data##
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=seed)
        
        def add_parameter(name_of_clf):
            params=dict()
            if name_of_clf=='SVM':
                c=st.sidebar.slider('C',0.01,15.0)
                params['c']=c
            else:
                name_of_clf=='KNN'
                
                k=st.sidebar.slider('K',0.01,15.0)
                params['k']=k
            return params
        params=add_parameter(classifier_name)
        def get_classifier(name_of_clf,params):
            clf=None
            if name_of_clf=='SVM':
                clf=SVC(C=params['c'])
            elif name_of_clf=='KNN':
                clf=KNeighborsClassifier(n_neighbors=params['k'])
            elif name_of_clf=='Desion tree':
                clf=DecisionTreeClassifier()
            elif name_of_clf=='Logistic reg':
                clf=LogisticRegression()
            else:
                st.warning('select your choice of algorithum')
            return clf
        clf=get_classifier(classifier_name,params)
        
        clf.fit(x_train,y_train)
        y_pred=clf.predict(x_test)
        st.write('prediction:',y_pred)
        accuracy=accuracy_score(y_test,y_pred)
        st.write('accuracy_score:',accuracy)
        
    
act()        


           
        
            
            
            
        
                               
                               
    
        
        
            
        
            
                         
     
