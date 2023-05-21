import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
ps=PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words('english')



rad=st.sidebar.radio("Navigation",["Home","Sentiment Analysis","Spam Detection","Sarcasm Detection"])

#Home Page
if rad=="Home":
    st.title("Text Analysis App")
    st.image("4.jpeg")
    st.text(" ")
    st.text("The following text analysiers are available:")
    st.text("1. Sentiment Analysis")
    st.text("2. Spam Detector")
    st.text("3. Sarcasm Detector")

#Data Cleaning and transformation:

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    c=[]
    for i in text:
        if i.isalnum():
            c.append(i)
    text=c[:]
    c=[]
    c=[ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(c)

#Sentiment Analysis

tfidf=TfidfVectorizer(stop_words=sw,max_features=20)
def transform(txt):
    textnew=tfidf.fit_transform(txt)
    return textnew.toarray()
df=pd.read_csv("./Sentiment Analysis.csv")
df.columns=["Text","Label"]
x=transform(df["Text"])
y=df["Label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)

if rad=="Sentiment Analysis":
    st.title("Get sentiment of the text")
    st.image("1.png",width=300)
    txt=st.text_area("Enter the text")
    transformed_txt=transform_text(txt)
    #tfidf
    vtxt=tfidf.transform([transformed_txt])
    prediction=model.predict(vtxt)[0]

    if st.button("Predict"):
        if prediction==0:
            st.warning("Negative Text:(")
        elif prediction==1:
            st.success("Positive Text:)")

#spam detection
tfidf=TfidfVectorizer(stop_words=sw,max_features=20)
def transform1(txt):
    textnew1=tfidf.fit_transform(txt)
    return textnew1.toarray()
df1=pd.read_csv("./Spam Detection.csv")
df1.columns=["Label","Text"]
x=transform1(df1["Text"])
y=df1["Label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
model1=LogisticRegression()
model1.fit(x_train,y_train)

if rad=="Spam Detection":
    st.title("Spam or Ham?")
    st.image("2.jpeg",width=300)
    txt1=st.text_area("Enter the text")
    transformed_txt1=transform_text(txt1)
    #tfidf
    vtxt1=tfidf.transform([transformed_txt1])
    prediction1=model1.predict(vtxt1)[0]

    if st.button("Predict"):
        if prediction1=="spam":
            st.warning("Spam Text:(")
        elif prediction1=="ham":
            st.success("Ham Text:)")


#Sarcasm 
tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt):
    textnew2=tfidf.fit_transform(txt)
    return textnew2.toarray()

df2=pd.read_csv("./Sarcasm Detection.csv")
df2.columns=["Text","Label"]
x=transform2(df2["Text"])
# Map labels to numeric values
label_mapping = {
    "figurative": 0,
    "irony": 0,
    "regular": 0,
    "sarcasm": 1
}

# Convert labels to numeric values
y = df2["Label"].map(label_mapping)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# Fit model
model2 = LogisticRegression()
model2.fit(x_train, y_train)

if rad == "Sarcasm Detection":
    st.title("Detect sarcastic text")
    st.image("3.jpg",width=300)
    txt2 = st.text_area("Enter the text")
    transformed_txt2 = transform_text(txt2)
    # tfidf
    vtxt2 = tfidf.transform([transformed_txt2])
    prediction2 = model2.predict(vtxt2)[0]

    if st.button("Predict"):
        if prediction2 == 0:
            st.info("Non-sarcastic Text")
        elif prediction2 == 1:
            st.success("Sarcastic Text :)")
