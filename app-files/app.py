import streamlit as st
import pickle
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

with open('NLP-category.pkl', 'rb') as f:
    svcat = pickle.load(f)
with open('NLP-sentiment.pkl', 'rb') as f:
    svsent = pickle.load(f)
with open('reviewstrain.pkl', 'rb') as f:
    trainrev = pickle.load(f)

vectorizer = CountVectorizer(binary = True)
train = vectorizer.fit_transform(trainrev)

def predict_cat(rev):
    test_rev = vectorizer.transform([rev])
    cat = svcat.predict(test_rev)
    return cat[0]

def predict_sen(rev):
    test_rev = vectorizer.transform([rev])
    sen = svsent.predict(test_rev)
    return sen[0]

st.title('Category and Sentiment Analyzer')
txt = st.text_area('Paste a product review below to analyze it', '''
    ''')
c = predict_cat(txt)
s = predict_sen(txt)
if st.button('Analyze Category'):
    st.write(c)
else:
    st.write('category')

if st.button('Analyze Sentiment'):
    st.write(s)
else:
    st.write('sentiment')