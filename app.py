import nltk
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    z = []
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            z.append(i)

    a = []
    for i in z:
        a.append(ps.stem(i))
    return " ".join(a)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS SPAM CLASSIFIER")
input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
