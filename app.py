import streamlit as st
import numpy as np
import pandas as pd
import pickle

pickli_in = open("spam-sms-mnb-model.pkl","rb")
model = pickle.load(pickli_in)
tfidf = pickle.load(open('tfidf-transform.pkl','rb'))


def welcome():
    return "welcome All"

def app_prediction(message):
    data = [message]
    vect = tfidf.transform(data).toarray()
    my_prediction = model.predict(vect)
    return my_prediction
    
def main():
    st.title("SMS-Spam-Classifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit SMS-Spam-Classifier ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    message = st.text_area("Enter your text","Type Here")
    result=""
    if st.button("Predict"):
        result = app_prediction(message)
    st.success('The message is {}'.format(result))

if __name__=='__main__':
    main()