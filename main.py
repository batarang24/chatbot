import streamlit as st
import joblib

model=joblib.load('chatbot.pkl')
cv=joblib.load('cv.pkl')
mssg=cv.transform(['hello'])
print(model.predict(mssg))
prompt = st.chat_input("Say something")
if prompt:
    help=[prompt]
    mssg=cv.transform(help)
    pred=model.predict(mssg)
    st.write(f"User has sent the following prompt: {pred[0]}")


