import streamlit as st
from transformers import pipeline


st.title('Langchain Demo With Summarization Of Text')
input_text=st.text_input("Input text")

model_path = "Chessmen/summary"
summary_task  = pipeline(
    "summarization",
    model=model_path,
)

if input_text:
	st.write(summary_task(input_text))
    