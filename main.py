from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

os.environ["AWS_PROFILE"] = "default"

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

modelID = "meta.llama2-13b-chat-v1"


llm = BedrockLLM(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"temperature": 0.9},
)


def my_chatbot(language, freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}",
        temperature=0.0,
        max_tokens=100,
    )
    bedrock_chain = prompt | llm
    response = bedrock_chain.invoke(
        {"language": language, "freeform_text": freeform_text}
    )
    return response


st.title("Bedrock Chatbot")
language = st.sidebar.selectbox("Language", ["English", "Spanish", "French", "German"])
if language:
    freeform_text = st.sidebar.text_input("Enter your text", max_chars=100)

if freeform_text:
    response = my_chatbot(language, freeform_text)
    st.write(response)
