import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import streamlit as st
import sqlite3
import http.server
from PIL import Image
import plotly.express as px
#from pydantic_settings import BaseSettings
import time
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI

client = OpenAI()

def upload_file(filepath):
      file = client.files.create(
      file=open("files/output_san_cristobal/DS_1.pdf", "rb"),
      purpose="assistants"
     )

      print("FILE ID: ", file.id)
      return file


def create_assistant(name,model,tools,file_id):
    assistant = client.beta.assistants.create(
         name=name,
         model=model,
         tools=[{"type": tools}],
         file_ids=[file_id.id]
     )

    thread = client.beta.threads.create()

    print("THREAD ID: ", thread.id)
    return thread
     



def chat_gpt_assistant(system,prompt,thread_id, assistant_id, file_id):

    message = client.beta.threads.messages.create(
        #thread_id="thread_objI64Kj60FdS2EQrNVB04EN",
        thread_id=thread_id,
        role="user",
        content=prompt,
        #file_ids=["file-5rQuwUoQzjjKSX8XvTc8W9eJ"]
        file_ids=file_id
    )

    print("MESSAGE ID: ", message.id)

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        #assistant_id="asst_pSbwUNXpra1hWUQ8cKN8in6U",
        assistant_id=assistant_id,
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}]
    )

    print("RUN ID: ", run.id)

    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if (run.completed_at):
            print("ESPERANDO...")
            break

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_message = messages.data[0]
    text = last_message.content[0].text.value
    print("RESPUESTA: ", text)
    return text


jd = st.text_area("Job Description summary", "")

def main():
    if st.button('procesar query'):
        if jd:
            system="Eres un asistente util"
            prompt=jd
            assistant_id="asst_pSbwUNXpra1hWUQ8cKN8in6U"
            thread_id="thread_objI64Kj60FdS2EQrNVB04EN"
            file_id=["file-5rQuwUoQzjjKSX8XvTc8W9eJ"]
            text=chat_gpt_assistant(system,prompt,thread_id, assistant_id, file_id)
            st.write(text)
        else:
            st.write("Please describe de job position")



