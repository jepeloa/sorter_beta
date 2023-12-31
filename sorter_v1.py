#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from nltk.tokenize import word_tokenize
#from numpy.linalg import norm
#from termcolor import colored
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re
import plotly.graph_objects as go
#import nltk
#nltk.download('punkt')
import os
import streamlit as st
import sqlite3
import http.server
from PIL import Image
import socketserver
import threading
import plotly.express as px
#from pydantic_settings import BaseSettings
import chromadb
import time
from pyresparser import ResumeParser
import openai
#openai.api_key = os.environ["OPENAI_API_KEY"]


jd=' '

from chromadb.utils import embedding_functions
client=chromadb.PersistentClient(path="./db")
#client = chromadb.Client()

conn = sqlite3.connect('pdf_database.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS mis_documentos (
    id INTEGER PRIMARY KEY,
    Filename TEXT
)
''')

#En esta parte 


def chat_gpt_action(system,prompt):
    response = openai.ChatCompletion.create(
    model='gpt-4-1106-preview',
    max_tokens=1500,
    messages=[
        {"role": "system", "content": f"{system}"},
        {"role": "user", "content": f"{prompt}"},
    ])
    message = response.choices[0]['message']
    print("{}: {}".format(message['role'], message['content']))
    return message

 

image = Image.open('logo.png')

st.sidebar.image(image, caption=' ', width=200)

uploaded_files=st.sidebar.file_uploader("Upload CVs", accept_multiple_files=True, type="pdf", key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="collapsed")
#JD_files=st.sidebar.file_uploader("Upload JD", accept_multiple_files=True, type="pdf", key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="collapsed")

####################################################
def store_to_sqlite(df):
    conn = sqlite3.connect('pdf_database.db')
    df.to_sql('pdf_list', conn, if_exists='replace', index=False)  # Guarda el dataframe en la tabla 'pdf_list'
    conn.close()


def read_from_sqlite():
    conn = sqlite3.connect('pdf_database.db')
    df = pd.read_sql('SELECT * FROM pdf_list', conn)
    conn.close()
    return df

def init_db():
    conn = sqlite3.connect('pdf_database.db')
    cursor = conn.cursor()
    
    # Crear la tabla "pdf_list" si no existe
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_list (
            Filename TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

###########################################################

def obtain_skills(pdf_file):
    data = []
    resume_data = ResumeParser(pdf_file).get_extracted_data()
    skills = resume_data.get('skills', [])
    return skills



def start_pdf_server(port=8081):
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    print(f"PDF server started at port {port}")

def read_chroma_db(query,quantity=1):
    cv_collection = client.get_or_create_collection(name="cv_collection")
    results = cv_collection.query(
    query_texts=[query],
    n_results=quantity
    )
    return results
    


###########################

# Inicia el servidor al comienzo del script



#####################################################################

def save_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        with open(os.path.join('./CV', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getvalue())
        #st.sidebar.success(f"Archivo {uploaded_file.name} guardado con éxito")

def save_JD_files(JD_files):
    for uploaded_file in JD_files:
        with open(os.path.join('./JD', "JD.pdf"), 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.sidebar.success(f"Archivo {uploaded_file.name} guardado con éxito")



def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                st.sidebar.success(f"Archivo {filename} eliminado con éxito")
        except Exception as e:
            st.error(f"Error al eliminar el archivo {filename}: {e}")




if uploaded_files:
            #client.delete_collection(name="cv_collection")
            save_uploaded_files(uploaded_files)
            
#if JD_files:
 #   if st.button('Upload JD_file'):
  #      save_JD_files(JD_files)



conn = sqlite3.connect('pdf_database.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS mis_documentos (
    id INTEGER PRIMARY KEY,
    Filename TEXT
)
''')


def delete_table_contents():
    try:
        conn = sqlite3.connect('pdf_database.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM pdf_list')
        conn.commit()
        conn.close()
        
        return "Contenido de la tabla borrado con éxito."
    except Exception as e:
        return f"Error al borrar el contenido de la tabla: {e}"
    

##################################################################





###################################################################






def store_CV_in_db(file_data):
    documents = []
    metadatas = []
    ids = []

    for index, data in enumerate(file_data):
        documents.append(data['content'])
        metadatas.append({'source': data['file_name']})
        ids.append(str(index + 1))

    # create collection of pet files 
    try:
        client.delete_collection(name="cv_collection")   #borro si puedo la coleccion
    except:
        pass
    cv_collection = client.get_or_create_collection("cv_collection")

    # add files to the chromadb collection
    cv_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    return cv_collection

def dividir_texto(texto, limite_palabras=256):
    palabras = texto.split()
    segmentos = []
    segmento_actual = []
    contador_palabras = 0

    for palabra in palabras:
        segmento_actual.append(palabra)
        contador_palabras += 1

        if contador_palabras >= limite_palabras:
            segmentos.append(' '.join(segmento_actual))
            segmento_actual = []
            contador_palabras = 0

    if segmento_actual:
        segmentos.append(' '.join(segmento_actual))

    return segmentos

def read_dataset(path_to_folder):
    df = pd.read_excel(f"{path_to_folder}/dataset.xlsx")
    df['data']=df['data'].astype(str)
    return df



def read_CV_from_pdf(path_to_folder):
    file_data = []
    all_files = os.listdir(path_to_folder)
    # Filtra solo los archivos PDF
    pdf_files = [file for file in all_files if file.endswith('.pdf')]
    j=0 #contador de archivos
    for pdf_file in pdf_files:
        pdf_path = os.path.join(path_to_folder, pdf_file)
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            resume = ''
            for i in range(len(pdf.pages)):
                pageObj = pdf.pages[i]
                text_to=pageObj.extract_text()
                resume+=text_to
        segmentos = dividir_texto(resume, 256)
        for segmento in segmentos:
            file_data.append({"file_name": pdf_file, "content": segmento})
        
        j=j+1
        progress_bar.progress(int(j*100/len(all_files)),text=f"processing: {pdf_file}")
        print("="*50)
    return file_data


def read_JD_from_pdf(path_to_folder="./JD/JD.pdf"):
        with open(path_to_folder, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            JD = ''
            for i in range(len(pdf.pages)):
                pageObj = pdf.pages[i]
                text_to=pageObj.extract_text()
                JD+=text_to  
    
        return JD

# Text area for the user to input the job description
st.title("CV Sorter")
st.write("Insert the job description and get the matching CVs.")
#jd = st.text_area("Job Description summary", "")
progress_bar = st.progress(0)
delete_cvs=st.sidebar.button('Borrar curriculums')
jd = st.text_area("Job Description summary", "")
process=st.button('procesar query')
delete_query = st.button('delete')

if delete_cvs:
        delete_files_in_directory('./CV')



def main():
    path_to_folder='./CV/'
    init_db()
    try:
        start_pdf_server()
    except:
        print("No se pudo inciar el servidor pdf")


    df_sorted=pd.DataFrame
    #model = Doc2Vec.load('cv_job_maching.model')

    
   
    if st.sidebar.button("procesar"):
        file_data = read_CV_from_pdf(path_to_folder)  #extraigo datos de los pdf
        #file_data=read_dataset("./")
        #st.write(file_data)
        cv_collection=store_CV_in_db(file_data)
        
            
    if delete_query:
        message = delete_table_contents()
        st.write(message)
    if process:
        if jd:
            #system_prompt="Eres un asistente util"
            #user_prompt="Sumariza el siguiente puesto de trabajo en no mas de 30 palabras"
            #jdsum=chat_gpt_action(jd,system_prompt,user_prompt)
            #print("{}: {}".format(jdsum['role'], jdsum['content']))
            #delete_table_contents()
            results=read_chroma_db(jd,30)
            file_values = [meta['source'] for meta in results['metadatas'][0]]
            match_values = results['distances'][0]
            documents=results['documents'][0]
            system_prompt="Eres un asistente util"
            


            # Creamos el DataFrame
            df_sorted = pd.DataFrame({
            'Filename': file_values,
            'documents':documents,
            'MatchValue': match_values
            })
            df_sorted = df_sorted.sort_values(by='MatchValue', ascending=True) # o ascending=False si lo quieres en orden descendente
            st.write(df_sorted)
            #print("entrando a modulo chatgpt")
            #combined_string = df_sorted.apply(lambda row: f"Nombre archivo Curriculum vitae {row['Filename']} contenido del curriculum {row['documents']}", axis=1).str.cat(sep=' ')
            #user_prompt=f"Quiero que ordenes los 5 curriculums vitae del contexto segun el que mas coincida con esta descripcion de trabajo: {jd}. Los Curriculumns son: {combined_string} da una breve explicacion de como se selecciono y el nombre del archivo pdf. Si no hay ninguna coincidencia di: No hay ningun candidato que cumpla los requisitos solicitados"
            #cv_selected=chat_gpt_action(system_prompt,user_prompt)
            #print(f"el cv seleccionado es {cv_selected}")
            #user_prompt=f"quiero que extraigas el primer nombre de archivo con extension pdf del siguiente texto: {cv_selected}. Solo indica el nombre de archivo en tu respuesta. Por ejemplo javier.pdf"
            #File=chat_gpt_action(system_prompt,user_prompt)
            store_to_sqlite(df_sorted)
        else:
            st.write("Please enter a job description to process.")
    
    col1, col2 = st.columns([3, 1])
    df_sorted_from_db = read_from_sqlite()
    if not df_sorted_from_db.empty:
        selected_pdf = st.selectbox('Elige un PDF:', df_sorted_from_db['Filename'].tolist())
        if jd:
            pdf_url = f"http://143.198.139.51:8081/CV/{selected_pdf}"
            skills=obtain_skills(f"./CV/{selected_pdf}")
            st.text("Resultado")
            #st.markdown(str(cv_selected['content']))
            st.markdown(f'<iframe src="{pdf_url}" width="700" height="900"></iframe>', unsafe_allow_html=True)
            s=""
            for i in skills:
                s += "- " + i + "\n"
            st.markdown(s)
            df_sorted_from_db['MatchValue']=((1-(df_sorted_from_db['MatchValue']/df_sorted_from_db['MatchValue'].max()))/(1-(df_sorted_from_db['MatchValue']/df_sorted_from_db['MatchValue'].max())).max())*100
            fig = px.bar(df_sorted_from_db, x='Filename', y='MatchValue', title='Match Values by Filename')

        # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig)
  

if __name__ == "__main__":
    main()


