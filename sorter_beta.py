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
#client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    #persist_directory="db/"
                                #))


client = chromadb.Client()


    



image = Image.open('logo.png')

st.sidebar.image(image, caption=' ', width=200)

uploaded_files=st.sidebar.file_uploader("Upload CVs", accept_multiple_files=True, type="pdf", key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="collapsed")


####################################################
def store_to_sqlite(df):
    conn = sqlite3.connect('pdf_database.db')
    #df.to_sql('pdf_list', conn, if_exists='replace', index=False)  # Guarda el dataframe en la tabla 'pdf_list'
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


def start_pdf_server(port=8081):
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    print(f"PDF server started at port {port}")

def read_chroma_db(query,quantity=1):
    cv_collection = client.get_collection(name="cv_collection")
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
        st.sidebar.success(f"Archivo {uploaded_file.name} guardado con éxito")


def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
               # st.sidebar.success(f"Archivo {filename} eliminado con éxito")
        except Exception as e:
            #st.error(f"Error al eliminar el archivo {filename}: {e}")
            pass



if uploaded_files:
        if st.sidebar.button('Guardar PDFs'):
            save_uploaded_files(uploaded_files)


if st.sidebar.button('Borrar CVs'):
        delete_files_in_directory('./CV')



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
        pass
        #return f"Error al borrar el contenido de la tabla: {e}"
    

##################################################################

progress_bar = st.progress(0)





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
    cv_collection = client.create_collection("cv_collection")

    # add files to the chromadb collection
    cv_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    return cv_collection




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
        file_data.append({"file_name": pdf_file, "content": resume})
        
        j=j+1
        progress_bar.progress(int(j*100/len(all_files)),text=f"processing: {pdf_file}")
        print("="*50)
    return file_data






def main():
    path_to_folder='./CV/'
    selected_pdf=pd.DataFrame()
    init_db()
    try:
        start_pdf_server()
    except:
        print("No se pudo inciar el servidor pdf")
    

    df_sorted=pd.DataFrame
    #model = Doc2Vec.load('cv_job_maching.model')
    st.title("CV Sorter")
    st.write("Insert the job description and get the matching CVs.")

    # Text area for the user to input the job description
    jd = st.text_area("Job Description", "")
    if st.button("Process cv"):
        file_data = read_CV_from_pdf(path_to_folder)  #extraigo datos de los pdf
        cv_collection=store_CV_in_db(file_data)
        
            
    if st.button('Borrar contenido de la tabla'):
        message = delete_table_contents()
        st.write(message)
    if st.button('procesar query'):
        if jd:
            results=read_chroma_db(jd,5)
            file_values = [meta['source'] for meta in results['metadatas'][0]]
            match_values = results['distances'][0]

            # Creamos el DataFrame
            df_sorted = pd.DataFrame({
            'Filename': file_values,
            'MatchValue': match_values
            })
            store_to_sqlite(df_sorted)
            st.write(df_sorted)
        else:
            st.write("Please enter a job description to process.")


    df_sorted_from_db = read_from_sqlite()
    st.write(df_sorted_from_db)
    df_sorted_from_db=pd.DataFrame()
    if not df_sorted_from_db.empty:
        selected_pdf = st.selectbox('Elige un PDF:', df_sorted_from_db['Filename'].tolist())
        pdf_url = f"http://143.198.139.51/:8081/CV/{selected_pdf}"
        st.markdown(f'<iframe src="{pdf_url}" width="700" height="900"></iframe>', unsafe_allow_html=True)
        fig = px.scatter(df_sorted_from_db, x="Filename", y="MatchValue", title="Match Values por Filename", height=1000)
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig)
        st.write(df_sorted_from_db)
if __name__ == "__main__":
    main()


