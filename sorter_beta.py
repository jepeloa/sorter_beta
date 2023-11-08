from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from termcolor import colored
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re
import plotly.graph_objects as go
import nltk
#nltk.download('punkt')
import os
import streamlit as st
import sqlite3
import http.server
from PIL import Image
import socketserver
import threading
import plotly.express as px


class person_cv():
      def __init__(self, nombre, pdf_path, email, telefono):
        self.nombre = nombre
        self.pdf_path = pdf_path  
        self.email = email
        self.telefono = telefono
    





image = Image.open('logo.png')

st.sidebar.image(image, caption=' ', width=200)

uploaded_files=st.sidebar.file_uploader("Upload CVs", accept_multiple_files=True, type="pdf", key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="collapsed")


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


def start_pdf_server(port=8081):
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    print(f"PDF server started at port {port}")


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
                st.sidebar.success(f"Archivo {filename} eliminado con éxito")
        except Exception as e:
            st.error(f"Error al eliminar el archivo {filename}: {e}")



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
        return f"Error al borrar el contenido de la tabla: {e}"
    

##################################################################

progress_bar = st.progress(0)





###################################################################


def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)
    
    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def evaluate_candidate(input_CV,input_JD,model):
    v1 = model.infer_vector(input_CV.split())
    v2 = model.infer_vector(input_JD.split())
    similarity = 100*((np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2))))
    return round(similarity, 2)


def process_JD_and_get_matches(jd,model,path_to_folder):
    all_files = os.listdir(path_to_folder)
    df = pd.DataFrame(columns=['Filename', 'MatchValue', 'Skills'])
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
                text_to_translate=pageObj.extract_text()

                resume+=text_to_translate
        input_CV = preprocess_text(resume)
        input_JD = preprocess_text(jd)
        match = evaluate_candidate(input_CV, input_JD, model)
        df.loc[j] = {'Filename': pdf_file, 'MatchValue': match}
        j=j+1
        progress_bar.progress(int(j*100/len(all_files)),text=f"processing: {pdf_file}")
        print("="*50)
    df['MatchValue'] = df['MatchValue'].astype(float)
    df_sorted = df.sort_values(by='MatchValue', ascending=False)
    return df_sorted






def main():
    path_to_folder='./CV/'
    selected_pdf=pd.DataFrame()
    init_db()
    try:
        start_pdf_server()
    except:
        print("No se pudo inciar el servidor pdf")
    

    df_sorted=pd.DataFrame
    model = Doc2Vec.load('cv_job_maching.model')
    st.title("CV Sorter")
    st.write("Insert the job description and get the matching CVs.")

    # Text area for the user to input the job description
    jd = st.text_area("Job Description", "")
    if st.button("Process"):
        if jd:
            df_sorted = process_JD_and_get_matches(jd,model,path_to_folder)
            store_to_sqlite(df_sorted)
        else:
            st.write("Please enter a job description to process.")
    if st.button('Borrar contenido de la tabla'):
        message = delete_table_contents()
        st.write(message)

    df_sorted_from_db = read_from_sqlite()
    if not df_sorted_from_db.empty:
        selected_pdf = st.selectbox('Elige un PDF:', df_sorted_from_db['Filename'].tolist())
        pdf_url = f"http://localhost:8081/CV/{selected_pdf}"
        st.markdown(f'<iframe src="{pdf_url}" width="700" height="900"></iframe>', unsafe_allow_html=True)
        fig = px.scatter(df_sorted_from_db, x="Filename", y="MatchValue", title="Match Values por Filename", height=1000)
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig)
        st.write(df_sorted_from_db)
if __name__ == "__main__":
    main()


