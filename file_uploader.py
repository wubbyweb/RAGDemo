__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import PyPDF2
from dotenv import load_dotenv
import openai


def read_file_content(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            text = "" 
            
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            return "OK",text
    
    except (Exception) as e:
        return "ERROR",e
    
def split_text(text,parm_chunk_size,parm_overlap_size):
    #loader = TextLoader(text,encoding='utf8')
    #pages = loader.load()



    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parm_chunk_size,
        chunk_overlap=parm_overlap_size, 
        separators=["\n\n", "\n", " "]
    )

    docs = text_splitter.split_text(text=text)

    return docs 


def upload_and_embed_file(file):
    os.makedirs("docs", exist_ok=True)
    db_path = os.path.join(os.getcwd(), "db")
    os.makedirs(db_path, exist_ok=True)

    destination = os.path.join("docs", os.path.basename(file.name))
    shutil.copy(file.name, destination)

    content = read_file_content(destination)

    if content[0] == "OK":
        splitted_text = split_text(content[1],1500,150)
    else:   
        return "ERROR"
    
    try:
        persist_directory = 'db/'
        collection_name = os.path.splitext(os.path.basename(file.name))[0]

        embedding = OpenAIEmbeddings()

        vectordb = Chroma.from_texts(
            texts=splitted_text,
            embedding=embedding,
            persist_directory=persist_directory,
            collection_name=collection_name
        )

    

        vectordb.persist()

        return collection_name
    
    except (Exception) as e:
        print(e)

if __name__ == "__main__":
    with open('db/Medmal.pdf','r') as file:
        upload_and_embed_file(file)