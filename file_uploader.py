__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
from chromadb import PersistentClient
from chromadb.config import Settings
from openai import OpenAI
import tiktoken
import chardet

def read_file_content(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    detected = chardet.detect(raw_data)
    encoding = detected['encoding']

    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        return f"Unable to read file content. The file may be binary or use an unsupported encoding: {encoding}"

def get_embedding(text, model="text-embedding-ada-002"):
    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def upload_and_embed_file(file):
    os.makedirs("project/docs", exist_ok=True)
    db_path = os.path.join(os.getcwd(), "db")
    os.makedirs(db_path, exist_ok=True)

    destination = os.path.join("project/docs", os.path.basename(file.name))
    shutil.copy(file.name, destination)

    content = read_file_content(destination)

    chroma_client = PersistentClient(path=db_path, settings=Settings(allow_reset=True))

    collection_name = os.path.splitext(os.path.basename(file.name))[0]
    
    # Get or create the collection
    try:
        collection = chroma_client.get_collection(name=collection_name)
        # If collection exists, we'll overwrite it
        collection.delete(where={'collection_name': collection_name})
    except ValueError:
        # Collection doesn't exist, create a new one
        collection = chroma_client.create_collection(name=collection_name)

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    max_tokens = 8000
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for line in content.split('\n'):
        line_tokens = len(encoding.encode(line))
        if current_tokens + line_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = line
            current_tokens = line_tokens
        else:
            current_chunk += line + '\n'
            current_tokens += line_tokens

    if current_chunk:
        chunks.append(current_chunk)

    for i, chunk in enumerate(chunks):
        chunk_embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[chunk_embedding],
            ids=[f"{collection_name}_{i}"]
        )

    return collection_name