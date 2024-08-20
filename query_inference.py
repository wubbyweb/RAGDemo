import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb import Client
from chromadb.config import Settings
import requests
from openai import embeddings
import chromadb.utils.embedding_functions as embedding_functions

def infer_query(collection_name, query):
    # Retrieve context from ChromaDB
    db_path = os.path.join(os.getcwd(), "db/")
    os.makedirs(db_path, exist_ok=True)
    #chroma_client = Client(database=db_path)
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(),
        tenant="default_tenant",
        database="default_database"
    )


    api_key = os.getenv("PPLEX_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable is not set")
    

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-ada-002"
            )

    collection = chroma_client.get_collection(name=collection_name,embedding_function=openai_ef)
    results = collection.query(query_texts=[query], n_results=5)
    
    context = " ".join(results['documents'][0])

    # Use Perplexity API for inference
    api_key = os.getenv("PPLEX_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-sonar-small-128k-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's query."},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}\n\nPlease provide an answer based on the given context:"}
        ]
    }

    response = requests.post("https://api.perplexity.ai/chat/completions", json=data, headers=headers)
    
    if response.status_code == 200:
        return_response = [
                ["You: ",query],
                ["System: ",response.json()['choices'][0]['message']['content']]
            ]
        return return_response
    
    else:
        return f"Error: Unable to get response from Perplexity API. Status code: {response.status_code}"
    

if __name__ == "__main__":
    print(infer_query('default','test'))