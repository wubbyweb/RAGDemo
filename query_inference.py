import os
from chromadb import PersistentClient
from chromadb.config import Settings
from openai import OpenAI

def get_embedding(text, model="text-embedding-ada-002"):
    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def infer_query(collection_name, query):
    db_path = os.path.join(os.getcwd(), "db")
    chroma_client = PersistentClient(path=db_path, settings=Settings(allow_reset=True))
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except ValueError:
        return f"Error: Collection '{collection_name}' does not exist. Please upload a file first."

    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    if not results['documents']:
        return "No relevant information found in the document."

    context = " ".join(results['documents'][0])

    openai_client = OpenAI()
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's query."},
        {"role": "user", "content": f"Context: {context}\n\nQuery: {query}\n\nPlease provide an answer based on the given context:"}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    print(response.choices[0].message.content)

    return response.choices[0].message.content