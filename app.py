import gradio as gr
import os
from file_uploader import upload_and_embed_file
from query_inference import infer_query

def process_upload(file):
    if file is not None:
        file_name = os.path.basename(file.name)
        collection_name = upload_and_embed_file(file)
        return f"File uploaded: {file_name}", collection_name
    return "No file uploaded", ""

def process_query(collection_name, query):
    if collection_name and query:
        response = infer_query(collection_name, query)
        return response
    return "Please provide both collection name and query."

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            file_upload = gr.File(label="Upload File")
            file_info = gr.Textbox(label="File Info")
        with gr.Column(scale=8):
            collection_name = gr.Textbox(label="Collection Name")
            query_input = gr.Textbox(label="Enter your query")
            submit_btn = gr.Button("Submit")
            chat_output = gr.Chatbot()

    file_upload.upload(process_upload, inputs=[file_upload], outputs=[file_info, collection_name])
    submit_btn.click(process_query, inputs=[collection_name, query_input], outputs=[chat_output])

if __name__ == "__main__":
    demo.launch()