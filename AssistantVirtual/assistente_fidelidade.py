import gradio as gr
from openai import AzureOpenAI
from utils import (
    create_assistant,
    create_thread,
    check_assistant_exists,
    load_and_upload_files,
    send_message_to_assistant
)
import os
import pickle
import json

# Configurações iniciais
api_key = 'yourAPIKey'
endpoint = 'https://ai-bcds.openai.azure.com/'
assistantFilename = 'AssistantID.TXT'
vector_data = 'vector_store.pkl'

# Cliente Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-05-01-preview"
)

# Load links e prompt
with open("document_links.json", "r", encoding="utf-8") as f:
    doc_link_map = json.load(f)
with open("prompt_rules.txt", "r", encoding="utf-8") as f:
    prompt_rules = f.read()

if os.path.exists(vector_data):
    with open(vector_data, "rb") as file:
        vector_store = pickle.load(file)
    print(f"Loaded vector_store with ID: {vector_store.id}")
else:
    vector_store = load_and_upload_files(client, link_map=doc_link_map)
    print(f"Created vector_store with ID: {vector_store.id}")


# Assistente
if os.path.exists(assistantFilename):
    with open(assistantFilename, "r") as file:
        assistant_id = file.read().strip()
    exists, assistant = check_assistant_exists(client, assistant_id)
    if exists:
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions="És um assistente virtual da seguradora Fidelidade...",
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )
        print(f"Assistant updated with ID: {assistant.id} linked to vector_store ID: {vector_store.id}")
    else:
        assistant = create_assistant(client, "És um assistente virtual da seguradora Fidelidade...", assistantFilename, vector_store)
        print(f"Assistant created with ID: {assistant.id} linked to vector_store ID: {vector_store.id}")
else:
    assistant = create_assistant(client, "És um assistente virtual da seguradora Fidelidade...", assistantFilename, vector_store)
    print(f"Assistant created with ID: {assistant.id} linked to vector_store ID: {vector_store.id}")


# Sessão de conversa
thread = create_thread(client)
displayedMessagesIDs = []

# Lógica do chatbot
def chatbot_interface(user_input, history):
    print("Input do usuário recebido no Gradio:", user_input)
    response = send_message_to_assistant(
        client,
        thread,
        assistant,
        user_input,
        prompt_rules,
        displayedMessagesIDs
    )
    print("Resposta obtida para o Gradio:", response)
    history.append((user_input, response))
    return history, history


# Interface do Gradio
with gr.Blocks() as demo:
    gr.Markdown("🛡️ **Assistente Virtual Fidelidade**")
    chatbot = gr.Chatbot()
    input_box = gr.Textbox(label="Escreva aqui a sua pergunta...")

    state = gr.State([])

    input_box.submit(chatbot_interface, inputs=[input_box, state], outputs=[chatbot, state])
    input_box.submit(lambda: "", None, input_box)

demo.launch()


#Para correr: 
#python assistente_fidelidade.py
