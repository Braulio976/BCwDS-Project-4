import gradio as gr
from openai import AzureOpenAI
import os
import pickle
import json

from utils import (
    create_assistant,
    create_thread,
    check_assistant_exists,
    load_and_upload_files,
    send_message_to_assistant
)

# --- Initial Configuration ---
API_KEY = 'EPg5Wj22q1CfEPQaVw3L6kVk5NVPSFOjKpNfC9mNLGr5rH5vFefFJQQJ99BDACYeBjFXJ3w3AAABACOGZ80v'
ENDPOINT = 'https://ai-bcds.openai.azure.com/'
ASSISTANT_FILENAME = 'AssistantID.TXT'
VECTOR_DATA_PATH = 'vector_store.pkl'

# --- Initialize Azure OpenAI client ---
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version="2024-05-01-preview"
)

# --- Load documents links and prompt rules ---
with open("document_links.json", "r", encoding="utf-8") as f:
    doc_link_map = json.load(f)

with open("prompt_rules.txt", "r", encoding="utf-8") as f:
    prompt_rules = f.read()

# --- Load or create vector store ---
if os.path.exists(VECTOR_DATA_PATH):
    with open(VECTOR_DATA_PATH, "rb") as file:
        vector_store = pickle.load(file)
else:
    vector_store = load_and_upload_files(client, link_map=doc_link_map)

# --- Load or create assistant ---
if os.path.exists(ASSISTANT_FILENAME):
    with open(ASSISTANT_FILENAME, "r") as file:
        assistant_id = file.read().strip()
    exists, assistant = check_assistant_exists(client, assistant_id)
    if exists:
        # Update assistant with latest instructions and vector store reference
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions=prompt_rules,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )
    else:
        assistant = create_assistant(client, prompt_rules, ASSISTANT_FILENAME)
else:
    assistant = create_assistant(client, prompt_rules, ASSISTANT_FILENAME)

# --- Create conversation thread ---
thread = create_thread(client)
displayedMessagesIDs = []

# --- Chatbot interface function ---
def chatbot_interface(user_input, history):
    response = send_message_to_assistant(
        client,
        thread,
        assistant,
        user_input,
        displayedMessagesIDs
    )
    history.append((user_input, response))
    return history, history

# --- Gradio UI setup ---
with gr.Blocks() as demo:
    gr.Markdown("🛡️ **Fidelidade Virtual Assistant**")
    chatbot = gr.Chatbot(type="messages")  # Set to 'messages' to avoid deprecation warning
    input_box = gr.Textbox(label="Type your question here...")

    state = gr.State([])

    input_box.submit(chatbot_interface, inputs=[input_box, state], outputs=[chatbot, state])
    input_box.submit(lambda: "", None, input_box)

demo.launch()

#python assistente_fidelidade.py
