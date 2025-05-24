import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
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

# --- Azure OpenAI config ---
API_KEY = 'EPg5Wj22q1CfEPQaVw3L6kVk5NVPSFOjKpNfC9mNLGr5rH5vFefFJQQJ99BDACYeBjFXJ3w3AAABACOGZ80v'
ENDPOINT = 'https://ai-bcds.openai.azure.com/'
ASSISTANT_FILENAME = 'AssistantID.TXT'
VECTOR_DATA_PATH = 'vector_store.pkl'

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version="2024-05-01-preview"
)

# --- Load documents & prompt ---
with open("document_links.json", "r", encoding="utf-8") as f:
    doc_link_map = json.load(f)

with open("prompt_rules.txt", "r", encoding="utf-8") as f:
    prompt_rules = f.read()

# --- Load vector store ---
if os.path.exists(VECTOR_DATA_PATH):
    with open(VECTOR_DATA_PATH, "rb") as file:
        vector_store = pickle.load(file)
else:
    vector_store = load_and_upload_files(client, link_map=doc_link_map)

# --- Assistant creation ---
if os.path.exists(ASSISTANT_FILENAME):
    with open(ASSISTANT_FILENAME, "r") as file:
        assistant_id = file.read().strip()
    exists, assistant = check_assistant_exists(client, assistant_id)
    if exists:
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions=prompt_rules,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )
    else:
        assistant = create_assistant(client, prompt_rules, ASSISTANT_FILENAME)
else:
    assistant = create_assistant(client, prompt_rules, ASSISTANT_FILENAME)

thread = create_thread(client)
displayedMessagesIDs = []

# --- Tkinter UI ---
def send_message():
    user_input = entry.get()
    if not user_input.strip():
        return

    chat_window.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    response = send_message_to_assistant(client, thread, assistant, user_input, displayedMessagesIDs)
    chat_window.insert(tk.END, f"Bot: {response}\n")
    chat_window.yview(tk.END)

# Main window
root = tk.Tk()
root.title("Fidelidade Virtual Assistant")
root.geometry("600x700")

# Background image
background_image = Image.open("fidelidade_logo.png")
background_image = background_image.resize((600, 700), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(background_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# Chat display
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="white", font=("Helvetica", 12))
chat_window.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.75)
chat_window.config(state=tk.NORMAL)

# Input box
entry = tk.Entry(root, font=("Helvetica", 12))
entry.place(relx=0.05, rely=0.83, relwidth=0.7, relheight=0.06)

# Send button
send_button = tk.Button(root, text="Send", font=("Helvetica", 12), command=send_message)
send_button.place(relx=0.77, rely=0.83, relwidth=0.18, relheight=0.06)

root.mainloop()
