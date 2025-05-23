# Imports
import os
import re
import json
import time
import pandas as pd
from openai import AzureOpenAI
from PIL import Image
from IPython.display import Markdown, display
import pickle
from datetime import datetime, timedelta
import fitz
from docx import Document  

# Function to create the assistant with code interpreter capability
def create_assistant(client, aRole, assistantFilename):
    assistant = client.beta.assistants.create(
        name="Agent",
        instructions=aRole,
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}]
    )
    assistant_id = assistant.id
    
    # Save the ID to a file
    with open(assistantFilename, "w") as file:
        file.write(assistant_id)
    
    return assistant

# Function to create a new thread for a conversation
def create_thread(client):
    thread = client.beta.threads.create()
    displayedMessagesIDs = []
    return thread

# Function to check if the assistant exists
def check_assistant_exists(client, assistant_id):
    try:
        response = client.beta.assistants.retrieve(assistant_id)       
        return True if response else False, response
    except Exception as e:
        print(f"An error occurred while checking the assistant: {e}", assistant_id)
        return False, None


# Extract text from docx document
def extract_text_from_docx(path, link_map=None):
    from docx import Document
    doc = Document(path)
    text = '\n'.join([p.text for p in doc.paragraphs])
    filename = os.path.splitext(os.path.basename(path))[0] # Take name of file without .docx (for example QandA.docx --> QandA)

    # Classify documents in formats a1,a2,b1,c1 etc as documents with informations from concurrents, rest as fidelidade documents
    def is_fidelidade_document(name):
        import re
        if re.match(r"^[a-zA-Z]\d+$", name):
            return name[0].lower() in ["n", "o"]
        return True

    categoria = "Fidelidade" if is_fidelidade_document(filename) else "Concorrente"

    #     IMPORTANT SO THE MODEL KNOW THE NAMES OF EACH DOCUMENT USED
    header = (f"Document Name: {filename}\nCategoria: {categoria}\n") * 50 # Add this name 50 times at beginning of text of document (so it surely appears in batch used by model)
    link = link_map.get(filename) if link_map else None   # Get link from linkmap
    link_block = (f"Fonte: {link}\n") * 50 if link else "" # Add this link 50 times at beginning after the 50x names, so model surely has acess to link

    return f"{header}\n{text}\n{link_block}"

    
# Extract text from pdf document
def extract_text_from_pdf(path, link_map=None):
    doc = fitz.open(path)
    text = '\n'.join([page.get_text().strip() for page in doc])
    filename = os.path.splitext(os.path.basename(path))[0]  # Take name of file without .pdf (for example a1.pdf --> a1)

    # Classify documents in formats a1,a2,b1,c1 etc as documents with informations from concurrents, rest as fidelidade documents
    def is_fidelidade_document(name):
        import re
        if re.match(r"^[a-zA-Z]\d+$", name):
            return name[0].lower() in ["n", "o", "y"]
        return True  # Alle anderen Formate gelten als Fidelidade

    categoria = "Fidelidade" if is_fidelidade_document(filename) else "Concorrente"

    #     #IMPORTANT SO THE MODEL KNOW THE NAMES OF EACH DOCUMENT USED
    header = (f"Document Name: {filename}\nCategoria: {categoria}\n") * 50 # Add this name 50 times at beginning of text of document (so it surely appears in batch used by model)

    link = link_map.get(filename) if link_map else None   # Get link from linkmap
    link_block = (f"Fonte: {link}\n") * 50 if link else ""  # Add this link 50 times at beginning after the 50x names, so model surely has acess to link

    return f"{header}\n{text}\n{link_block}"

# Recursively find all pdf and docx documents in folder 'documents' (searching through all sub-folders)
def find_documents(folder="documents"):
    pdf_files = []
    docx_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith("~$"):  # Naturally generated temporary files (ignore)
                continue
            if file.lower().endswith(".pdf"): # If pdf, append to list with pdf documents to be loaded
                pdf_files.append(os.path.join(root, file))
            elif file.lower().endswith(".docx"):   # If docx, append to list with docx documents to be loaded
                docx_files.append(os.path.join(root, file))
    return pdf_files, docx_files



# Main function to load and upload files
def load_and_upload_files(client, link_map=None):
    print("Searching './documents' for PDF and DOCX files...")
    pdf_files, docx_files = find_documents("documents")
    print(f"Found {len(pdf_files)} PDFs and {len(docx_files)} DOCX files.")

    if not pdf_files and not docx_files:
        print("No documents found.")
        return None

    vector_store = client.vector_stores.create(name="Documents Vector Store")
    print("Vector store created:", vector_store)

    for path in pdf_files:
        text = extract_text_from_pdf(path, link_map)
        temp_txt_path = path + ".txt"
        with open(temp_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        with open(temp_txt_path, "rb") as f:
            client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=[f]
            )
        os.remove(temp_txt_path)
        print("Uploaded PDF:", path)

    for path in docx_files:
        text = extract_text_from_docx(path, link_map)
        temp_txt_path = path + ".txt"
        with open(temp_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        with open(temp_txt_path, "rb") as f:
            client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=[f]
            )
        os.remove(temp_txt_path)
        print("Uploaded DOCX:", path)

    with open("vector_store.pkl", "wb") as file:
        pickle.dump(vector_store, file)
    print("Vector store saved.")

    return vector_store


# Function to add a message to the thread
def add_message_to_thread(client, thread_id, user_message):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    return message



# display INTERFACE
def display_messages(client, thread, message, displayedMessagesIDs):
    messages = client.beta.threads.messages.list(thread_id=thread.id, order='asc', after=message.id)
    collected_outputs = []  # NEU

    if messages:
        for message in messages:
            if message.id not in displayedMessagesIDs and message.role == 'assistant':
                try:
                    messageType = message.content[0].type
                    displayedMessagesIDs.append(message.id)

                    if messageType == 'text':
                        content = message.content[0].text.value
                        content = re.sub(r"【.*?】", "", content)

                        # Datei-Links entfernen, wie gehabt (optional)
                        if hasattr(message.content[0].text, 'annotations'):
                            for annotation in message.content[0].text.annotations:
                                if annotation.type == 'file_path':
                                    start = annotation.start_index
                                    end = annotation.end_index
                                    if start is not None and end is not None:
                                        content = content[:start] + content[end:]
                                    # Optional: file_download(...)

                        collected_outputs.append(content)

                    elif messageType == 'image_file':
                        # Bilddateien kannst du später bei Bedarf einbauen
                        collected_outputs.append("[Bild empfangen]")

                except Exception as e:
                    collected_outputs.append(f"[Fehler bei Nachricht: {e}]")

    return "\n\n".join(collected_outputs)


# SEND INTERFACE
def send_message_to_assistant(client, thread, assistant, user_input, full_prompt, displayedMessagesIDs):
    print("You:", user_input)
    print("Thinking...")

    message = add_message_to_thread(client, thread.id, full_prompt)

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        # Keine Anzeige hier – wir warten auf finalen Abschluss

    if run.status == 'completed':
        return display_messages(client, thread, message, displayedMessagesIDs)
    elif run.status == 'requires_action':
        return "Der Assistant benötigt zusätzliche Aktionen."
    else:
        return f"Status: {run.status}"
