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


# Function to display messages not displayed yet
def display_messages(client, thread, message, displayedMessagesIDs):
    messages = client.beta.threads.messages.list(thread_id=thread.id, order='asc', after=message.id)
    if messages:
        for message in messages:
            if message.id not in displayedMessagesIDs:
                
                # Check if the message is from the assistant and print it
                if message.role == 'assistant':
                    # Load JSON message into a Python object
                    answer = json.loads(message.model_dump_json(indent=2))
                    # Show answer
                    try:
                        messageType = message.content[0].type
                        # Add message id  to the list as displayed
                        displayedMessagesIDs.append(message.id)
                        
                        # Text message
                        if  messageType=='text':
                            content = 'Assistant: ' + answer['content'][0]['text']['value']
                            content = re.sub(r"【.*?】", "", content)

                            
                            # Check if there are links for file downloads
                            file_link = None
                            if 'annotations' in answer['content'][0]['text']:
                                for annotation in answer['content'][0]['text']['annotations']:
                                    if annotation['type'] == 'file_path':
                                        file_link = annotation['text']
                                        file_id = annotation['file_path'].get('file_id')
                                        start_index = annotation.get('start_index')
                                        end_index = annotation.get('end_index')
                                        # Remove the link from the value if start_index and end_index are present
                                        if start_index is not None and end_index is not None:
                                            content = content[:start_index] + content[end_index:]
                                        # Download the file
                                        fileName = file_download(file_id, thread.id, message.id, file_link=file_link,is_image=False)
                            
                            # Display as Markdown
                            display(Markdown(content))
                        # Image
                        elif messageType == 'image_file':
                            # Get the ID of the image
                            fileID = answer['content'][0]['image_file']['file_id']
                            
                            # Download the image
                            fileName = file_download(fileID, thread.id, message.id, file_link='', is_image=True)
                            
                            # Display the image in the default image viewer
                            image = Image.open(fileName)
                            image.show()
                    except:
                        continue


def send_message_to_assistant(client, thread, assistant, user_input, displayedMessagesIDs):
    import time

    print("You:", user_input)
    print("Thinking...")

    # Adiciona a mensagem do user (não precisa das regras no prompt)
    message = add_message_to_thread(client, thread.id, user_input)

    # Inicia o run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread.id).data
        assistant_msgs = [m for m in messages if m.role == "assistant"]

        if assistant_msgs:
            last_msg = assistant_msgs[-1]

            full_text = ""
            for block in last_msg.content:
                if block.type == "text":
                    full_text += block.text.value + "\n"
            return full_text.strip()

    elif run.status == 'requires_action':
        return "O assistente precisa de ações adicionais."
    else:
        return f"Ocorreu um erro. Estado: {run.status}"

