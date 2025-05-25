# Imports
import streamlit as st
import os
import json
import pickle
import re
from openai import AzureOpenAI
import time
import base64

from utils import (
    create_assistant,
    create_thread,
    check_assistant_exists,
    load_and_upload_files,
    send_message_to_assistant
)

# Configurations for Azure OpenAI
api_key = 'yourAPIKey'
endpoint = 'https://ai-bcds.openai.azure.com/'

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-05-01-preview"
)

# Load document links
with open("document_links.json", "r", encoding="utf-8") as f:
    doc_link_map = json.load(f)

vector_data = 'vector_store.pkl'

if os.path.exists(vector_data):
    with open(vector_data, "rb") as file:
        vector_store = pickle.load(file)
else:
    vector_store = load_and_upload_files(client, link_map=doc_link_map)

# Assistant role description
aRole = (
    "És um assistente virtual da seguradora Fidelidade, fiável e rápido, que apoia os colaboradores durante os atendimentos a clientes.\n"
    "O teu objetivo é fornecer respostas claras, corretas e rápidas, ajudando os colaboradores da Fidelidade a responder com confiança.\n"
    "Entendes o contexto da conversa e considera sempre as perguntas anteriores para manter coerência nas respostas.\n"
    "Se o colaborador fizer uma pergunta de seguimento, lembre-se do que foi dito antes.\n"
    "Não interages diretamente com o cliente final, mas atua como um suporte eficiente para os colaboradores.\n"
    "Responde de forma natural, amigável e clara."
)

assistantFilename = 'AssistantID.TXT'
assistant_id = None
assistant = None

if os.path.exists(assistantFilename):
    with open(assistantFilename, "r") as file:
        assistant_id = file.read().strip()

    exists, assistant = check_assistant_exists(client, assistant_id)
    if exists:
        assistant = client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions=aRole,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )
    else:
        assistant_id = None

if assistant_id is None:
    assistant = create_assistant(client, aRole, assistantFilename)
    assistant_id = assistant.id
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )

thread = create_thread(client)

with open("prompt_rules.txt", "r", encoding="utf-8") as f:
    prompt_rules = f.read()

displayedMessagesIDs = []

# Light/Dark Mode
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

toggle = st.sidebar.checkbox("🌙 Modo Escuro", value=st.session_state.dark_mode)
st.session_state.dark_mode = toggle

# Styling
bg_color = "#1e1e1e" if st.session_state.dark_mode else "#f5f5f5"
text_color = "#f9f9f9" if st.session_state.dark_mode else "#333333"
user_bg = "#8B0000" if st.session_state.dark_mode else "#d80027" 
assistant_bg = "#2e2e2e" if st.session_state.dark_mode else "#ffffff"

st.markdown(
    f"""
    <style>
    body, .block-container {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: {bg_color};
        color: {text_color};
    }}

    .title {{
        text-align: center;
        color: {user_bg};
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 25px;
    }}

    .chat-container {{
        max-width: 700px;
        margin: 0 auto 30px auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}

    .user-message, .assistant-message {{
        padding: 12px 18px;
        border-radius: 15px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        font-size: 16px;
        animation: fadeIn 0.3s ease-in;
    }}

    .user-message {{
        align-self: flex-end;
        background-color: {user_bg};
        color: white;
        border-radius: 15px 15px 0 15px;
        box-shadow: 0 4px 12px rgba(216, 0, 39, 0.6);
    }}

    .assistant-message {{
        align-self: flex-start;
        background-color: {assistant_bg};
        color: {text_color};
        border-radius: 15px 15px 15px 0;
        border: 1.5px solid {user_bg};
    }}

    input[type="text"] {{
        flex-grow: 1;
        padding: 12px 15px;
        border: 2px solid {user_bg};
        border-radius: 10px;
        font-size: 18px;
        outline: none;
        color: {text_color};
        background-color: transparent;
    }}

    input[type="text"]::placeholder {{
        color: {text_color};
        opacity: 0.7;
    }}

    input[type="text"]:focus {{
        border-color: #8B0000;
        box-shadow: 0 0 5px #8B0000;
        background-color: transparent;
        color: {text_color};
    }}

    .send-button {{
        background-color: {user_bg};
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }}

    .send-button:hover {{
        background-color: #8B0000;
    }}

        button[role="button"] {{
        background-color: transparent !important;
        color: {user_bg} !important;
        border: 2px solid {user_bg} !important;
        border-radius: 10px !important;
        padding: 8px 15px !important;
        font-weight: 600 !important;
        margin: 5px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }}

    button[role="button"]:hover {{
        background-color: {user_bg} !important;
        color: white !important;
    }}

    div.stButton > button {{
        height: 100% !important;
        margin-top: auto !important;
        margin-bottom: auto !important;
        padding-top: 12px !important;
        padding-bottom: 12px !important;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @media (max-width: 768px) {{
        .chat-container {{
            max-width: 95%;
        }}

        input[type="text"], .send-button {{
            width: 100%;
        }}
    }}

    .sidebar .sidebar-content {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    .sidebar-title {{
        font-weight: 700;
        color: {user_bg};
        font-size: 22px;
        margin-bottom: 15px;
    }}

    .sidebar-text {{
        font-size: 16px;
        line-height: 1.5;
    }}
    .fixed-bottom-right {{
        position: fixed;
        bottom: 10px;
        right: 10px;
        width: 120px;
        z-index: 9999;
        opacity: 0.9;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="title">🤖 Assistente Fidelidade</div>', unsafe_allow_html=True)

# Start a new conversation
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to handle chatbot response
def chatbot_response(user_input):
    full_prompt = f"{prompt_rules}\n\nUser question: {user_input}"
    response = send_message_to_assistant(client, thread, assistant, user_input, full_prompt, displayedMessagesIDs)

    # Optional
    if "Fonte:" in response:
        response = re.sub(r'Fonte:.*', 'Fonte: https://www.fidelidade.pt', response)

    return response

# Add message to session state
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Show chat messages
def display_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, msg in enumerate(reversed(st.session_state.messages)):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message"><strong>🧑‍💻 Você:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-message"><strong>🤖 Assistente:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# Perguntas Frequentes
faq_questions = [
    "Como posso abrir uma conta poupança?",
    "Quais são os benefícios do My Savings?",
    "Como funciona o seguro automóvel?",
]

with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Digite a sua pergunta aqui:", placeholder="Ex: Como posso abrir uma conta poupança?")
    send_btn = st.form_submit_button("Enviar")

    if send_btn and user_input.strip() != "":
        typing_placeholder = st.empty()
        for i in range(6):
            dots = "." * ((i % 3) + 1)
            typing_placeholder.markdown(f"🤖 Assistente está a pensar{dots}")
            time.sleep(0.4)
        typing_placeholder.empty()

        add_message("user", user_input)
        answer = chatbot_response(user_input)
        add_message("assistant", answer)
        st.experimental_rerun()

display_chat()

st.markdown("### Perguntas frequentes")
cols = st.columns(len(faq_questions))
for i, question in enumerate(faq_questions):
    if cols[i].button(question):
        typing_placeholder = st.empty()

        # Thinking
        for j in range(6):
            dots = "." * ((j % 3) + 1)
            typing_placeholder.markdown(f"🤖 Assistente está a pensar{dots}")
            time.sleep(0.4)

        typing_placeholder.empty()

        add_message("user", question)
        answer = chatbot_response(question)
        add_message("assistant", answer)
        st.experimental_rerun()

# Sidebar styling
st.markdown(
    """
    <style>
    /* Fundo da sidebar inteira */
    section[data-testid="stSidebar"] {
        background-color: #d80027 !important;
    }

    /* Conteúdo da sidebar */
    .sidebar .sidebar-content {
        color: white !important;
        padding: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Títulos na sidebar */
    .sidebar-title {
        font-weight: 700;
        font-size: 22px;
        margin-bottom: 15px;
        color: white;
    }

    /* Texto da sidebar */
    .sidebar-text {
        font-size: 16px;
        line-height: 1.5;
        color: white;
    }

    /* Texto forte/em */
    .sidebar-text strong, .sidebar-text em {
        color: #fff9f9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Content of the sidebar
st.sidebar.markdown('<div class="sidebar-title">Instruções</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
    <div class="sidebar-text">
    - Digite sua pergunta e clique em <em>Enviar</em>.<br>
    - Para terminar, escreva <strong>Obrigado, até à próxima!</strong>.<br>
    - Para iniciar nova conversa, escreva <strong>Nova Conversa</strong>.<br>
    - Para dúvidas técnicas, contacte a equipa especializada.
    </div>
""", unsafe_allow_html=True)

def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{b64}"

img_b64 = img_to_base64("fidelidade_logo.png")

st.markdown(f"""
    <style>
    .image-container {{
        display: flex;
        justify-content: center;  /* centraliza horizontalmente */
        align-items: center;      /* centraliza verticalmente */
        height: 250px;            /* altura do container */
        border: 1px solid transparent; /* opcional, para testar o contêiner */
    }}
    .image-container img {{
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }}
    </style>
    <div class="image-container">
        <img src="{img_b64}" alt="Logo Fidelidade"/>
    </div>
""", unsafe_allow_html=True)


# TO RUN THIS: 
#cd /Users/anaazinheira/Documents/BCwDS-Project-4/AssistantVirtual - TÊM QUE MUDAR A DIRETRIZ PARA O VOSSO PC 
# streamlit run app.py