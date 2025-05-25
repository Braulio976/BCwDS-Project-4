import streamlit as st
import os
import json
import pickle
from openai import AzureOpenAI

from utils import (
    create_assistant,
    create_thread,
    check_assistant_exists,
    load_and_upload_files,
    send_message_to_assistant
)

# --- Configurações iniciais do Azure OpenAI ---
api_key = 'yourAPIKey'
endpoint = 'https://ai-bcds.openai.azure.com/'

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-05-01-preview"
)

# Carregar mapa de links
with open("document_links.json", "r", encoding="utf-8") as f:
    doc_link_map = json.load(f)

vector_data = 'vector_store.pkl'

if os.path.exists(vector_data):
    with open(vector_data, "rb") as file:
        vector_store = pickle.load(file)
else:
    vector_store = load_and_upload_files(client, link_map=doc_link_map)

# Role do assistente
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

# --- Estilos customizados ---
st.markdown(
    """
    <style>
    /* Fonte geral */
    body, .block-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Cabeçalho */
    .title {
        text-align: center;
        color: #B22222; /* Vermelho escuro */
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 25px;
    }

    /* Contêiner do chat */
    .chat-container {
        max-width: 700px;
        margin: 0 auto 30px auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    /* Mensagem do usuário */
    .user-message {
        align-self: flex-end;
        background-color: #B22222; /* Vermelho escuro */
        color: white;
        padding: 12px 18px;
        border-radius: 15px 15px 0 15px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 2px 2px 8px rgba(178, 34, 34, 0.3);
        font-size: 16px;
    }

    /* Mensagem do assistente */
    .assistant-message {
        align-self: flex-start;
        background-color: #F4F4F4;
        color: #333;
        padding: 12px 18px;
        border-radius: 15px 15px 15px 0;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        font-size: 16px;
    }

    /* Input container */
    .input-container {
        display: flex;
        max-width: 700px;
        margin: 0 auto 40px auto;
        gap: 10px;
    }

    /* Caixa de texto */
    input[type="text"] {
        flex-grow: 1;
        padding: 12px 15px;
        border: 2px solid #B22222;
        border-radius: 10px;
        font-size: 16px;
        outline: none;
    }

    input[type="text"]:focus {
        border-color: #8B0000;
        box-shadow: 0 0 5px #8B0000;
    }

    /* Botão */
    .send-button {
        background-color: #B22222;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }

    .send-button:hover {
        background-color: #8B0000;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .sidebar-title {
        font-weight: 700;
        color: #B22222;
        font-size: 22px;
        margin-bottom: 15px;
    }

    .sidebar-text {
        font-size: 16px;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Cabeçalho ---
st.markdown('<div class="title">🤖 Chatbot Fidelidade</div>', unsafe_allow_html=True)

# Inicializa histórico da conversa
if "messages" not in st.session_state:
    st.session_state.messages = []

# Função para obter resposta do assistente
def chatbot_response(user_input):
    full_prompt = f"{prompt_rules}\n\nUser question: {user_input}"
    response = send_message_to_assistant(client, thread, assistant, user_input, full_prompt, displayedMessagesIDs)
    return response

# Adiciona mensagem ao histórico
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Mostrar histórico da conversa com estilo
def display_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>Você:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Assistente:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input e botão lado a lado
with st.form(key="input_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 2])
    user_input = col1.text_input("Digite a sua pergunta aqui:")
    send_btn = col2.form_submit_button("Enviar")

    if send_btn and user_input.strip() != "":
        add_message("user", user_input)
        answer = chatbot_response(user_input)
        add_message("assistant", answer)
        st.experimental_rerun()

display_chat()

# Sidebar com instruções
st.sidebar.markdown('<div class="sidebar-title">ℹ️ Instruções</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div class="sidebar-text">
    - Digite sua pergunta e clique em <em>Enviar</em>.<br>
    - Para terminar, escreva <strong>Obrigado, até à próxima!</strong>.<br>
    - Para iniciar nova conversa, escreva <strong>Nova Conversa</strong>.<br>
    - Para dúvidas técnicas, contacte a equipa especializada.
    </div>
    """,
    unsafe_allow_html=True
)


# Para correr (têm que mudar a diretriz)
# cd /Users/anaazinheira/Documents/BCwDS-Project-4/AssistantVirtual
# streamlit run app.py
