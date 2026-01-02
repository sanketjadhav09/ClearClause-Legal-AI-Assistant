import streamlit as st
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from docx import Document
import fitz  # PyMuPDF

# --- Basic Setup ---
load_dotenv()
api_key = os.getenv("API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API Key not found. Please set it in the .env file.")
    st.stop()

# --- Helper: Safe text extraction ---
def safe_extract_text(response):
    try:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text
    except Exception:
        pass
    return "(No content returned — possibly cold start or quota limit)"

# --- Helper: Robust Gemini call ---
def robust_generate(prompt, retries=2):
    models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash"]
    for model_name in models_to_try:
        model = genai.GenerativeModel(model_name)
        for attempt in range(retries + 1):
            try:
                resp = model.generate_content(prompt)
                text = safe_extract_text(resp)
                if text.strip() and "No content returned" not in text:
                    return text
            except Exception as e:
                if "429" in str(e):  # Quota exceeded
                    break  # switch to next model
                if attempt < retries:
                    time.sleep(1)
                    continue
                return f"(Error: {e})"
    return "(No model could return a valid response)"

# --- Warm-up ---
try:
    robust_generate("Hello")  # prime the connection
except Exception as e:
    print(f"Warm-up failed: {e}")

# --- Document Processing ---
def get_document_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.name.endswith('.pdf'):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
        elif file.name.endswith('.docx'):
            document = Document(file)
            for para in document.paragraphs:
                text += para.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("Could not extract text from documents. Please check the files.")
        return
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.raw_text = "\n".join(text_chunks)
        st.sidebar.success("Documents Processed Successfully!", icon="✅")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()

def get_conversational_chain():
    prompt_template = "You are ClearClause... Context:\n {context}\n Question: \n{question}\n Answer:"
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_summary_chain():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    return load_summarize_chain(model, chain_type="map_reduce")

# --- QnA & Summarization ---
def handle_document_qna(user_question):
    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def handle_general_qna(user_question):
    prompt = f"You are ClearClause, a helpful Legal AI Assistant. Answer: {user_question}"
    return robust_generate(prompt)

def generate_summary(custom_instruction):
    if "raw_text" not in st.session_state or not st.session_state.raw_text:
        st.error("No document text found to summarize.")
        return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([st.session_state.raw_text])
    chain = get_summary_chain()
    with st.spinner("Generating detailed summary..."):
        initial_summary = chain.run(docs)
        refine_prompt = f"Given the following summary, refine it to follow: '{custom_instruction}'.\n\nSUMMARY:\n{initial_summary}"
        final_summary = robust_generate(refine_prompt)
    st.session_state.chat_history.append(("ClearClause", f"**Summary (Instruction: *{custom_instruction}*)**\n\n" + final_summary))

def translate_text(text, target_language):
    prompt = f"Translate the following English text to {target_language}: '{text}'"
    return robust_generate(prompt)

# --- Page Config ---
st.set_page_config(page_title="ClearClause - Legal AI", page_icon="⚖️", layout="wide")
st.markdown("""
    <style> 
        <style>
   @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* A safer way to apply the font */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background-color: #F0F2F6;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6EAF1;
    }
    
    .main .block-container {
        padding: 2rem;
    }

    /* Custom App Header Bar */
    .main h1 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #212529;
        background-color: #FFFFFF;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E6EAF1;
        margin-bottom: 1.5rem;
    }

    /* Chat Messages Styling */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border-radius: 0.75rem;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E6EAF1;
    }
    
    [data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) {
        background-color: #E6F1FF;
    }
    
    /* Form Input Styling */
    [data-testid="stTextArea"] textarea, 
    [data-testid="stTextInput"] input, 
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border: 1px solid #D1D9E4;
        border-radius: 0.5rem;
        background-color: #FFFFFF;
    }
    [data-testid="stTextArea"] textarea:focus, 
    [data-testid="stTextInput"] input:focus, 
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: #0D6EFD;
        box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25);
    }
</style>""", unsafe_allow_html=True)

# --- Session State ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "raw_text" not in st.session_state: st.session_state.raw_text = ""

# --- Sidebar ---
with st.sidebar:
    st.image("logo.png", width=250)
    st.title("ClearClause Menu")
    st.write("---")

    st.header("1. Upload Document")
    uploaded_files = st.file_uploader("Upload your legal documents", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Analyzing documents..."):
                raw_text = get_document_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
        else:
            st.warning("Please upload at least one document.", icon="⚠️")

    if st.session_state.vector_store:
        st.write("---")
        st.header("2. Custom Summary")
        custom_instruction = st.text_input("Custom Summary Instruction (Optional):", "Provide a standard one-paragraph summary.")
        if st.button("Generate Summary", use_container_width=True):
            generate_summary(custom_instruction)

    st.write("---")
    st.header("3. Manage Chat")
    with st.expander("Conversation History"):
        if st.session_state.chat_history:
            for role, text in st.session_state.chat_history:
                if role == "You":
                    st.text(f"You: {text[:40]}...")
        else:
            st.text("No questions asked yet.")

    if st.button("Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.raw_text = ""
        st.rerun()

# --- Main UI ---
st.title("⚖️ ClearClause - Your Legal AI Assistant")
chat_tab, translate_tab = st.tabs(["Chat", "Translate"])

with chat_tab:
    for role, text in st.session_state.chat_history:
        with st.chat_message(role, avatar="👤" if role == "You" else "🤖"):
            st.markdown(text)

with translate_tab:
    st.subheader("Translate Text")
    text_to_translate = st.text_area("Enter English text to translate:", height=150)
    target_language = st.selectbox("Select Target Language", ["Marathi", "Hindi"])
    if st.button("Translate", use_container_width=True):
        if text_to_translate:
            with st.spinner(f"Translating to {target_language}..."):
                translated_text = translate_text(text_to_translate, target_language)
                if translated_text:
                    st.success("Translation:")
                    st.text_area("", value=translated_text, height=150)
        else:
            st.warning("Please enter text to translate.")

# --- Chat Input ---
prompt = st.chat_input("Ask a question about your document or a general legal query...")
if prompt:
    st.session_state.chat_history.append(("You", prompt))
    if st.session_state.vector_store:
        response_text = handle_document_qna(prompt)
    else:
        response_text = handle_general_qna(prompt)
    st.session_state.chat_history.append(("ClearClause", response_text))
    st.rerun()
