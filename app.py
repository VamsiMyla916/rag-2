import streamlit as st
import os
import torch
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List, Dict, Any 

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

# --- LangChain Core Imports (for LCEL) ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Transformers Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="📄 Chat with Your Documents",
    page_icon="📄",
    layout="wide"
)

# --- App Title ---
st.title("📄 Chat with Your Documents")

# --- Sidebar for File Upload & Contact Info ---
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Choose a .pdf, .docx, or .txt file", type=["pdf", "docx", "txt"])

    st.divider()
    st.markdown(
        """
        **I would appreciate any valuable feedback and open to discuss any ideas on how to improve this app further. Please find my contact here:**
        - [LinkedIn](https://www.linkedin.com/in/vamsimyla/)
        - [GitHub](https://github.com/VamsiMyla916/RAG-chatbot-streamlit)
        - [Email](mailto:mylavamsikrishnasai@gmail.com)
        """
    )

# --- Functions (all unchanged) ---

@st.cache_resource
def download_nltk_data():
    """Downloads NLTK 'punkt' and 'punkt_tab' tokenizer models."""
    st.write("Downloading NLTK data (for word/sentence counting)...")
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        st.write("NLTK data is ready.")
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        st.info("Please check your internet connection and restart the app.")


def get_vector_store(chunks: List[Any]) -> FAISS:
    """Creates a FAISS vector store from document chunks."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vector_store

@st.cache_resource
def get_llm_pipeline():
    """Loads and caches the HuggingFace LLM pipeline."""
    model_id = "Qwen/Qwen1.5-1.8B-Chat"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        device_map=device,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, 
        do_sample=False
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def clean_response(text: str) -> str:
    """Cleans the raw LLM output by splitting on the final assistant tag."""
    if "<|assistant|>" in text:
        text = text.rsplit("<|assistant|>", 1)[-1]
            
    return text.strip()

# --- Main App Logic ---

if uploaded_file is not None:
    # --- 1. Download NLTK data ---
    download_nltk_data()
    
    # --- 2. Process Document (if new file) ---
    if "vector_store" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        with st.spinner("Processing document..."):
            # Setup temp directory
            temp_dir = "temp_files"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- Document Loader Logic ---
            file_extension = os.path.splitext(uploaded_file.name)[1]
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path, encoding="utf-8")
            else:
                st.error("Unsupported file type!")
                st.stop()
                
            documents = loader.load()
            
            # --- Calculate Stats & Store Full Text ---
            full_text = "\n\n".join([doc.page_content for doc in documents])
            st.session_state.full_text = full_text
            
            words = word_tokenize(full_text)
            sentences = sent_tokenize(full_text)
            
            st.session_state.doc_stats = {
                "Character Count": len(full_text),
                "Word Count": len(words),
                "Sentence Count": len(sentences)
            }
            
            # --- Chunk and Vectorize ---
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            st.session_state.vector_store = get_vector_store(chunks)
            st.success("Document processed and knowledge base created!")

    # --- 3. Create Tabs ---
    tab1, tab2 = st.tabs(["💬 Chat with Document", "📄 Document Content & Stats"])

    # --- CHAT HISTORY TAB ---
    with tab1:
        st.header("Chat with your Document")
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display all messages from session state
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
       

    # --- DOCUMENT STATS TAB ---
    with tab2:
        st.header("Document Content & Statistics")
        
        st.subheader("Document Statistics")
        if "doc_stats" in st.session_state:
            st.json(st.session_state.doc_stats)
        
        st.subheader("Full Document Content")
        if "full_text" in st.session_state:
            st.text_area("Content", st.session_state.full_text, height=400, disabled=True)

    
    if prompt := st.chat_input("Ask a question about your document..."):
        
        # 1. Add user message to state (but don't display yet)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Run the RAG Chain
        with st.spinner("Thinking..."):
            llm = get_llm_pipeline()
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            
            # Build the Chat History String
            history_string = ""
            for msg in st.session_state.messages[:-1]: 
                if msg["role"] == "user":
                    history_string += f"<|user|>\n{msg['content']}<|endoftext|>\n"
                elif msg["role"] == "assistant":
                    history_string += f"<|assistant|>\n{msg['content']}<|endoftext|>\n"
            
            # Define the Prompt Template (for Qwen)
            prompt_template = """
            <|system|>
            You are a helpful AI assistant. Use the provided context to answer the user's question.
            If the answer is a list, format it with markdown bullets.
            If you don't know the answer, simply state that you don't know.
            Do not repeat the question or the context in your answer. Provide only the helpful answer itself.

            Context:
            {context}

            Chat History:
            {chat_history}<|endoftext|>
            <|user|>
            Question: {question}<|endoftext|>
            <|assistant|>
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["chat_history", "context", "question"]
            )
            
            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])

            chain_input = {
                "context": (lambda x: x) | retriever | format_docs,
                "question": (lambda x: x),
                "chat_history": (lambda x: history_string)
            }
            
            chain = (
                chain_input
                | PROMPT
                | llm
                | StrOutputParser()
            )
            
            raw_response = chain.invoke(prompt)
            cleaned_response = clean_response(raw_response)
            
        # 3. Add assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
        
        # 4. Rerun the app
        st.rerun() 

else:
    st.info("Please upload a .pdf, .docx, or .txt file to begin.")