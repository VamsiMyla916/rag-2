import streamlit as st
# --- NEW: Import new loaders and chain ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# --- FIX: 'PromptTemplate' is in 'langchain_core', not 'langchain' ---
from langchain_core.prompts import PromptTemplate
# ---
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
# --- NEW: Imports for text stats ---
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List, Dict, Any # For type hinting

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
    # --- NEW: Allow multiple file types ---
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

# --- Functions ---

# --- REVISED FUNCTION (FINAL) ---
@st.cache_resource
def download_nltk_data():
    """Downloads NLTK 'punkt' and 'punkt_tab' tokenizer models."""
    st.write("Downloading NLTK data (for word/sentence counting)...")
    # Download the main tokenizer model
    nltk.download('punkt')
    # --- NEW: Download the required supplementary table ---
    nltk.download('punkt_tab')
    st.write("NLTK data is ready.")


def get_vector_store(chunks: List[Any]) -> FAISS:
    """Creates a FAISS vector store from document chunks."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vector_store


@st.cache_resource
def get_llm_pipeline():
    """Loads and caches the HuggingFace LLM pipeline."""
    # --- NEW: Qwen 1.8B Model ---
    # This is a small but very powerful model, a great balance.
    model_id = "Qwen/Qwen1.5-1.8B-Chat"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        # Use bfloat16 for speed and memory
        torch_dtype=torch.bfloat16, 
        device_map=device,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, 
        # --- FIX: Use do_sample=False for deterministic, factual answers ---
        do_sample=False
        # --- (We remove temperature=0.0, as do_sample=False achieves the same goal) ---
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def clean_response(text: str) -> str:
    """Cleans the raw LLM output by splitting on the final assistant tag."""
    # The model's output includes the entire prompt. We only want
    # the text *after* the final <|assistant|> tag.
    if "<|assistant|>" in text:
        # Split the text at the *last* occurrence of <|assistant|>
        # and take the part that comes after it.
        text = text.rsplit("<|assistant|>", 1)[-1]
            
    return text.strip()

# --- Main App Logic ---

if uploaded_file is not None:
    # --- NEW: Download NLTK data for stats ---
    download_nltk_data()
    
    # Check if file has changed. If so, re-process it.
    if "vector_store" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        with st.spinner("Processing document..."):
            temp_dir = "temp_files"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- NEW: Document Loader Logic ---
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
            
            # --- NEW: Calculate Stats & Store Full Text ---
            full_text = "\n\n".join([doc.page_content for doc in documents])
            st.session_state.full_text = full_text
            
            words = word_tokenize(full_text)
            sentences = sent_tokenize(full_text)
            
            st.session_state.doc_stats = {
                "Character Count": len(full_text),
                "Word Count": len(words),
                "Sentence Count": len(sentences)
            }
            # --- End New Stats Logic ---
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            st.session_state.vector_store = get_vector_store(chunks)
            st.success("Document processed and knowledge base created!")

    # --- NEW: Create Tabs for Chat and Document Content ---
    tab1, tab2 = st.tabs(["💬 Chat with Document", "📄 Document Content & Stats"])

    with tab1:
        st.header("Chat with your Document")
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display prior chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- NEW: RAG Logic with Memory (LCEL Syntax) ---
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = get_llm_pipeline()
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                    
                    # 1. Build the Chat History String
                    history_string = ""
                    # Get all messages *except* the new user prompt
                    for msg in st.session_state.messages[:-1]: 
                        if msg["role"] == "user":
                            history_string += f"<|user|>\n{msg['content']}</s>\n"
                        elif msg["role"] == "assistant":
                            # Store the *clean* response for better history
                            history_string += f"<|assistant|>\n{msg['content']}</s>\n"
                    
                    # 2. Define the Prompt Template (with history)
                    # 2. Define the Prompt Template (with history)
                    # --- NEW: This is the official chat template for Qwen1.5 ---
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
                    # --- END OF NEW TEMPLATE ---
                    
                    PROMPT = PromptTemplate(
                        template=prompt_template, 
                        input_variables=["chat_history", "context", "question"]
                    )
                    # --- NEW: 3. Create the LCEL Chain ---
                    
                    # This helper function formats the retrieved documents
                    def format_docs(docs):
                        return "\n\n".join([doc.page_content for doc in docs])

                    # This runnable will take the user's prompt (a string)
                    # and return a dictionary for the PROMPT
                    chain_input = {
                        "context": (lambda x: x) | retriever | format_docs, # Pass prompt to retriever, then format
                        "question": (lambda x: x), # Pass through the original prompt
                        "chat_history": (lambda x: history_string) # Pass the history string we built
                    }
                    
                    # The full LCEL chain
                    chain = (
                        chain_input  # Input: "prompt" string
                        | PROMPT         # -> Populates the template
                        | llm            # -> Passes to the LLM
                        | StrOutputParser() # -> Gets the string output
                    )
                    
                    # 4. Invoke the chain
                    # The input to the chain is now *just* the prompt string
                    raw_response = chain.invoke(prompt)
                    
                    # 5. Clean and Display the Response
                    cleaned_response = clean_response(raw_response)
                    
                    st.markdown(cleaned_response)
                    
                    # --- NEW: Store the *clean* response for memory ---
                    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})

    # --- NEW: Tab 2 Logic ---
    with tab2:
        st.header("Document Content & Statistics")
        
        st.subheader("Document Statistics")
        if "doc_stats" in st.session_state:
            st.json(st.session_state.doc_stats)
        
        st.subheader("Full Document Content")
        if "full_text" in st.session_state:
            st.text_area("Content", st.session_state.full_text, height=400, disabled=True)

else:
    st.info("Please upload a .pdf, .docx, or .txt file to begin.")