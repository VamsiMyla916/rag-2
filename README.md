# 📄 Upgraded RAG Chatbot

This is an advanced Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit and LangChain. It allows you to upload various document types (`.pdf`, `.docx`, `.txt`) and hold a "memory-aware" conversation about their content.

The application runs a local, open-source LLM (`Qwen/Qwen1.5-1.8B-Chat`) to ensure 100% privacy and fast, local inference.

## ✨ Key Features

- 📂 **Multi-Document Support:** Ingests and processes `.pdf`, `.docx`, and `.txt` files.
- 🧠 **Conversational Memory:** Remembers previous turns of the conversation to answer follow-up questions contextually.
- 🤖 **Local & Private LLM:** Uses the powerful `Qwen/Qwen1.5-1.8B-Chat` model, running entirely on your local machine (or the server). No data is sent to external APIs.
- 📈 **Document Analysis Tab:** Provides a separate tab to view the document's full raw text and see statistics like word, character, and sentence counts.
- 🧹 **Clean & Stable UI:** Features a clean, modern chat interface where the chat input is correctly docked to the bottom of the page and does not produce "ghost" or "faded" duplicate messages.

## 🛠️ Tech Stack

This project integrates several key technologies to create a full-stack RAG pipeline.

- **Application Framework:** Streamlit
- **Core RAG Framework:** LangChain (using the modern LangChain Expression Language - LCEL)
- **Large Language Model (LLM):** `Qwen/Qwen1.5-1.8B-Chat`
- **Embeddings Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS (for fast, in-memory similarity search)
- **Document Loaders:** `PyPDFLoader`, `Docx2txtLoader`, `TextLoader`
- **Text Analysis:** NLTK (for word/sentence tokenization)

## 🚀 Getting Started

You can run this application on your local machine by following these steps.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Create and Activate a Virtual Environment
Bash

# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Install all the required Python packages from the requirements.txt file.

Bash

pip install -r requirements.txt
4. Run the Application
Bash

streamlit run app.py
Your app will open in your default web browser.

☁️ Deployment on Hugging Face Spaces
This app is optimized for deployment on Hugging Face Spaces.

Create a new Space on Hugging Face.

Select Streamlit as the Space SDK.

Upload your app.py file.

Create a requirements.txt file in your repository with the following content:

streamlit
transformers
torch
sentence-transformers
faiss-cpu
pypdf
accelerate
langchain
langchain-community
langchain-core
langchain-text-splitters
numpy
docx2txt
nltk
(CRITICAL) Create a packages.txt file in your repository to install the system dependency for FAISS. The file should contain only one line:

libomp-dev
Your app will build and launch automatically.
```

Future Work & Key Learnings:
This project is a great foundation, and I'm already planning the next steps. Here are a few features I'm excited to work on next, along with some key insights I gained during development:

Smarter Chat Memory: Right now, the app uses a basic "buffer" memory. I plan to upgrade this to a "summary" memory, where the AI summarizes the chat as it goes. This would allow for a much longer conversation, which is a key challenge for any real-world co-pilot.

Advanced Document Parsing: The new OCR feature is great for scanned text, but it can struggle with complex tables or charts. The next step would be to integrate a more powerful parsing library (like unstructured.io) to handle these complex layouts.

Why I Removed the Re-ranker: I experimented with adding a "re-ranker" model to double-check the search results. I found that it was actually less accurate than the simple, fast vector search and was adding unnecessary complexity, so I made the engineering decision to remove it.

Why I Removed the "Guardrail": I also prototyped a "relevance guardrail" to block bad answers. I found it had a major trade-off: to make it work, I had to limit the AI's context to only one text chunk. This made the app "dumber" and unable to answer broad questions (like "what is the author's name?").

A Better Solution ("Show Sources"): The guardrail failure led me to a much better feature. Instead of a "black box" that just says "no," I added the "Show Sources" button. This is more transparent and builds trust by letting you be the guardrail, which felt like a more honest and effective solution.
