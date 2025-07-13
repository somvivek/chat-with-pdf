# 🧠 Chat with Your PDF (LangChain + Streamlit)

A simple RAG (Retrieval-Augmented Generation) app that lets you upload a PDF and chat with it using OpenAI and LangChain. One at a time.

## 🚀 Features

- Upload any PDF file
- Ask questions based on the PDF content
- Remembers your conversation
- Built using:
  - LangChain
  - Streamlit
  - OpenAI GPT models

## 🔧 Setup

1. Clone the repo  
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

4. Add your OpenAI key in the UI when prompted

## 💡 Example Use Cases

- Summarize long travel itineraries  
- Ask questions about legal contracts  
- Extract insights from research papers  

## 🛡️ Note

Your OpenAI API key is not stored or logged. The app uses in-session memory only.
