# RAG Chatbot

I built this chatbot as part of my learning project. It lets you upload any document 
and chat with it — instead of reading the whole file yourself, just ask questions and 
get instant answers.

## What it does

You upload a PDF, Word document, or text file. The app reads it, breaks it into 
chunks, and stores it in a vector database. When you ask a question, it finds the 
most relevant parts and uses Gemini AI to give you a proper answer. It also gives 
a simple ELI5 (Explain Like I'm 5) version of every answer so it's easy to understand.

## Tools I used

- Streamlit for the UI
- LangChain to connect everything together
- Google Gemini as the language model
- HuggingFace embeddings to understand the document
- FAISS to store and search the chunks

## How to run it

1. Clone this repo
2. Install the packages
   pip install -r requirement.txt
3. Create a .env file and add your Gemini API key
   GOOGLE_API_KEY=your_key_here
4. Run the app
   streamlit run "rag chatbot.py"

