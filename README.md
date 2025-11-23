
# 🚀 Job Finder AI - Your Helpful AI Assistant

**A Retrieval-Augmented Generation (RAG) Chatbot using Groq, LangChain, Pinecone, and Streamlit**



## 📜 Overview
It is a Retrieval-Augmented Generation (RAG) based chatbot designed to give intelligent, contextual answers.  
It uses **LangChain** to manage retrieval and generation, **Pinecone** for storing and querying vectors, **Groq API** for fast large language model (LLM) inference, and a **Streamlit** app for the frontend.

---

## 🚀 Features
- 🔎 **Contextual retrieval** of information using **Pinecone** vector database.
- ⚡ **Groq-powered LLMs** (like Mixtral, LLaMA 3) generate human-like responses.
- 🔗 **LangChain** integration for flexible retrieval-generation workflows.
- 🖥️ **Streamlit UI** for a clean and interactive user experience.
- ⚙️ Fast, intelligent, and memory-efficient design using Groq's low-latency APIs.

---

## 🧩 How it Works
1. **User inputs a query.**
2. **Pinecone** retrieves top relevant documents.
3. Retrieved context + query is formatted into a **custom prompt**.
4. **Groq LLM** generates a coherent, helpful answer.
5. The **Streamlit** app displays the response.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **LangChain**
- **Pinecone**
- **Groq Python SDK**
- **Streamlit**
- **FAISS** (optional for local retrieval without Pinecone)

---

## 📂 Project Structure
```bash
.
├── app.py                # Main Streamlit application
├── chatbot/              
│   ├── embed_text.py      # Text embedding utilities
│   ├── load_data.py       # Load and prepare documents
│   ├── groq_setup.py      # Load Groq LLM models
│   ├── pinecone_setup.py  # Setup and manage Pinecone connection
│   ├── prompt.py          # Build custom prompt templates
│   ├── retrieval.py       # Retrieve context from Pinecone
│   ├── split_text.py      # Text splitting into chunks
│   ├── util.py            # Utility functions
├── requirements.txt       # Python dependencies
├── .gitignore             # Files/folders to ignore by Git
└── .devcontainer/         # Dev container setup for VS Code (optional)
````

---

## 🏗️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Asha-ai-chatbot.git
cd Asha-ai-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables

Create a `.env` file inside the `chatbot/` folder and add:

```ini
GROQ_API_KEY=your-groq-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=your-pinecone-index-name
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📚 Example Query

> **User:** "What are the job opportunities for women in India?"

> **Bot:**
> "There are several job opportunities for women in India, especially in companies like Google, Accenture, and IBM. Positions like Human Resource Specialist, Software Developer, and Healthcare Professional are quite common."

---

## ✨ Future Improvements

* Add **chat history** (memory feature).
* Enable **switching between multiple LLMs** (Groq, GPT-4, custom LLMs).
* Add **authentication** for secure document uploads.
* Improve **UI/UX** with chat animations and avatars.

---

## 📝 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute it!

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Pinecone](https://www.pinecone.io/)
* [Groq](https://groq.com/)
* [Streamlit](https://streamlit.io/)

```
## Screenshots

![Screenshot 1](https://github.com/user-attachments/assets/61332f82-ad98-4fe9-9935-bd5528722dfc)

![Screenshot 2](https://github.com/user-attachments/assets/1d2e59c0-1450-49f0-8671-31ce43ac9a03)

![Screenshot 3](https://github.com/user-attachments/assets/a7b052b8-501c-46a2-b871-04718bbea611)




```
