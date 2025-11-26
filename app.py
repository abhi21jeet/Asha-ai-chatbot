import os
import streamlit as st
import dotenv
from chatbot.load_data import loading_data
from chatbot.split_text import split_text
from chatbot.embed_text import loading_embeddings
from chatbot.pinecone_setup import initializing_pinecone, uploading_data_to_pinecone
from chatbot.model import load_groq
from chatbot.util import download_nltk
from chatbot.retrieval import retrieve_context
from chatbot.prompt import build_prompt_template
from huggingface_hub import login
# from langchain.chains.retrieval_qa.base import RetrievalQA

from streamlit_lottie import st_lottie
import requests
import re
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

os.environ["PYTORCH_JIT"] = "0"  # Avoids internal torch class issues
os.environ["STREAMLIT_WATCH_DIRECTORIES"] = (
    "false"  # Disables problematic path inspection
)

# Disable file watcher for better performance
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"


def load_lottieurl(url: str):
    """Load Lottie animation from URL with error handling"""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load animation: {e}")
    return None


def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown(
        """
        <style>
        /* Main app styling */
        .stApp {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 25%, #e9ecef 50%, #f1f3f4 75%, #ffffff 100%);
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .sub-header {
            font-size: 1.2rem;
            color: #495057;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 500;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Card styling */
        .info-card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 2px solid rgba(255, 255, 255, 0.4);
            border-radius: 10px;
            padding: 0.75rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .stTextInput > div > div > input:focus {
            border-color: rgba(255, 255, 255, 0.6);
            box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.8);
        }
        
        /* Button styling */
        .stButton > button {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            color: #495057;
            border: 2px solid rgba(255, 255, 255, 0.4);
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.15);
            background: rgba(255, 255, 255, 0.8);
            border-color: rgba(255, 255, 255, 0.6);
        }
        
        /* Success message styling */
        .stSuccess {
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Warning message styling */
        .stWarning {
            background: linear-gradient(45deg, #ffeaa7, #fab1a0);
            color: #2d3436;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Spinner styling */
        .stSpinner {
            color: #667eea;
        }
        
        /* Response box styling */
        .response-box {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            color: #495057;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Stats styling */
        .stats-container {
            display: flex;
            justify-content: space-around;
            margin: 2rem 0;
        }
        
        .stat-item {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            flex: 1;
            margin: 0 0.5rem;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #6c757d;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #2c3e50;
            margin-top: 0.5rem;
        }
        
        /* Animation for loading */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading-text {
            animation: pulse 2s infinite;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# def build_rag_pipeline(llm, retriever):
#     """Build RAG pipeline with error handling"""
#     try:
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             return_source_documents=True,
#             chain_type="stuff",
#         )
#         return qa_chain
#     except Exception as e:
#         st.error(f"Error building RAG pipeline: {e}")
#         return None

def build_rag_pipeline(llm, retriever):
    """Build RAG pipeline with error handling using modern LangChain modules"""
    try:
           
        # Create the prompt template
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create the document combination chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the retrieval chain
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
        
    except Exception as e:
        st.error(f"Error building RAG pipeline: {e}")
        return None

def clean_response(response):
    """Clean and format the response"""
    # Remove the template pattern from response
    pattern = r"""You are a helpful assistant who provides information about job opportunities specifically targeted towards women\.\n\nStrictly use only the provided context below to answer the user's query\.\n\nIf the answer is not available in the context, reply: "I couldn't find a suitable opportunity at the moment\."\nDo NOT make up, guess, or add any information not present in the context\.\nKeep your answers clear,\s*concise, and relevant to the question\.\nContext: Here are some of the top companies offering job opportunities for women:"""

    cleaned_response = re.sub(pattern, "", response, flags=re.MULTILINE | re.DOTALL)
    return cleaned_response.strip()


def display_header():
    """Display the app header with animation"""
    # Load and display Lottie animation
    lottie_chat = load_lottieurl(
        "https://assets7.lottiefiles.com/packages/lf20_qp1q7mct.json"
    )
    if lottie_chat:
        st_lottie(lottie_chat, height=200, key="chat")

    st.markdown(
        '<div class="main-header">🌟 Women\'s Job Finder AI 🌟</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">🚀 Discover amazing career opportunities designed for women! 💼✨</div>',
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Initialize session state variables"""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chunks_count" not in st.session_state:
        st.session_state.chunks_count = 0


def display_features():
    """Display app features"""
    st.markdown("### 🎯 What I Can Help You With:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔍 Job Search**
        - Women-focused positions
        - Remote opportunities
        - Internships & entry-level
        """)

    with col2:
        st.markdown("""
        **🏢 Company Insights**
        - Women-friendly workplaces
        - Diversity & inclusion
        - Career growth paths
        """)

    with col3:
        st.markdown("""
        **💡 Career Guidance**
        - Industry trends
        - Skill requirements
        - Application tips
        """)


def main():
    st.set_page_config(
        page_title="Women's Job Finder AI",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_custom_css()
    initialize_session_state()
    display_header()

    # Load environment variables
    dotenv.load_dotenv()

    # Check for required environment variables
    required_vars = ["HF_TOKEN", "PINECONE_API_KEY", "PINECONE_INDEX"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        st.error(
            f"❌ Missing required environment variables: {', '.join(missing_vars)}"
        )
        st.stop()

    # Display features
    display_features()

    # Job websites URLs
    URLs = [
        "https://www.herkey.com/jobs",
        "https://powertofly.com/jobs",
        "https://leanin.org/circles",
        "https://internshala.com/internships/work-from-home-internships/women/",
        "https://www.womenwhocode.com/jobs",
        "https://www.womentech.net/jobs",
        "https://internshala.com/jobs-for-women/",
        "https://apna.co/jobs/female-jobs-in-lucknow",
    ]

    # Initialize components
    try:
        download_nltk()

        # Login to Hugging Face
        hf_token = os.getenv("HF_TOKEN")
        login(hf_token)

        # Initialize embeddings and Pinecone
        embeddings = loading_embeddings()
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX")

        pc, index = initializing_pinecone(pinecone_api_key, pinecone_index, embeddings)

        # Load and process data (only once)
        if not st.session_state.data_loaded:
            with st.spinner("🔄 Loading job data from websites..."):
                progress_bar = st.progress(0)

                # Load data
                progress_bar.progress(25)
                data = loading_data(URLs)

                # Split text
                progress_bar.progress(50)
                chunks = split_text(data)
                st.session_state.chunks_count = len(chunks)

                # Upload to Pinecone
                progress_bar.progress(75)
                uploading_data_to_pinecone(index, chunks, embeddings)

                progress_bar.progress(100)
                st.session_state.data_loaded = True

                st.success(f"✅ Successfully loaded {len(chunks)} job listings!")
                time.sleep(1)
                st.rerun()

        # Load model (only once)
        if not st.session_state.model_loaded:
            with st.spinner("🤖 Loading AI model..."):
                llm = load_groq()
                prompt_template = build_prompt_template()
                st.session_state.llm = llm
                st.session_state.prompt_template = prompt_template
                st.session_state.model_loaded = True
                st.success("✅ AI model loaded successfully!")
                time.sleep(1)
                st.rerun()

        # Display stats
        if st.session_state.data_loaded:
            st.markdown("### 📊 Current Database Stats")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Job Listings", st.session_state.chunks_count, delta="Active")
            with col2:
                st.metric("Data Sources", len(URLs), delta="Updated")
            with col3:
                st.metric("AI Status", "Ready", delta="Online")

        # Chat interface
        st.markdown("---")
        st.markdown("### 💬 Ask Your Job-Related Questions")

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("#### Previous Conversations:")
            for i, (question, answer) in enumerate(
                st.session_state.chat_history[-3:]
            ):  # Show last 3
                with st.expander(f"Q{i + 1}: {question[:50]}..."):
                    st.write(f"**Q:** {question}")
                    st.write(f"**A:** {answer}")

        # Input field
        user_query = st.text_input(
            "Your Question:",
            placeholder="e.g., 'Show me remote jobs for women in tech' or 'What companies hire women developers?'",
            key="user_input",
        )

        # Query processing
        if (
            user_query
            and st.session_state.data_loaded
            and st.session_state.model_loaded
        ):
            with st.spinner("🔍 Searching for relevant opportunities..."):
                try:
                    # Retrieve context
                    context = retrieve_context(
                        user_query, index=index, embeddings=embeddings
                    )

                    # Generate response
                    full_prompt = st.session_state.prompt_template.format(
                        context=context, question=user_query
                    )

                    response = st.session_state.llm.invoke(full_prompt)

                    if isinstance(response, BaseMessage):
                        response = response.content

                    # Clean response
                    cleaned_response = clean_response(response)

                    # Display response
                    st.markdown("### 🎯 Here's What I Found:")
                    st.markdown(
                        f'<div class="response-box">{cleaned_response}</div>',
                        unsafe_allow_html=True,
                    )

                    # Add to chat history
                    st.session_state.chat_history.append((user_query, cleaned_response))

                    # Feedback buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Helpful", key="helpful"):
                            st.success("Thanks for your feedback!")
                    with col2:
                        if st.button("👎 Not Helpful", key="not_helpful"):
                            st.info("We'll work on improving our responses!")

                except Exception as e:
                    st.error(f"❌ Error processing your query: {e}")

        elif user_query and not st.session_state.data_loaded:
            st.warning("⏳ Please wait for data to load before asking questions.")

        elif user_query and not st.session_state.model_loaded:
            st.warning("⏳ Please wait for the AI model to load.")

    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.info("Please check your environment variables and try again.")


if __name__ == "__main__":
    main()
