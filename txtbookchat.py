import streamlit as st
import chromadb
import ollama
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()
# Password for accessing the application
APP_PASSWORD = os.getenv("APP_PASSWORD")  # Change this to your desired password

# Initialize Ollama client
ollama_client = ollama.Client(host="http://localhost:11434")

# Specify ChromaDB paths for the TVM use case
db1_path = r"C:\Users\nahertle\Desktop\textbook\TVM-time-value-of-money-extracted_chroma_db"
db2_path = r"C:\Users\nahertle\Desktop\textbook\TVM-interest-rate-extracted_chroma_db"

# Initialize ChromaDB clients
chroma_client1 = chromadb.PersistentClient(path=db1_path)
chroma_client2 = chromadb.PersistentClient(path=db2_path)

# Get collections
collection1 = chroma_client1.get_or_create_collection(name="TVM_time_value_collection")
collection2 = chroma_client2.get_or_create_collection(name="TVM_interest_rate_collection")

def get_relevant_context(query: str, collection) -> str:
    """Fetch relevant context from the ChromaDB collection based on a query."""
    try:
        query_embedding = ollama_client.embeddings(
            model="nomic-embed-text:latest",
            prompt=query
        )['embedding']

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,  # Only one document in each database
            include=['documents']
        )

        # Return the first document as the context
        documents = results.get('documents', [])
        return documents[0] if documents else ""
    except Exception as e:
        st.error(f"Error fetching context: {e}")
        return ""

def generate_response(messages: List[Dict], context: str, model: str, system_prompt: str) -> str:
    """Generate a response based on the context and user messages."""
    try:
        full_prompt = f"{system_prompt}\n\nContext: {context}"
        full_messages = [{"role": "system", "content": full_prompt}] + messages

        response = ollama_client.chat(
            model=model,
            messages=full_messages
        )

        return response['message']['content']
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Error generating response."

def synthesize_responses(response1: str, response2: str, model: str, system_prompt: str) -> str:
    """Synthesize responses from two models into a cohesive final output."""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"TVM Time Value Analysis:\n{response1}\n\nTVM Interest Rate Analysis:\n{response2}"}
        ]

        response = ollama_client.chat(
            model=model,
            messages=messages
        )

        return response['message']['content']
    except Exception as e:
        st.error(f"Error synthesizing responses: {e}")
        return "Error synthesizing responses."

# Initialize session state for password authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Authentication Block
if not st.session_state.authenticated:
    st.title("🔒 TVM Cross-Analysis Bot")
    password = st.text_input("Enter the password to access the application:", type="password")
    if st.button("Login"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.success("Authentication successful!")
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

# Main Application
if st.session_state.authenticated:
    st.title("TVM Cross-Analysis Bot")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar for model and prompt configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Model selection
        st.subheader("Models")
        model1 = st.selectbox("TVM Time Value Analysis Model", ["llama3.2", "llama3", "llama2", "mistral", "qwen2.5:14b"], key="model1")
        model2 = st.selectbox("TVM Interest Rate Analysis Model", ["llama3.2", "llama3", "llama2", "mistral", "qwen2.5:14b"], key="model2")
        synthesis_model = st.selectbox("Cross-Analysis Model", ["llama3.2", "llama3", "llama2", "mistral", "qwen2.5:14b"], key="synthesis")
        
        # System prompts
        st.subheader("System Prompts")
        prompt1 = st.text_area("TVM Time Value Analysis Prompt", "You are an expert in analyzing the time value of money (TVM). The concept of TVM emphasizes that a dollar today is worth more than the same dollar in the future due to its earning potential and associated risks. Using the principles covered in the textbook, explain how the present value (PV) and future value (FV) of cash flows are calculated. Include methods for handling compounding periods and the relevance of inflation and discount rates. Ensure all examples use accurate formulas and are displayed clearly in Markdown or LaTeX for precision.")
        prompt2 = st.text_area("TVM Interest Rate Analysis Prompt", "You are a financial expert focusing on the impact of interest rates on investments and debts. Using the textbook principles, provide an analysis of how interest rates influence the pricing of financial instruments, including discount bonds, simple loans, and fixed-payment loans. Discuss yield to maturity, real versus nominal rates, and the inverse relationship between bond prices and market interest rates. Present all formulas and examples using Markdown or properly formatted LaTeX for clarity.")
        synthesis_prompt = st.text_area("Synthesis Prompt", "Combine insights from the Time Value of Money (TVM) and Interest Rate Analyses. Discuss how the interplay of interest rates and the time value of money affects financial decision-making. Address practical applications, such as retirement planning, debt repayment strategies, and investment decisions. Provide your recommendations and conclusions clearly, with examples formatted in Markdown or LaTeX to ensure readability.")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about TVM insights:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get context and generate responses
        with st.spinner("Searching databases..."):
            context1 = get_relevant_context(prompt, collection1)
            context2 = get_relevant_context(prompt, collection2)

        with st.spinner("Analyzing data..."):
            response1 = generate_response(st.session_state.messages, context1, model1, prompt1)
            response2 = generate_response(st.session_state.messages, context2, model2, prompt2)

        # Synthesize responses
        with st.spinner("Synthesizing insights..."):
            final_response = synthesize_responses(response1, response2, synthesis_model, synthesis_prompt)

        with st.chat_message("assistant"):
            st.markdown(final_response)

        # Store assistant response
        st.session_state.messages.append({"role": "assistant", "content": final_response})
