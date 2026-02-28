import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_core.chat_history import BaseChatMessageHistory 
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.runnables.history import RunnableWithMessageHistory  
from langchain_classic.chains import create_history_aware_retriever


from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv 
load_dotenv()


## Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "RAG project OPENAI" 
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key =groq_api_key,model="qwen/qwen3-32b")


# prompt = ChatPromptTemplate.from_messages(


st.set_page_config(page_title="HR Policy Chatbot",page_icon='🧊',layout="centered")

st.title("HR Policy Chatbot") 

st.caption("Ask me anything about the India Leaves and Holiday Policy")

#-------------------------------------
# Sidebar - API Key & Config
#-------------------------------------
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.info("Enter your OpenAI API Key in the sidebar to start.")
    st.stop()


@st.cache_resource(show_spinner="Building vector store (one-time setup)...")
def build_retriever():
    loader = PyPDFLoader("India-Leaves and Holiday Policy.pdf")
    docs = loader.load()
    splitter =  RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 2. Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = FAISS.from_documents(chunks,embedding=embeddings)
    retriever = vectordb.as_retriever( search_kwargs={"k": 4})

    return retriever

def build_rag_chain (retriever,groq_api_key:str):

     # 4 LLM
    llm = ChatGroq(
         model = "qwen/qwen3-32b",
         temperature=0.1,
         api_key=groq_api_key,
         streaming=True
     )
     
    repharse_prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("human","{input}"),
      ("human", """Given the conversation above, rephrase the follow-up 
      question to be a standalone question. If no history, return as-is.""")
     ])
    
    history_aware_retriever = create_history_aware_retriever(llm,retriever,repharse_prompt)

    #5. QA chain
    qa_prompt =  ChatPromptTemplate.from_messages([("system","""
      You are an HR assistant for India leave policies.
      IMPORTANT: Always answer using ONLY the context provided below.
      Do NOT rely on general knowledge. If the answer is in the context, use it.
      If truly not found, say 'I don't know'.
      Context: {context}
      """),
      MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}")])

    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 6. Full Retrieval Chain
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    ## wrap with Message History:

    rag_with_history = RunnableWithMessageHistory(
     retrieval_chain,
     get_session_history=lambda session_id: st.session_state.chat_store.setdefault(session_id, ChatMessageHistory()),
     input_messages_key="input",
     history_messages_key="chat_history",
     output_messages_key="answer")
    
    return rag_with_history 


##Initilize

retriever  = build_retriever()

## Session-level:per user chain +history

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = build_rag_chain(retriever,groq_api_key)
    st.session_state.chat_store = {} ## Langchain message history
    st.session_state.messages = [] ## display history

# SIDEBAR — Clear chat

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_store = {}
    st.session_state.messages = []
    st.rerun()

##3 — STREAMLIT CHAT UI


# Show previous messages
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# Handle new input

if user_input := st.chat_input("Ask about leave or holiday policy..."):
    
    if user_input.strip().lower() in ['quit','exit']:
        st.session_state.exited = True
    else:
        st.session_state.messages.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user_1"}},
        )
    answer = response["answer"]

    st.session_state.messages.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

if st.session_state.get("exited"):
    st.success("Chat ended. Refresh the page to start again.")
    st.stop()  


# st.session_state.messages
# Stores the chat display history as a list of (role, text) tuples.
# Persists as long as the browser tab is open (until refresh/rerun)
# Used purely to re-render chat bubbles on screen after each Streamlit rerun
# Manually appended by you in the code



# st.session_state.chat_store
# Stores LangChain message history objects, keyed by session_id

# Contains structured HumanMessage and AIMessage objects (not plain text)
# This is what gets passed as chat_history to the LLM
# Auto-managed by RunnableWithMessageHistory — you don't append manually
# Enables the LLM to remember previous turns and rephrase follow-up questions correctly

