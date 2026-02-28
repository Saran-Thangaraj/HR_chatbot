HR Policy Chatbot
An AI-powered conversational chatbot built with Streamlit and LangChain that lets employees ask natural language questions about the India Leaves and Holiday Policy using Retrieval-Augmented Generation (RAG).

📋 Features

RAG-based Q&A — Retrieves answers directly from your HR policy PDF
Conversation Memory — Maintains chat history for multi-turn, context-aware conversations
Query Rephrasing — Automatically rephrases follow-up questions into standalone queries
Fast Inference — Powered by Groq's LPU inference engine with the llm model
Chat UI — Clean Streamlit chat interface with message history
Clear Chat — Reset conversation anytime via the sidebar


🛠️ Tech Stack
LayerTechnologyUIStreamlitLLMGroq (LLM)OrchestrationLangChainEmbeddingsHuggingFace (emdding)Vector StoreFAISSDocument LoaderLangChain PyPDFLoader

⚙️ How It Works

1.PDF Loading — The HR policy PDF is loaded and split into overlapping text chunks (500 tokens, 100 overlap).
2.Embeddings — Chunks are embedded using sentence-transformers/all-mpnet-base-v2 and stored in a FAISS vector store (built once and cached).
3.History-Aware Retrieval — On each user question, chat history is used to rephrase follow-up questions into standalone queries before retrieval.
4.Answer Generation — The LLM generates answers strictly based on the retrieved context, falling back to "I don't know" if the answer isn't found in the policy.
5.Session Memory — LangChain's RunnableWithMessageHistory maintains conversation state per session.

⚠️ Limitations

Answers are limited to information present in the uploaded PDF — the model will not use general knowledge.
The vector store is rebuilt on first run; subsequent runs use a cached version.
Currently supports a single PDF document as the knowledge source.

