# pipeline.py - builds the LangChain RAG pipeline
# this is where the magic happens - history-aware question reformulation + answering

import yaml
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


# --- history store ---
# this dict maps session_id -> ChatMessageHistory object
# it's simple, in-memory. for production you'd use Redis or something
history_store = {}


def get_session_history(session_id: str):
    # returns the chat history for a session, creates if not exists
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]


# --- LLM ---
def get_llm(api_key):
    # groq LLM using config settings
    return ChatGroq(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        groq_api_key=api_key
    )


# --- contextualize question chain ---
# this takes the chat history + new question and rewrites it
# so it's a standalone question that doesn't depend on previous context
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, formulate a standalone question "
    "which can be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# --- answer chain prompt ---
# this is what the LLM sees when answering - strict instructions to use context only
answer_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def create_rag_chain_with_history(vector_store, api_key):
    # builds the full RAG chain with history awareness
    
    llm = get_llm(api_key)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": cfg["retrieval"]["k"]}
    )
    
    # helper to format retrieved docs into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # --- step 1: contextualize question using history ---
    # if there's history, rewrite the question to be standalone
    contextualize_q_chain = (
        contextualize_q_prompt
        | llm
        | StrOutputParser()
    )
    
    # --- step 2: retrieve docs based on the (possibly rewritten) question ---
    # this is a custom runnable that contextualizes then retrieves
    def contextualized_retriever(input_dict):
        # input_dict has "input" (question) and "chat_history"
        # first check if there's chat history
        if input_dict.get("chat_history"):
            # rewrite question using history
            standalone_q = contextualize_q_chain.invoke(input_dict)
        else:
            # no history, use question as-is
            standalone_q = input_dict["input"]
        
        # retrieve docs using standalone question
        docs = retriever.invoke(standalone_q)
        return docs
    
    # --- step 3: answer using retrieved docs + history ---
    # build the RAG chain
    rag_chain = (
        {
            "context": lambda x: format_docs(contextualized_retriever(x)),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    
    # --- step 4: wrap with message history ---
    # this automatically manages chat_history per session_id
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return chain_with_history, history_store