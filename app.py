# app.py - main Streamlit app with multi-session chat
# each session has its own isolated chat history

import streamlit as st
import os
import tempfile

from engines.base import ChatEngine
from engines.rag_engine import RagEngine
from rag.ingest import load_pdfs, split_docs, create_vector_store, get_vector_store, count_docs
from rag.pipeline import create_rag_chain_with_history


class Chatbot:
    def __init__(self, engine: ChatEngine):
        self.engine = engine
    
    def ask(self, session_id: str, question: str) -> str:
        return self.engine.answer(session_id, question)
    
    def history(self, session_id: str):
        return self.engine.get_history(session_id)


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
    st.header("📚 Conversational RAG Q&A with PDFs")
    
    # --- init session state ---
    if "sessions" not in st.session_state:
        # each session has: messages list + display name
        st.session_state.sessions = {"session-1": {"messages": [], "name": "Chat 1"}}
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = "session-1"
    if "session_counter" not in st.session_state:
        st.session_state.session_counter = 1
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    
    # --- sidebar: sessions list ---
    with st.sidebar:
        st.title("💬 Sessions")
        st.caption("Each session has its own chat history")
        
        # loop through all sessions and show as buttons
        for sid, sdata in st.session_state.sessions.items():
            is_current = (sid == st.session_state.current_session_id)
            
            cols = st.columns([4, 1])
            
            with cols[0]:
                icon = "●" if is_current else "○"
                display_name = sdata["name"][:18] + "..." if len(sdata["name"]) > 18 else sdata["name"]
                msg_count = len(sdata["messages"])
                label = f"{icon} {display_name} ({msg_count})"
                
                if st.button(label, key=f"sel_{sid}", use_container_width=True):
                    # save current session messages before switching
                    st.session_state.sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
                    # switch to clicked session
                    st.session_state.current_session_id = sid
                    # load that session's messages
                    st.session_state.messages = st.session_state.sessions[sid]["messages"]
                    st.rerun()
            
            with cols[1]:
                # only show delete if more than 1 session exists
                if len(st.session_state.sessions) > 1:
                    if st.button("×", key=f"del_{sid}"):
                        # if deleting current session, save its messages first
                        if sid == st.session_state.current_session_id:
                            st.session_state.sessions[sid]["messages"] = st.session_state.messages
                        
                        del st.session_state.sessions[sid]
                        
                        # if we deleted the current session, switch to another one
                        if sid == st.session_state.current_session_id:
                            new_sid = list(st.session_state.sessions.keys())[0]
                            st.session_state.current_session_id = new_sid
                            st.session_state.messages = st.session_state.sessions[new_sid]["messages"]
                        
                        st.rerun()
        
        # + button to create new session
        if st.button("➕ New Chat", key="btn_new_chat", use_container_width=True):
            # save current session before creating new one
            st.session_state.sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
            
            # create new session with unique ID
            st.session_state.session_counter += 1
            new_sid = f"session-{st.session_state.session_counter}"
            st.session_state.sessions[new_sid] = {"messages": [], "name": f"Chat {st.session_state.session_counter}"}
            
            # switch to new empty session
            st.session_state.current_session_id = new_sid
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # --- sidebar: settings ---
        st.title("⚙️ Settings")
        
        # show active session name
        current_name = st.session_state.sessions[st.session_state.current_session_id]["name"]
        st.caption(f"Active: {current_name}")
        
        # API key
        api_key = st.text_input("Groq API Key:", type="password")
        if not api_key:
            st.warning("Enter your Groq API key")
        else:
            st.success("API Key Added Successfully !")
        st.markdown("[Get key here](https://console.groq.com/keys)")
        
        st.divider()
        
        # PDF upload
        st.subheader("📄 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Drop PDFs here",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) ready")
        
        if st.button("🚀 Process PDFs", type="primary"):
            if not api_key:
                st.error("Need API key first")
            elif not uploaded_files:
                st.error("Upload PDFs first")
            else:
                with st.spinner("Processing... this may take a minute"):
                    temp_paths = []
                    for upfile in uploaded_files:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp.write(upfile.read())
                        tmp.close()
                        temp_paths.append(tmp.name)
                    
                    docs = load_pdfs(temp_paths)
                    chunks = split_docs(docs)
                    create_vector_store(chunks)
                    
                    for p in temp_paths:
                        os.remove(p)
                    
                    vector_store = get_vector_store()
                    chain, hist_store = create_rag_chain_with_history(vector_store, api_key)
                    engine = RagEngine(chain, hist_store)
                    st.session_state.chatbot = Chatbot(engine)
                    st.session_state.pdf_processed = True
                    
                    st.success(f"Done! {count_docs(vector_store)} chunks stored")
        
        st.divider()
        
        # clear chat for current session only
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.sessions[st.session_state.current_session_id]["messages"] = []
            # also clear RAG history for this session
            if st.session_state.chatbot:
                sid = st.session_state.current_session_id
                try:
                    from rag.pipeline import history_store
                    if sid in history_store:
                        history_store[sid].clear()
                except:
                    pass
            st.rerun()
    
    # --- main chat area ---
    
    # show which session we're in
    current_name = st.session_state.sessions[st.session_state.current_session_id]["name"]
    st.caption(f"Session: {current_name}")
    
    # display chat messages for current session
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # chat input
    if not st.session_state.pdf_processed:
        st.info("👈 Upload PDFs and click 'Process PDFs' to start")
    else:
        user_input = st.chat_input("Ask something...")
        
        if user_input:
            # show user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # get bot answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    sid = st.session_state.current_session_id
                    bot = st.session_state.chatbot
                    answer = bot.ask(sid, user_input)
                    st.markdown(answer)
            
            # show bot message
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # save messages back to session storage
            # (this keeps them safe when you switch sessions)
            st.session_state.sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages


if __name__ == "__main__":
    main()