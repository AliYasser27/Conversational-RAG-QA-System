# rag_engine.py - this is the smart RAG engine
# it implements the ChatEngine interface so we can swap it in

from typing import List, Tuple
from .base import ChatEngine


class RagEngine(ChatEngine):
    def __init__(self, rag_chain_with_history, history_store):
        # rag_chain_with_history is the full chain from pipeline.py
        # history_store maps session_id -> ChatMessageHistory
        self.chain = rag_chain_with_history
        self.history_store = history_store
    
    def answer(self, session_id: str, question: str) -> str:
        # invoke the chain with session_id in config
        # RunnableWithMessageHistory handles the history automatically
        resp = self.chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return resp
    
    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        # convert ChatMessageHistory to list of (role, content) tuples
        h = self.history_store.get(session_id)
        if not h:
            return []
        
        # ChatMessageHistory stores messages as objects with .type and .content
        # human = user, ai = assistant
        result = []
        for msg in h.messages:
            role = "user" if msg.type == "human" else "assistant"
            result.append((role, msg.content))
        return result