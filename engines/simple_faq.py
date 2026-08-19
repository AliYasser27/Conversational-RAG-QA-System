# simple_faq.py - this is the old dumb bot from project 1
# we keep it so we can swap between simple FAQ and RAG

from typing import List, Tuple
from .base import ChatEngine


class SimpleFAQEngine(ChatEngine):
    def __init__(self, faq_map: dict):
        # faq_map is just a dict like {"what is ai?": "AI is..."}
        self.faq_map = faq_map
        # store history per session
        self.histories = {}
    
    def answer(self, session_id: str, question: str) -> str:
        # look up answer, case insensitive
        ans = self.faq_map.get(question.lower(), "Sorry, I don't know.")
        
        # save to history
        if session_id not in self.histories:
            self.histories[session_id] = []
        
        self.histories[session_id].append(("user", question))
        self.histories[session_id].append(("assistant", ans))
        
        return ans
    
    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        return self.histories.get(session_id, [])