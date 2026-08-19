# base.py - this is the interface all engines must follow


from typing import Protocol, List, Tuple


class ChatEngine(Protocol):
    # every engine needs to answer a question for a given session
    def answer(self, session_id: str, question: str) -> str:
        ...
    
    # every engine needs to give back the chat history for a session
    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        ...