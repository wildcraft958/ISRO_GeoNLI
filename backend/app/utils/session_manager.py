from typing import Optional, List
from app.orchestrator import app


class SessionManager:
    """Utility for managing LangGraph session state and history"""
    
    def __init__(self, graph_app=None):
        self.app = graph_app or app
    
    def get_session_history(self, session_id: str) -> List[dict]:
        """Retrieve all messages in session"""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.app.get_state(config)
            if state and state.values:
                return state.values.get("messages", [])
            return []
        except Exception:
            return []
    
    def get_session_state(self, session_id: str) -> Optional[dict]:
        """Retrieve full session state"""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.app.get_state(config)
            if state and state.values:
                return state.values
            return None
        except Exception:
            return None
    
    def update_session_context(self, session_id: str, key: str, value):
        """Update session_context dict (e.g., user prefs)
        Note: Updates via next invocation, not direct state manipulation
        """
        # This would be called during next invoke with updated context
        # For now, just a placeholder - actual update happens in orchestrator
        pass

