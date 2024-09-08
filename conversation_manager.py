class ConversationManager:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_conversation_history(self):
        return self.history

    def clear_history(self):
        self.history = []
