chat_history = []

def add_to_memory(user, bot):
    chat_history.append({"user": user, "bot": bot})

def get_memory():
    return chat_history[-5:]