import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from  dotenv import load_dotenv


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

HISTORY_FILE = "chat_history.json"

def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history_data = json.load(file)
            return [
                SystemMessage(content=msg["content"]) if msg["type"] == "system" else
                HumanMessage(content=msg["content"]) if msg["type"] == "human" else
                AIMessage(content=msg["content"]) for msg in history_data
            ]
    except (FileNotFoundError, json.JSONDecodeError):
        return [SystemMessage(content="You are a helpful assistant.")]

def save_history(history):
    history_data = [
        {"type": "system", "content": msg.content} if isinstance(msg, SystemMessage) else
        {"type": "human", "content": msg.content} if isinstance(msg, HumanMessage) else
        {"type": "ai", "content": msg.content} for msg in history
    ]
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history_data, file, ensure_ascii=False, indent=4)

def chat():
    history = load_history()
    while True:
        message = input("You: ")
        if message.lower() == "exit":
            save_history(history)
            print("Chat history saved. Exiting...")
            break
        history.append(HumanMessage(content=message))
        response = llm.invoke(history)
        content = response.content
        history.append(AIMessage(content=content))
        print("AI:", content)

if __name__ == "__main__":
    chat()