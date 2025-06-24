#This program runs with local model which the model runs on the "localhost:11434" address.

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema import AIMessage
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_firestore import FirestoreLoader
from google.cloud import firestore


#Load environments from .env file
load_dotenv()
PROJECT_ID="chathistory-aa2f7"
SESSION_ID="user-1"
COLLECTION_NAME="chat_history"

print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)


print("Initializing Firestore Chat Message History")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

print("Chat History initialized.")
# print("Current chat history :", chat_history.messages)



template = """Question: {question}
Answer: Let's think step by step"""

ChatPromptTemplate.from_template(template)

model = OllamaLLM(model = "llama3:latest")

# chat_history = []

while True:
    query = input("You : ")
    if query.lower() == "exit":
        break
    chat_history.add_user_message(HumanMessage(content=query))

    response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(AIMessage(content=response))

    print(f"AI : {response}")

print("----Message History----")
print(chat_history)