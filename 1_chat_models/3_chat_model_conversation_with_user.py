#This program runs with local model which the model runs on the "localhost:11434" address.

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema import AIMessage

#Load environments from .env file
load_dotenv()

template = """Question: {question}
Answer: Let's think step by step"""

ChatPromptTemplate.from_template(template)

model = OllamaLLM(model = "llama3:latest")

chat_history = []

while True:
    query = input("You : ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response))

    print(f"AI : {response}")

print("----Message History----")
print(chat_history)