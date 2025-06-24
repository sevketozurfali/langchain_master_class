#This program runs with local model which the model runs on the "localhost:11434" address.

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

#Load environments from .env file
load_dotenv()

template = """Question: {question}
Answer: Let's think step by step"""

model = OllamaLLM(model = "llama3:latest")

ChatPromptTemplate.from_template(template)

response = model.invoke("what is 81 divided by 9")
print(response)


