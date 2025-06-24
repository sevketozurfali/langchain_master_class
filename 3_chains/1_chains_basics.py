from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama.llms import OllamaLLM

load_dotenv()

model = OllamaLLM(model = "llama3:latest")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

chain = prompt_template | model 

result = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result)
