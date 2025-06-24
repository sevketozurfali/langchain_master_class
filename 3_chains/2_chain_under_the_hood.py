from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_ollama.llms import OllamaLLM

load_dotenv()

model = OllamaLLM(model = "llama3:latest")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

format_template= RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x)

chain = RunnableSequence(first=format_template, middle=[invoke_model], last=parse_output)

result = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result)
