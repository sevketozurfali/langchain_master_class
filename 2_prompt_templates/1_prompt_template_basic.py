# from langchain.chains.summarize.stuff_prompt import prompt_template
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("----Prompt from Template-----")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)

