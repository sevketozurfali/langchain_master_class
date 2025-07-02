import os

from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "How did Juliet die?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold": 0.1},
)

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i};\n{doc.page_content}\n")
    print(f"Soruce: {doc.metadata['source']}\n")

