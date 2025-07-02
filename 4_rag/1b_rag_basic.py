import os

from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "Who is Jove"

retriever = db.as_retriever(search_type="similarity_score_threshold",
                            search_kwargs={"k": 1, "score_threshold": 0.3}
                            )

relevant_docs = retriever.invoke(query)

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
