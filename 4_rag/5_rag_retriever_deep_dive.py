import os
from webbrowser import Chrome

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import TextLoader


current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

def query_vector_store(store_name, query, embedding_function, search_type, search_kwargs):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector store {store_name} ---")
        db_query = Chroma(persist_directory=persistent_directory,
                    embedding_function=embedding_function,
                    )

        retriever = db_query.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

        relevant_docs = retriever.invoke(query)

        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('Source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")



query = "How did Juliet die?"

print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db_with_metadata", query, embeddings, "similarity", {"k":3})

print("\n--- Using Max Marginal Relevance MMR ---")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "mmr",
    {"k":3, "fetch_k":20, "lambda_mult": 0.5},
)

