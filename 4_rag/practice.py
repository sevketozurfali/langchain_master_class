import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db", "chroma_db_practice")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"{file_path} file not found"
    )

loader = TextLoader(file_path)
document = loader.load()

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

def create_vectore_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vectore store {store_name} ---")
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store")
    else:
        print(f"The db directory {store_name} already exists!!!")

print("\n--- Using Character Based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(document)
create_vectore_store(char_docs, "chroma_db_practice")

def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

        retriever = db.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k": 2, "score_threshold": 0.2}
        )

        relevant_docs = retriever.invoke(query)

        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('Source', 'Unknown')}\n")

    else:
        print(f"Vector store {store_name} does exist.")

query = "How did Juliet die?"

query_vector_store("chroma_db_practice", query)