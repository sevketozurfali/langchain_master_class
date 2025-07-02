import os

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file which you added {file_path} not found!!!")

loader = TextLoader(file_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print("\n--- Document chunk information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk: \n {docs[0].page_content}\n")

def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print("\n--- Vector store creating... ---")
        db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persistent_directory)

        print("\n--- Vector store finished. ---")
    else:
        print(f"Vector store {store_name} is already exists.")

ollamaEmbeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
create_vector_store(docs,ollamaEmbeddings, "chroma_db_ollama")

huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)

    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store ---")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_function)
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k":1, "score_threshold": 0.1},
        )

        relevant_docs = retriever.invoke(query)

        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('Source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} doesn't exist.")

query = "Who is Odysseus wife?"
query_vector_store("chroma_db_ollama", query, ollamaEmbeddings)
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

print("Done!")