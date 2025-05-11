import os
import logging
from colorama import Fore
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

#OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
client = OpenAI(api_key=api_key,base_url=base_url)

# 设置日志级别为 ERROR
logging.basicConfig(level=logging.ERROR)

#Set the log level to ERROR
def save_faiss(index, file_path):
    index.save_local(file_path)
    print(f"FAISS index saved at {file_path}")
    print(f"Current distance strategy: {index.distance_strategy}")

#Function to load FAISS
def load_faiss(file_path):
    if os.path.exists(file_path):
        print(f"Loading FAISS index from {file_path}")
        # create an embedding model
        embeddings = OpenAIEmbeddings(model="your_embedding_model")
        return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"FAISS index not found at {file_path}")
    return None
# Function to delete an existing database
def delete_existing_faiss(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted existing FAISS index at {file_path}")

#Update object faiss db
def renew_object_faiss_db():
    #Load the object safety guidelines document
    loader_object = TextLoader(
        file_path = "object_document_file_path",
        encoding = "utf-8"
    )

    document_object = loader_object.load()
    assert len(document_object) == 1

    #Split the document
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    # create an embedding model
    embeddings = OpenAIEmbeddings(model="your_embedding_model")
    # Set file path
    object_faiss_path = "object_faiss_db_path"
    os.makedirs("your_db_folder", exist_ok=True)

    # Previous database path
    object_faiss = r"your_db_folder\object_faiss_db\index.faiss"
    object_pkl = r"your_db_folder\object_faiss_db\index.pkl"

    # Delete the previous database
    delete_existing_faiss(object_faiss)
    delete_existing_faiss(object_pkl)

    #Split the document
    object_all_splits = text_splitter.split_documents(document_object)
    print(f"Number of object safety guideline splits: {len(object_all_splits)}")
    safe_object_rules_db = FAISS.from_documents(object_all_splits, embeddings)
    save_faiss(safe_object_rules_db, object_faiss_path)

    print(Fore.RED + "Object FAISS db has been renewed." + Fore.RESET)

# Update operation faiss db
def renew_operation_faiss_db():
    # Load the operation COHS guidelines document
    loader_operation = TextLoader(
        file_path="operation_document_file_path",
        encoding="utf-8"
    )

    document_operation = loader_operation.load()
    assert len(document_operation) == 1

    #Split the document
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    # Create an embedding model
    embeddings = OpenAIEmbeddings(model="your_embedding_model")
    # Set file path
    operation_faiss_path = "operation_faiss_db_path"
    os.makedirs("your_db_folder", exist_ok=True)
    # Previous database path
    operation_faiss = r"your_db_folder\operation_faiss_db\index.faiss"
    operation_pkl = r"your_db_folder\operation_faiss_db\index.pkl"
    # Delete the previous database
    delete_existing_faiss(operation_faiss)
    delete_existing_faiss(operation_pkl)
    # Split the document
    operation_all_splits = text_splitter.split_documents(document_operation)
    print(f"Number of operation safety guideline splits: {len(operation_all_splits)}")
    safe_operation_rules_db = FAISS.from_documents(operation_all_splits, embeddings)
    save_faiss(safe_operation_rules_db, operation_faiss_path)

    print(Fore.RED + "Operation FAISS db has been renewed." + Fore.RESET)

if __name__ == "__main__":
    renew_object_faiss_db()
    renew_operation_faiss_db()
    # Set the knowledge base path
    object_faiss_path = r"object_faiss_db_path"
    operation_faiss_path = r"operation_faiss_db_path"

    os.makedirs("your_db_folder", exist_ok=True)

    # Load the FAISS database
    safe_object_rules_db = load_faiss(object_faiss_path)  # object safety guidelines
    safe_operation_rules_db = load_faiss(operation_faiss_path)  # operation safety guidelines

    print(safe_object_rules_db.distance_strategy)
    print(safe_operation_rules_db.distance_strategy)