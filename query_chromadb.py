import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import sys
#from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def retrieve_docs(chroma_db_path, model_name, user_query):
    # Initialize the embedding function with the same model used during indexing
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    #embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    
    # Load Chroma DB from disk
    db = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_function)
    
    # Debug: Print status of loading Chroma DB
    print(f"Chroma DB loaded from: {chroma_db_path}")

    # Perform similarity search
    docs = db.similarity_search(user_query, k=10)
    
    # Debug: Print the number of documents retrieved
    print(f"Number of documents retrieved: {len(docs)}")
    
    return docs

def main(chroma_db_path=None, model_name=None, user_query=None):
    parser = argparse.ArgumentParser(description="Retrieve documents from Chroma DB based on user query.")
    parser.add_argument("--chroma_db_path", type=str, help="Path to the Chroma DB directory.", default=chroma_db_path)
    parser.add_argument("--model_name", type=str, help="Embedding model name.", default=model_name)
    parser.add_argument("--user_query", type=str, help="User query.", default=user_query)
    
    args = parser.parse_args()
    
    # Debug: Print the input arguments
    print(f"Chroma DB Path: {args.chroma_db_path}")
    print(f"Model Name: {args.model_name}")
    print(f"User Query: {args.user_query}")
    
    # Validate the provided arguments
    if not args.chroma_db_path or not args.model_name or not args.user_query:
        print("Error: All arguments (chroma_db_path, model_name, user_query) must be provided.")
        sys.exit(1)
    
    # Retrieve documents
    docs = retrieve_docs(args.chroma_db_path, args.model_name, args.user_query)
    
    # Print retrieved documents
    for i, doc in enumerate(docs):
        # Debug: Print the document number and content
        print(f"Document {i+1}:\n{doc.page_content}\n")

if __name__ == "__main__":
    # Provide default values here if needed
    chroma_db_path = "/mnt/tier2/project/p200475/crewai_app/app_scripts/mychroma_db"  # Default path to Chroma DB
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Ensure this matches the model used during indexing
    user_query = "Tell me about Orbit Fab's fuel port"  # Default user query
    
    main(chroma_db_path, model_name, user_query)
