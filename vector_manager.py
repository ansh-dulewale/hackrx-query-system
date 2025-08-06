import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Initialize Clients ---
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
except TypeError:
    print("API keys not found. Make sure you have a .env file with OPENAI_API_KEY and PINECONE_API_KEY")
    exit()

INDEX_NAME = "hackrx-index"

def setup_pinecone():
    """
    Sets up the Pinecone index, creating it if it doesn't exist.
    """
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Dimension for OpenAI's text-embedding-3-small
            metric="cosine", # Cosine similarity is great for text
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
    return pc.Index(INDEX_NAME)

def embed_and_store(chunks: list[str], index):
    """
    Generates embeddings for text chunks and stores them in Pinecone.
    
    Args:
        chunks: A list of text chunks from the document.
        index: The Pinecone index object.
    """
    print(f"Embedding {len(chunks)} chunks...")
    
    # We use the newer, more cost-effective embedding model
    response = openai_client.embeddings.create(input=chunks, model="text-embedding-3-small")
    
    vectors_to_upsert = []
    for i, (chunk, embedding_data) in enumerate(zip(chunks, response.data)):
        vector = {
            "id": f"vec-{i}",
            "values": embedding_data.embedding,
            "metadata": {"text": chunk} # Store the original text as metadata
        }
        vectors_to_upsert.append(vector)
    
    # Upsert in batches to be efficient
    index.upsert(vectors=vectors_to_upsert, namespace="doc-namespace")
    print("Embeddings stored successfully in Pinecone.")

def query_vector_db(query: str, index) -> str:
    """
    Queries the vector database to find the most relevant text chunks.

    Args:
        query: The user's question.
        index: The Pinecone index object.

    Returns:
        A string containing the combined text of the most relevant chunks.
    """
    query_embedding = openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=3, # Retrieve the top 3 most relevant chunks
        namespace="doc-namespace",
        include_metadata=True # Ensure we get the text back
    )
    
    # Combine the text from the top results to form the context
    context = " ".join([match['metadata']['text'] for match in results['matches']])
    return context

# --- Main execution block for testing ---
if __name__ == '__main__':
    from document_processor import process_pdf_from_url

    pinecone_index = setup_pinecone()
    
    # 1. Process a document
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv-2023-01-03&st="
    text_chunks = process_pdf_from_url(test_url)
    
    # 2. Embed and store it
    if text_chunks:
        embed_and_store(text_chunks, pinecone_index)
        
        # 3. Query it
        test_question = "What is the grace period for premium payment?"
        relevant_context = query_vector_db(test_question, pinecone_index)
        
        print(f"\n--- Context for question: '{test_question}' ---")
        print(relevant_context)
