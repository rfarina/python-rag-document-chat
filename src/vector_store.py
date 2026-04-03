import pinecone, openai
from pinecone import Pinecone, ServerlessSpec 
from langchain_community.embeddings import OpenAIEmbeddings

from src.config import pinecone_api_key, pinecone_environment

# Define constants
index_name = "knowledge-docs"
embedding_model = "text-embedding-ada-002"

pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment, spec={"index_name": index_name})
index = pc.Index(index_name)



def init_pinecone():
    """Initialize Pinecone and ensure the index exists."""
    print("In init_pinecone")
    
    try:
        if index_name not in pc.list_indexes():
            pc.create_index(index_name, dimension=1536, metric="cosine",       
                spec=ServerlessSpec(
                    cloud='aws', 
                    region=pinecone_environment
                ) 
            )
            print(f"Created Pinecone index {index_name}")
        else:
            print(f"Pinecone index {index_name} already exists")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise  # Re-raise the exception to stop the execution


def upsert_documents(docs: list, index_name="knowledge-docs", namespace="default"):
    """Embed and upsert documents into the Pinecone vector store."""
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    
    # Upsert documents
    vectors = []
    for doc in docs:
        embedding = embeddings_model.embed_query(doc["content"])
        vectors.append({"id": doc["id"], "values": embedding, "metadata": {"content": doc["content"]}})
    index.upsert(vectors=vectors, namespace=namespace)
    print(f"Upserted {len(docs)} documents.")

def query_vector_store(query: str, top_k: int = 5) -> str:
    """
    Query the vector store to find the most relevant context for a given query.
    
    Args:
        query (str): The user query to search for.
        top_k (int): Number of top matches to return.

    Returns:
        str: Combined content of the most relevant matches.
    """


    try:
        # Generate embedding for the query
        print("Querying vector store...")
        embeddings = OpenAIEmbeddings(model=embedding_model)
        query_embedding = embeddings.embed_query(query)

        print("Querying Pinecone index...")
        # Query the vector store
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default"
        )

        print("Processing query results...")
        # Extract content from metadata and return as context
        context = "\n".join([match["metadata"]["content"] for match in results["matches"]])
        return context

    except Exception as e:
        print(f"Error querying vector store: {e}")
        return "Error querying vector store."

