from qdrant_client import QdrantClient, models, AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, PointStruct


class QdrantConfig:
    host = "localhost"
    port = 6333
    timeout = 15

    def __init__(self, host="localhost", port=6333, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout

    def get_config(self):
        return [self.host, self.port, self.timeout]


def check_qdrant_status(config: QdrantConfig):
    host, port, timeout = config.get_config()
    try:
        client = QdrantClient(host=host, port=port, timeout=timeout)
        response = client.get_collections()
        print("Connected. Collections:", [c.name for c in response.collections])
        return True, client
    except ConnectionRefusedError:
        print("Cannot connect. Qdrant not running.")
        return False, None
    except UnexpectedResponse as e:
        print("Qdrant responded but error occurred:", e)
        return False, None
    except Exception as e:
        print("Unexpected error:", e)
        return False, None


def delete_all_existing_collections(config: QdrantConfig):
    host, port, timeout = config.get_config()
    client = QdrantClient(host=host, port=port, timeout=timeout)

    collections = client.get_collections().collections
    for c in collections:
        name = c.name
        client.delete_collection(name)
        print(f"Deleted collection: {name}")

    print("All collections deleted.")


def create_collection_with_embeddings(client: QdrantClient, collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,  # Size of dense embeddings from all-MiniLM-L6-v2
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier="idf"  # Enable IDF for sparse vectors
            )
        }
    )

def create_collection_without_embeddings(client: QdrantClient, collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=2, distance="Cosine")  # size = embedding dim
    )

def get_async_client(qdrantConfig:QdrantConfig):
    return AsyncQdrantClient(host = qdrantConfig.host, port = qdrantConfig.port, timeout=qdrantConfig.timeout)