from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import SparseVector, Filter, MatchValue, FieldCondition
from sentence_transformers import SentenceTransformer

def custom_embed_function(texts, embedding_model):
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()

def dense_query_search(client: QdrantClient, query_text: str, number_of_results: int = 5, print_results: bool = False):
    print(f"\n--- : Querying Small Chunks for: '{query_text}' ---")
    dense_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

    query_vector = custom_embed_function(embedding_model=dense_embedding_model, texts=[query_text])
    print(len(query_vector[0]))

    results = client.query_points(
        collection_name="wiki_small_chunks",
        query=query_vector[0],
        using="dense",
        limit=number_of_results,
        with_payload=True
    )
    if print_results:
        [print(point.payload["text"].replace("\n", " ")) for point in results.points]
    return results


def sparse_query_search(client: QdrantClient, query_text: str, number_of_results: int = 5, print_results: bool = False):
    print(f"\n--- : Querying Small Chunks for: '{query_text}' ---")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    query_vector = list(sparse_model.embed([query_text]))[0]

    results = client.query_points(
        collection_name="wiki_small_chunks",
        query=SparseVector(
            indices=query_vector.indices,
            values=query_vector.values,
        ),
        using="sparse",
        limit=number_of_results,  # Top 5 matches
    )

    if print_results:
        [print(point.payload["text"].replace("\n", " ")) for point in results.points]
    return results


def get_topics_list(client: QdrantClient, num_topics: int = 10, print_results = False):
    result = client.scroll(
        collection_name="wiki_large_chunks",
        limit=num_topics,
        with_payload=True,
        with_vectors=True
    )
    if(print_results):
        for point in result[0]:
            print(point.payload["title"])
    return [point.payload["title"] for point in result[0]]


def hierarchical_search(client: QdrantClient, query_text: str, use_deep_embedding: bool = True,
                        number_of_results: int = 5, print_results: bool = False):
    if use_deep_embedding:
        child_results = dense_query_search(client, query_text, print_results=False)
    else:
        child_results = sparse_query_search(client, query_text, print_results=False)
    parent_ids = [point.payload["parent_id"] for point in child_results.points]
    results = client.retrieve(
        collection_name="wiki_large_chunks",
        ids=parent_ids,
        with_payload=True
    )
    if print_results:
        [print(point.payload["text"].replace("\n", " ")) for point in results]
    return results
def get_point_with_common_parent_id(client:QdrantClient, collection_name, parent_id, n=10, print_results: bool = False):
    points = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="parent_id", match=MatchValue(value=parent_id))]
        ),
        limit=n,
         with_vectors=True,
        with_payload=True
    )

    if print_results:
        [print(point) for point in points[0]]
    return points

def get_point_with_common_title(client:QdrantClient, collection_name, title, n=10):
    points = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="title", match=MatchValue(value=title))]
        ),
        with_payload=True,
        with_vectors=True,
        limit=n  # adjust as needed
    )
    return points

def get_parent_title(client: QdrantClient, parent_id, collection_name="wiki_large_chunks"):
    result = client.retrieve(
        collection_name="wiki_large_chunks",
        ids=[parent_id],
        with_payload=True
    )
    return result[0].payload["title"]
