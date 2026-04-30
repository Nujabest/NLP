import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


def build_topic_model(
    custom_stopwords: list,
    token_pattern: str = r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]+\b",
    nr_topics: int = 11,
    # UMAP
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    # HDBSCAN
    hdbscan_min_cluster_size: int = 10,
) -> BERTopic:
    vectorizer_model = CountVectorizer(
        stop_words=custom_stopwords,
        token_pattern=token_pattern,
    )
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        metric="cosine",
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics,
        verbose=True,
    )


def fit_topic_model(topic_model: BERTopic, docs: list, embeddings: np.ndarray):
    """Entraîne le modèle et retourne topics, probs."""
    return topic_model.fit_transform(docs, embeddings)


def reduce_outliers(
    topic_model: BERTopic,
    docs: list,
    topics,
    probs,
    embeddings: np.ndarray,
):
    """Réduit les outliers (-1) par stratégie embeddings."""
    return topic_model.reduce_outliers(
        documents=docs,
        topics=topics,
        probabilities=probs,
        embeddings=embeddings,
        strategy="embeddings",
    )
