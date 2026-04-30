import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocessing import CUSTOM_STOPWORDS


def build_lda(
    df: pd.DataFrame,
    stopwords: list = CUSTOM_STOPWORDS,
    n_topics: int = 10,
    n_features: int = 1000,
    token_pattern: str = r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]+\b",
):
    """Vectorise (CountVectorizer) et entraîne un modèle LDA.

    Returns: tf_vectorizer, tf, lda
    """
    tf_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=5,
        max_features=n_features,
        stop_words=CUSTOM_STOPWORDS,
        token_pattern=token_pattern,
    )
    tf = tf_vectorizer.fit_transform(df["lemmatized_text"].fillna(""))
    print(f"Matrice DTM : {tf.shape[0]} docs × {tf.shape[1]} termes")

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )
    lda.fit(tf)
    return tf_vectorizer, tf, lda


def plot_top_words(model, vectorizer, n_top_words: int, title: str, n_cols: int = 5, save_path: str = None):
    """Affiche les mots les plus importants par topic (LDA ou NMF)."""
    feature_names = vectorizer.get_feature_names_out()
    n_topics = len(model.components_)
    nb_lines = math.ceil(n_topics / n_cols)
    fig, axes = plt.subplots(nb_lines, n_cols, figsize=(30, nb_lines * 6), sharex=False)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 20})
        ax.tick_params(axis="both", which="major", labelsize=14)
        for spine in "top right left".split():
            ax.spines[spine].set_visible(False)
    for idx in range(n_topics, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(title, fontsize=30)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
