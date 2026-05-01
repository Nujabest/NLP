import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def build_party_profiles(distr_df: pd.DataFrame):
    """Calcule le profil thématique moyen par parti.

    Returns: party_profiles, dominant_topic, topic_cols
    """
    topic_cols = [
        c for c in distr_df.columns
        if c != "parti" and pd.api.types.is_numeric_dtype(distr_df[c])
    ]
    party_profiles = distr_df.groupby("parti")[topic_cols].mean()
    dominant_topic = party_profiles.idxmax(axis=1)
    return party_profiles, dominant_topic, topic_cols


def pca_coords(party_profiles: pd.DataFrame, dominant_topic: pd.Series, n_components: int = 2):
    """Réduit les profils en n_components dimensions (PCA).

    Returns: coords, pca, colors, color_map, unique_topics
    """
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(party_profiles.values)
    unique_topics = dominant_topic.unique()
    palette = plt.cm.tab10(np.linspace(0, 1, len(unique_topics)))
    color_map = dict(zip(unique_topics, palette))
    colors = [color_map[t] for t in dominant_topic]
    return coords, pca, colors, color_map, unique_topics

def build_analysis_df(distr_df: pd.DataFrame, df: pd.DataFrame):
    """Joint les distributions avec les métadonnées candidats.

    Returns: analysis_df, topic_cols
    """
    meta_cols = [
        "year", "titulaire-sexe", "titulaire-age-tranche",
        "titulaire-mandat-passe", "departement-nom",
    ]
    analysis_df = distr_df.copy()
    for col in meta_cols:
        analysis_df[col] = df[col].values
    topic_cols = [
        c for c in distr_df.columns
        if c != "parti" and pd.api.types.is_numeric_dtype(distr_df[c])
    ]
    return analysis_df, topic_cols
