import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import CUSTOM_STOPWORDS

TOPIC_LABELS = {
    0: "Politique sociale & économique",
    1: "Extrême droite / FN",
    2: "Gauche ouvrière / PCF-LO",
    3: "Écologie radicale (bruit/Alsace?)",
    4: "Documents allemands (Alsace-Moselle)",
    5: "Centre-droit / UDF — identité & sécu.",
    6: "Écologie politique / Les Verts",
    7: "Candidature locale & mandat",
    8: "Gauche réformiste / PCF-PS",
    9: "Anti-patronat / Lutte des classes",
}


def build_nmf(
    df: pd.DataFrame,
    stopwords: list = CUSTOM_STOPWORDS,
    n_topics: int = 10,
    token_pattern: str = r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]+\b",
):
    """Vectorise (TF-IDF) et entraîne un modèle NMF.

    Returns: tfidf_vectorizer, tfidf, nmf, W_normalized, df (avec dominant_topic et topic_score)
    """
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=10,
        max_features=2000,
        stop_words=CUSTOM_STOPWORDS,
        token_pattern=token_pattern,
    )
    tfidf = tfidf_vectorizer.fit_transform(df["lemmatized_text"].fillna(""))
    print(f"Matrice TF-IDF : {tfidf.shape[0]} docs × {tfidf.shape[1]} termes")

    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(tfidf)
    W_normalized = W / W.sum(axis=1, keepdims=True)

    df = df.copy()
    df["dominant_topic"] = np.argmax(W_normalized, axis=1)
    df["topic_score"] = np.max(W_normalized, axis=1)

    return tfidf_vectorizer, tfidf, nmf, W_normalized, df
