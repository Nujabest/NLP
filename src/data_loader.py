import os

import numpy as np
import pandas as pd

NM = "non mentionné"
YEARS = [1973, 1978, 1981, 1988, 1993]
TEXT_ROOT = "arkindex_archelec/text_files"


def load_metadata(csv_path: str, years: list = YEARS) -> pd.DataFrame:
    """Charge le CSV, crée la colonne 'parti', filtre par années."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["parti"] = np.select(
        [
            df["titulaire-liste"] != NM,
            df["suppleant-liste"] != NM,
            df["titulaire-soutien"] != NM,
            df["suppleant-soutien"] != NM,
        ],
        [
            df["titulaire-liste"],
            df["suppleant-liste"],
            df["titulaire-soutien"],
            df["suppleant-soutien"],
        ],
        default=NM,
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df[df["date"].dt.year.isin(years)]
    return df


def load_texts(df: pd.DataFrame, text_root: str = TEXT_ROOT) -> pd.DataFrame:
    """Charge les fichiers .txt, joint avec df, filtre les lignes sans texte."""
    texts = {}
    for dirpath, _, files in os.walk(text_root):
        for fname in files:
            if fname.endswith(".txt"):
                doc_id = fname.replace(".txt", "")
                text = open(
                    os.path.join(dirpath, fname), encoding="utf-8", errors="ignore"
                ).read()
                texts[doc_id] = text

    df = df.copy()
    df["text"] = df["id"].map(texts)
    df["text"] = df["text"].str.replace("Sciences Po / fonds CEVIPOF", "", regex=False)
    df = df[df["text"].notna()].copy().reset_index(drop=True)
    df["year"] = df["date"].dt.year
    return df


EMBEDDINGS_ROOT = "data/embeddings"


def compute_and_save_embeddings(
    docs: list,
    name: str,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 64,
    embeddings_root: str = EMBEDDINGS_ROOT,
) -> np.ndarray:
    """Calcule les embeddings avec SentenceTransformer et les sauvegarde en .npy.

    Args:
        docs: Liste de textes à encoder.
        name: Nom du fichier de sauvegarde (sans extension).
        model_name: Modèle SentenceTransformer à utiliser.
        batch_size: Taille des batchs pour l'encodage.
        embeddings_root: Dossier de sauvegarde.

    Returns:
        Les embeddings calculés.
    """
    from sentence_transformers import SentenceTransformer

    os.makedirs(embeddings_root, exist_ok=True)
    out_path = os.path.join(embeddings_root, f"{name}.npy")

    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=batch_size)
    np.save(out_path, embeddings)
    print(f"Embeddings sauvegardés : {out_path}  ({embeddings.shape})")
    return embeddings


def load_embeddings(name: str, embeddings_root: str = EMBEDDINGS_ROOT) -> np.ndarray:
    """Charge des embeddings depuis un fichier .npy.

    Args:
        name: Nom du fichier (sans extension).
        embeddings_root: Dossier source.

    Returns:
        Les embeddings chargés.
    """
    path = os.path.join(embeddings_root, f"{name}.npy")
    return np.load(path)
