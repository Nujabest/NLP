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
