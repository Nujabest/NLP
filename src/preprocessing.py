import os
import unicodedata

import pandas as pd

TOKEN_PATTERN = r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ]+\b"

STOPWORDS = [x.strip() for x in open("data/stop_word_fr.txt").readlines()]

CUSTOM_STOPWORDS = STOPWORDS + [
    "politique", "election", "electoral",
    "candidat", "candidature", "circonscription",
    "depute", "assemblee", "nationale",
    "vote", "voter", "scrutin",
    "parti", "programme", "campagne",
    "janvier", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "aout", "septembre", "octobre", "novembre", "decembre",
    "france", "francais",
    # Mots allemands
    "und", "der", "die", "das", "ist", "zu", "den", "von", "mit",
    "sich", "des", "auf", "fur", "an", "im", "nicht", "ein", "eine", "auch",
    "als", "bei", "nach", "war", "werden", "aber", "aus", "hat",
    "dass", "sie", "wird", "noch", "wie", "einem", "eines", "wir", "so", "durch",
    "sollen", "mehr", "pouvoir", "national",
    # Partis et sigles politiques
    "ps", "pcf", "pc", "rpr", "udf", "fn", "fn", "rpf", "mds", "mrg", "psg",
    "mrc", "lcr", "lrc", "lutte", "ouvriere", "lo", "pse", "prt", "ptb",
    "socialiste", "communiste", "republicain", "republicaine",
    "gaulliste", "centriste", "rassemblement", "union",
    # Noms de personnalités politiques
    "mitterrand", "chirac", "giscard", "barre", "jospin", "rocard",
    "marchais", "fabius", "mauroy", "beregovoy", "balladur",
    # Formules rhétoriques de profession de foi
    "cher", "madame", "monsieur", "electeur", "electrice",
    "honneur", "confiance", "soutien", "appel", "demande",
    "present", "futur", "avenir", "ensemble", "engagement",
    "liste", "tete", "suppléant", "suppleant", "titulaire",
    "gauche", "ecologiste", "pompidou", "lepen", 'jean', "pierre", "michel", 'um'
    "francois", 'leben', 'order', 'in', 'man', 'georges', 'zeit', 'wenn', 'acker'
]


def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")


def lemmatize(
    df: pd.DataFrame,
    save_path: str = "data/lemmatized.csv",
    batch_size: int = 2000,
) -> pd.DataFrame:
    """Lemmatise les textes avec spaCy, avec sauvegarde incrémentale."""
    import spacy

    if os.path.exists(save_path):
        done = pd.read_csv(save_path)
        done_ids = set(done["id"])
        print(f"{len(done)} textes déjà lemmatisés, {len(df) - len(done)} restants")
    else:
        done = pd.DataFrame(columns=["id", "lemmatized_text"])
        done_ids = set()
        print("Aucune sauvegarde trouvée, on repart de zéro")

    remaining = df[~df["id"].isin(done_ids)][["id", "text"]].reset_index(drop=True)

    if len(remaining) > 0:
        nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
        results = []
        for i, doc in enumerate(nlp.pipe(remaining["text"], batch_size=64)):
            results.append({
                "id": remaining.loc[i, "id"],
                "lemmatized_text": " ".join(
                    [strip_accents(token.lemma_) for token in doc]
                ),
            })
            if (i + 1) % batch_size == 0:
                done = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
                done.to_csv(save_path, index=False)
                results = []
                print(f"  {i + 1}/{len(remaining)} sauvegardés")
        if results:
            done = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
            done.to_csv(save_path, index=False)
            print(f"  {len(remaining)}/{len(remaining)} sauvegardés")

    return df.merge(done[["id", "lemmatized_text"]], on="id", how="left")
