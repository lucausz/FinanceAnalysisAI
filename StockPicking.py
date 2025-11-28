import numpy as np
import pandas as pd
import yfinance as yf

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report


# -----------------------------------------------------
# 1. Charger le dataset HuggingFace et préparer le texte
# -----------------------------------------------------

print("Chargement du dataset SP500-EDGAR-10K...")
ds = load_dataset("jlohding/sp500-edgar-10k", split="train")


def add_fields(row):
    # Texte : concat Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A)
    parts = []
    for key in ["item_1", "item_1A", "item_7"]:
        txt = row.get(key)
        if txt is not None:
            parts.append(txt)
    row["full_text"] = "\n\n".join(parts)

    # Année du filing
    row["year"] = pd.to_datetime(row["date"]).year
    return row


ds = ds.map(add_fields)

# On garde seulement les lignes avec un texte suffisant
ds = ds.filter(lambda x: x["full_text"] is not None and len(x["full_text"]) > 200)

# Conversion en DataFrame pandas
df = pd.DataFrame(ds)
df["date"] = pd.to_datetime(df["date"])

# On garde uniquement les lignes avec un retour 252 jours non NaN
df = df.dropna(subset=["252_day_return"])

print(f"Nombre de filings après filtrage texte+return : {len(df)}")


# -----------------------------------------------------
# 2. Récupérer le S&P 500 et calculer le retour 1 an futur
# -----------------------------------------------------

print("Téléchargement de l'historique du S&P 500 (^GSPC)...")
raw_spx = yf.download("^GSPC", start="2000-01-01", progress=False)

if isinstance(raw_spx.columns, pd.MultiIndex):
    # Nouveau yfinance : colonnes MultiIndex (field, ticker)
    level0 = raw_spx.columns.get_level_values(0)
    if "Adj Close" in level0:
        price = raw_spx["Adj Close"]["^GSPC"]
    else:
        price = raw_spx["Close"]["^GSPC"]
else:
    # Colonnes simples
    if "Adj Close" in raw_spx.columns:
        price = raw_spx["Adj Close"]
    else:
        price = raw_spx["Close"]

spx = price.to_frame("spx_close")

# Retour 1 an futur de l'indice : prix(t+252) / prix(t)
spx["spx_fwd_252"] = spx["spx_close"].shift(-252) / spx["spx_close"]
spx = spx.dropna(subset=["spx_fwd_252"])

# Préparer pour merge_asof
spx = spx.reset_index().rename(columns={"Date": "date"})
spx["date"] = pd.to_datetime(spx["date"])


# -----------------------------------------------------
# 3. Aligner chaque 10-K avec le S&P 500 et construire le label
# -----------------------------------------------------

df = df.sort_values("date")
spx = spx.sort_values("date")

print("Merge des 10-K avec le S&P 500 (retour 1 an futur)...")
df = pd.merge_asof(
    df,
    spx[["date", "spx_fwd_252"]],
    on="date",
    direction="backward"
)

# On enlève les lignes sans retour SPX
df = df.dropna(subset=["spx_fwd_252"])

# Surperformance vs S&P 500
df["excess_252"] = df["252_day_return"] / df["spx_fwd_252"]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["excess_252"])

# Label : 1 si l'action bat le S&P 500, 0 sinon
df["y"] = (df["excess_252"] > 1.0).astype(int)

print("Répartition des labels (1 = bat le S&P 500) :")
print(df["y"].value_counts(normalize=True))


# -----------------------------------------------------
# 4. Créer des features supplémentaires (numériques)
# -----------------------------------------------------

print("Création des features numériques...")

# Longueur du texte
df["len_text"] = df["full_text"].str.len()
df["n_words"] = df["full_text"].str.split().str.len()

# Log de la market cap
df["log_mkt_cap"] = np.log(df["mkt_cap"].where(df["mkt_cap"] > 0))

# Flag Covid (2020-2021)
df["covid_flag"] = df["year"].between(2020, 2021).astype(int)

# Lexique simple risque / positif
risk_words = ["risk", "uncertain", "uncertainty", "volatility", "downturn", "adverse", "decline"]
pos_words  = ["growth", "opportunity", "strong", "improve", "increase", "expansion", "record"]


def count_words(text, vocab):
    t = text.lower()
    return sum(t.count(w) for w in vocab)


df["risk_count"] = df["full_text"].apply(lambda t: count_words(t, risk_words))
df["pos_count"]  = df["full_text"].apply(lambda t: count_words(t, pos_words))

df["risk_ratio"] = df["risk_count"] / (df["n_words"] + 1)
df["pos_ratio"]  = df["pos_count"]  / (df["n_words"] + 1)

# On choisit les colonnes numériques à utiliser
num_cols = [
    "log_mkt_cap",
    "len_text",
    "n_words",
    "sic",
    "covid_flag",
    "risk_ratio",
    "pos_ratio",
]

X_num = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


# -----------------------------------------------------
# 5. Split temporel Train / Test
# -----------------------------------------------------

# Exemple : train <= 2015, test >= 2016
train_mask = df["year"] <= 2015
test_mask  = df["year"] >= 2016

train_df = df[train_mask].copy()
test_df  = df[test_mask].copy()

X_num_train = X_num[train_mask].values
X_num_test  = X_num[test_mask].values

y_train = train_df["y"].values
y_test  = test_df["y"].values

print(f"Train : {len(train_df)} filings, Test : {len(test_df)} filings")

print(f"Train : {len(train_df)} filings, Test : {len(test_df)} filings")


# -----------------------------------------------------
# 6. Embeddings du texte (Item 1 + 1A + 7) avec MiniLM
# -----------------------------------------------------

print("Chargement du modèle d'embeddings (MiniLM)...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Encodage des textes (train)...")
X_train_text = train_df["full_text"].tolist()
emb_train = embedder.encode(X_train_text, batch_size=16, show_progress_bar=True)

print("Encodage des textes (test)...")
X_test_text = test_df["full_text"].tolist()
emb_test  = embedder.encode(X_test_text, batch_size=16, show_progress_bar=True)


# -----------------------------------------------------
# 7. Combiner embeddings + features numériques
# -----------------------------------------------------

print("Standardisation des features numériques et concaténation...")

scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled  = scaler.transform(X_num_test)

X_train_all = np.hstack([emb_train, X_num_train_scaled])
X_test_all  = np.hstack([emb_test,  X_num_test_scaled])


# -----------------------------------------------------
# 8. Modèle 1 : régression logistique sur (embeddings + num)
# -----------------------------------------------------

print("Entraînement de la régression logistique...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_all, y_train)

y_proba_lr = clf.predict_proba(X_test_all)[:, 1]
y_pred_lr  = clf.predict(X_test_all)

print("\n=== Résultats LogReg (surperformance vs S&P 500, 1 an) ===")
print("AUC :", roc_auc_score(y_test, y_proba_lr))
print(classification_report(y_test, y_pred_lr))


# -----------------------------------------------------
# 9. Modèle 2 : RandomForest sur (embeddings + num)
# -----------------------------------------------------

"""print("Entraînement du RandomForest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=0,
)
rf.fit(X_train_all, y_train)

y_proba_rf = rf.predict_proba(X_test_all)[:, 1]
y_pred_rf  = rf.predict(X_test_all)

print("\n=== Résultats RandomForest (surperformance vs S&P 500, 1 an) ===")
print("AUC :", roc_auc_score(y_test, y_proba_rf))
print(classification_report(y_test, y_pred_rf))"""


# -----------------------------------------------------
# 10. Analyse "trading" : top 20 % des scores vs le reste (LogReg)
#      avec clipping des outliers sur excess_252
# -----------------------------------------------------

test_df = test_df.reset_index(drop=True)
test_df["y_proba_lr"] = y_proba_lr

# Clipper les outliers de performance pour éviter les moyennes explosives
test_df["excess_252_clipped"] = test_df["excess_252"].clip(lower=0.2, upper=5.0)

q = test_df["y_proba_lr"].quantile(0.8)
top = test_df[test_df["y_proba_lr"] >= q]
rest = test_df[test_df["y_proba_lr"] < q]

print("\n=== Analyse portefeuille (excess_252_clipped, LogReg) ===")
print(f"Nb obs test total : {len(test_df)}")
print(f"Nb obs TOP 20% : {len(top)}, RESTE : {len(rest)}")

print("Perf moyenne 1 an vs S&P 500 (excess_252_clipped) :")
print(" - TOP 20% :", top["excess_252_clipped"].mean())
print(" - RESTE   :", rest["excess_252_clipped"].mean())
print(" - TOUT    :", test_df["excess_252_clipped"].mean())
