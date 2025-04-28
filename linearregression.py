import re
import pandas as pd
import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import textstat   

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(
    "subreddit_prolife_file.csv",
    usecols=["article","num_comments"],
    engine="python",
    on_bad_lines="skip"
).dropna()

def clean(text):
    t = re.sub(r"http\S+", "", str(text))
    t = re.sub(r"[^A-Za-z\s\.!?]", "", t)
    return t.lower()

df["clean"] = df["article"].apply(clean)

df["sentences"] = df["clean"].apply(lambda t: [s for s in re.split(r"[\.!?]+", t) if s.strip()])
df["sent_count"] = df["sentences"].apply(len)
df["tokens"]    = df["clean"].str.split()

df["word_count"]   = df["tokens"].apply(len)
df["avg_sent_len"] = df["word_count"] / df["sent_count"].replace(0,1)
df["sent_len_var"] = df["sentences"].apply(lambda ss: np.var([len(s.split()) for s in ss]) if ss else 0)
df["punct_count"]  = df["article"].apply(lambda t: sum(c in "!?.,;:" for c in str(t)))

charged = {"extreme","radical","outrage","hate","violence","freedom","rights"}
df["charged_count"]        = df["tokens"].apply(lambda toks: sum(w in charged for w in toks))
df["mean_charge_per_sent"] = df["charged_count"] / df["sent_count"].replace(0,1)

sid = SentimentIntensityAnalyzer()
df["sentiment"] = df["article"].apply(lambda t: sid.polarity_scores(str(t))["compound"])

docs = list(nlp.pipe(df["clean"], batch_size=100))
df["subj_count"]    = [sum(tok.dep_=="nsubj" for tok in doc) for doc in docs]
df["pred_count"]    = [sum(tok.dep_=="ROOT"  for tok in doc) for doc in docs]
df["advmod_count"]  = [sum(tok.dep_=="advmod" for tok in doc) for doc in docs]
df["amod_count"]    = [sum(tok.dep_=="amod"   for tok in doc) for doc in docs]
df["adj_count"]     = [sum(tok.pos_=="ADJ"    for tok in doc) for doc in docs]

# Readability
# Flesch Reading Ease: higher = easier to read
df["readability"] = df["clean"].apply(lambda t: textstat.flesch_reading_ease(t))

features = [
  "word_count","avg_sent_len","sent_len_var","punct_count",
  "charged_count","mean_charge_per_sent","sentiment",
  "subj_count","pred_count","advmod_count","amod_count","adj_count",
  "readability"
]
X = df[features].values
y = df["num_comments"].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xs, y, random_state=42)

model = LinearRegression().fit(X_train, y_train)
r2 = model.score(X_test, y_test)

print(f"R² on test set: {r2:.4f}")
print("Coefficients:")
for feat,coef in zip(features, model.coef_):
    print(f"  {feat:>20}: {coef:.4f}")
