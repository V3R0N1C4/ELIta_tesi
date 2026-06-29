import pandas as pd

for fname in ["posts_Italia_multi.csv", "comments_Italia_multi.csv", "corpus_Italia_multi.csv"]:
    df = pd.read_csv(fname)
    if "author" in df.columns:
        df["author"] = "[anonymous]"
    df.to_csv(fname, index=False)