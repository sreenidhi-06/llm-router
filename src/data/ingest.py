import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sqlite3
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

SEED = 42
LETTERS = "ABCDE"
TECHNICAL_SUBJECTS = ["machine_learning", "college_computer_science",
                      "high_school_computer_science", "computer_security",
                      "high_school_statistics", "electrical_engineering",
                      "econometrics", "college_mathematics"]
BUSINESS_SUBJECTS = ["business_ethics", "management", "marketing",
                     "professional_accounting", "public_relations",
                     "high_school_microeconomics", "high_school_macroeconomics"]


def read_split(repo, folder, split="test"):
    files = list_repo_files(repo, repo_type="dataset")
    matches = sorted(f for f in files
                     if f.startswith(f"{folder}/{split}") and f.endswith(".parquet"))
    if not matches:
        raise FileNotFoundError(f"No {split} parquet file found for {repo}/{folder}")
    path = hf_hub_download(repo, matches[0], repo_type="dataset")
    return pd.read_parquet(path)


def fmt_options(options):
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def load_mmlu(n=800):
    ds = read_split("cais/mmlu", "all")
    parts = []
    for subjects in (TECHNICAL_SUBJECTS, BUSINESS_SUBJECTS):
        sub = ds[ds["subject"].isin(subjects)]
        parts.append(sub.sample(n=min(n // 2, len(sub)), random_state=SEED))
    ds = pd.concat(parts)
    return pd.DataFrame({
        "source": "mmlu",
        "subject": ds["subject"].values,
        "prompt": [f"{q}\n{fmt_options(c)}" for q, c in zip(ds["question"], ds["choices"])],
        "gold_answer": [LETTERS[int(a)] for a in ds["answer"]],
    })


def load_gsm8k(n=800):
    ds = read_split("openai/gsm8k", "main")
    ds = ds.sample(n=min(n, len(ds)), random_state=SEED)
    gold = ds["answer"].str.split("####").str[-1].str.strip().str.replace(",", "", regex=False)
    return pd.DataFrame({"source": "gsm8k", "subject": "grade_school_math",
                         "prompt": ds["question"].values, "gold_answer": gold.values})


def load_arc(n=800):
    ds = read_split("allenai/ai2_arc", "ARC-Challenge")
    ds = ds.sample(n=min(n, len(ds)), random_state=SEED)
    prompts, golds = [], []
    for _, r in ds.iterrows():
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        prompts.append(r["question"] + "\n" + fmt_options(texts))
        golds.append(LETTERS[labels.index(r["answerKey"])])
    return pd.DataFrame({"source": "arc", "subject": "science",
                         "prompt": prompts, "gold_answer": golds})


def load_custom(path="datasets/custom_questions.csv"):
    if not os.path.exists(path):
        print("No custom questions file found, skipping")
        return pd.DataFrame()
    c =     c = pd.read_csv(path, keep_default_na=False, na_values=[""])
    required = ["domain", "subject", "question", "option_a", "option_b",
                "option_c", "option_d", "answer"]
    missing = [col for col in required if col not in c.columns]
    if missing:
        raise ValueError(f"Custom CSV is missing columns: {missing}")
    c = c.dropna(subset=required)
    c["answer"] = c["answer"].astype(str).str.strip().str.upper()
    bad = c[~c["answer"].isin(list("ABCD"))]
    if len(bad):
        raise ValueError(f"Answer must be A-D. Check spreadsheet rows: {list(bad.index + 2)}")
    prompts = [f"{r.question}\n" + fmt_options([r.option_a, r.option_b, r.option_c, r.option_d])
               for r in c.itertuples()]
    return pd.DataFrame({"source": "custom", "subject": c["subject"].values,
                         "prompt": prompts, "gold_answer": c["answer"].values,
                         "domain": c["domain"].values})


def main():
    df = pd.concat([load_mmlu(), load_gsm8k(), load_arc(), load_custom()], ignore_index=True)
    df = df.dropna(subset=["prompt", "gold_answer"])
    df = df[df["gold_answer"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset="prompt").reset_index(drop=True)
    if "domain" not in df.columns:
        df["domain"] = None
    df["domain"] = df["domain"].fillna(df["subject"].apply(
        lambda s: "technical" if s in TECHNICAL_SUBJECTS
        else "business" if s in BUSINESS_SUBJECTS else "general"))
    df.insert(0, "id", range(1, len(df) + 1))
    df["prompt_chars"] = df["prompt"].str.len()
    df["word_count"] = df["prompt"].str.split().str.len()
    os.makedirs("data", exist_ok=True)
    with sqlite3.connect("data/router.db") as con:
        df.to_sql("queries", con, if_exists="replace", index=False)
    print(df.groupby("source").size())
    print(df.groupby("domain").size())


if __name__ == "__main__":
    main()