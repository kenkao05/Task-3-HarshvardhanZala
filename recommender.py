"""
Project 3: AI Recommendation Logic
Tech Stack Recommender — DecodeLabs Batch 2026

Pipeline:
  1. Ingestion   — collect user skills (min 3)
  2. Scoring     — TF-IDF vectorization + Cosine Similarity
  3. Sorting     — rank all job roles by similarity score
  4. Filtering   — return Top-N results
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────
# STEP 0: Load dataset
# ──────────────────────────────────────────────────────────────
def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["skills"] = df["skills"].str.lower().str.strip()
    return df


# ──────────────────────────────────────────────────────────────
# STEP 1: Ingestion — capture user state
# ──────────────────────────────────────────────────────────────
def get_user_input() -> list[str]:
    print("\n" + "=" * 55)
    print("   TECH STACK RECOMMENDER — DecodeLabs AI Intern")
    print("=" * 55)
    print("\nEnter your skills one by one (minimum 3).")
    print("Type 'done' when finished.\n")

    skills = []
    while True:
        skill = input(f"  Skill #{len(skills) + 1}: ").strip().lower()

        if skill == "done":
            if len(skills) < 3:
                print(f"  ⚠  Need at least 3 skills. You have {len(skills)}.")
                continue
            break
        elif skill == "":
            print("  ⚠  Empty input ignored.")
            continue
        elif skill in skills:
            print(f"  ⚠  '{skill}' already added.")
            continue
        else:
            skills.append(skill)
            print(f"  ✓  Added: {skill}")

    return skills


# ──────────────────────────────────────────────────────────────
# STEP 2: Scoring — TF-IDF vectorization + Cosine Similarity
# ──────────────────────────────────────────────────────────────
def build_tfidf_matrix(df: pd.DataFrame) -> tuple:
    """
    Fits a TF-IDF vectorizer on the job role skill corpus.
    Returns the fitted vectorizer and the document-term matrix.
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-zA-Z0-9_]+",   # handles underscore-joined terms
        sublinear_tf=True                   # apply log(1 + tf) to dampen freq
    )
    tfidf_matrix = vectorizer.fit_transform(df["skills"])
    return vectorizer, tfidf_matrix


def score_user_profile(
    user_skills: list[str],
    vectorizer: TfidfVectorizer,
    tfidf_matrix
) -> np.ndarray:
    """
    Transforms the user's skill list into a TF-IDF vector
    using the SAME vocabulary as the job role corpus,
    then computes cosine similarity against every role.
    """
    user_string = " ".join(user_skills)
    user_vector = vectorizer.transform([user_string])
    scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    return scores


# ──────────────────────────────────────────────────────────────
# STEPS 3 & 4: Sorting + Filtering — produce Top-N list
# ──────────────────────────────────────────────────────────────
def get_top_recommendations(
    df: pd.DataFrame,
    scores: np.ndarray,
    top_n: int = 3
) -> pd.DataFrame:
    df = df.copy()
    df["similarity_score"] = scores

    # Sort descending, then slice Top-N
    results = (
        df.sort_values("similarity_score", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )
    results.index += 1   # 1-indexed ranks
    return results


# ──────────────────────────────────────────────────────────────
# Output display
# ──────────────────────────────────────────────────────────────
def display_results(
    results: pd.DataFrame,
    user_skills: list[str]
) -> None:
    print("\n" + "=" * 55)
    print("   TOP CAREER RECOMMENDATIONS")
    print("=" * 55)
    print(f"\n  Your skills: {', '.join(user_skills)}\n")

    for rank, row in results.iterrows():
        score_pct = row["similarity_score"] * 100
        bar_len = int(score_pct / 5)          # 20 chars max
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"  #{rank}  {row['job_role']}")
        print(f"      Match: [{bar}] {score_pct:.1f}%")
        print(f"      Key skills: {row['skills'][:60]}...")
        print()

    if results["similarity_score"].iloc[0] == 0:
        print("  ⚠  No matches found. Your skills don't overlap")
        print("     with the dataset vocabulary. Try different terms.\n")
    else:
        print("=" * 55)
        print("  Tip: Add more specific skills for tighter matches.")
        print("=" * 55)


# ──────────────────────────────────────────────────────────────
# COLD START GUARD
# ──────────────────────────────────────────────────────────────
def handle_cold_start(scores: np.ndarray) -> bool:
    """
    Detects the cold start condition:
    user profile vector is all zeros (no vocab overlap).
    """
    return np.all(scores == 0)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    DATASET_PATH = "raw_skills.csv"
    TOP_N = 3

    # Load
    df = load_dataset(DATASET_PATH)

    # Build TF-IDF corpus from job roles
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)

    # Ingest user
    user_skills = get_user_input()

    # Score
    scores = score_user_profile(user_skills, vectorizer, tfidf_matrix)

    # Cold start detection
    if handle_cold_start(scores):
        print("\n  ⚠  Cold Start Detected: none of your skills exist")
        print("     in the dataset vocabulary.")
        print(f"     Known skills include: {', '.join(list(vectorizer.vocabulary_.keys())[:10])}...")
        return

    # Sort + Filter
    results = get_top_recommendations(df, scores, top_n=TOP_N)

    # Display
    display_results(results, user_skills)


if __name__ == "__main__":
    main()
