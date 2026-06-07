# Project 3 — AI Recommendation Logic 🎯

**DecodeLabs AI Internship | Batch 2026**
**Intern:** Harshvardhan Zala
**Track:** Artificial Intelligence

---

## Overview

A content-based recommendation engine built on **TF-IDF vectorization and cosine similarity** — no collaborative filtering, no user history, no black-box neural networks. Every recommendation is mathematically traceable from input to ranked output.

This project demonstrates the core personalization architecture that powers production systems like Netflix's genre matching and Amazon's product relevance engine — the shift from passive classification to **active prediction**.

---

## Architecture — The IPO Model

```
INPUT             →        PROCESS          →        OUTPUT
(User State)               (Similarity Logic)         (Top-N List)

Skill Ingestion       TF-IDF Vectorization       Ranked Career
& Validation       &  Cosine Similarity Score      Paths (Top 3)
```

### Why TF-IDF over Binary Vectors?

| Approach | Flaw | TF-IDF Fix |
|---|---|---|
| Binary (1/0) presence | Treats "python" and "kubernetes" as equal weight | Penalizes common terms, rewards rare specific ones |
| Raw term frequency | Biased toward longer descriptions | Normalized by document length (TF component) |
| Euclidean distance | Sensitive to vector magnitude | Cosine similarity is magnitude-invariant |

Binary vectors treat generic, high-frequency words the same as highly specific tags. TF-IDF assigns **lower weight to common terms** (appearing across many job roles) and **higher weight to distinctive ones** — making "solidity" a stronger signal than "python".

---

## Project Specifications (Logic Skeleton)

- ✅ **INGESTION** — Interactive skill input loop, enforces minimum 3 skills
- ✅ **DEDUPLICATION** — Rejects duplicate skill entries in the same session
- ✅ **TF-IDF VECTORIZATION** — Shared vocabulary space across user + item corpus
- ✅ **COSINE SIMILARITY SCORING** — Magnitude-invariant similarity metric
- ✅ **SORTED RANKING** — Descending sort by similarity score
- ✅ **TOP-N FILTERING** — Returns Top 3 matches, prevents choice overload
- ✅ **COLD START GUARD** — Detects zero-vector profiles, surfaces vocabulary hints
- ✅ **VISUAL OUTPUT** — ASCII progress bar with percentage match score

---

## How to Run

**Requirements:** Python 3.x, scikit-learn, pandas, numpy

```bash
pip install scikit-learn pandas numpy
python recommender.py
```

**Example session:**
```
=======================================================
   TECH STACK RECOMMENDER — DecodeLabs AI Intern
=======================================================

Enter your skills one by one (minimum 3).
Type 'done' when finished.

  Skill #1: python
  ✓  Added: python
  Skill #2: machine_learning
  ✓  Added: machine_learning
  Skill #3: deep_learning
  ✓  Added: deep_learning
  Skill #4: done

=======================================================
   TOP CAREER RECOMMENDATIONS
=======================================================

  Your skills: python, machine_learning, deep_learning

  #1  Machine Learning Engineer
      Match: [█████████░░░░░░░░░░░] 48.5%
      Key skills: python machine_learning deep_learning tensorflow pytorch...

  #2  NLP Engineer
      Match: [█████████░░░░░░░░░░░] 47.6%
      Key skills: python nlp transformers huggingface deep_learning pytorch...

  #3  AI Engineer
      Match: [█████░░░░░░░░░░░░░░░] 27.7%
      Key skills: python deep_learning nlp computer_vision tensorflow...

=======================================================
  Tip: Add more specific skills for tighter matches.
=======================================================
```

---

## The Math Behind the Match

### TF-IDF Weighting

```
TF  =  (Count of term t in document d) / (Total terms in document d)

IDF =  log( Total Documents / Documents containing term t )

TF-IDF(t, d)  =  TF × IDF
```

The logarithm in IDF acts as a **dampening function** — ensuring the penalty for high-frequency words scales logarithmically rather than linearly. A term appearing in every job role gets IDF ≈ 0. A term appearing in only one role gets a high IDF score.

### Cosine Similarity

```
cos(θ)  =  (A · B) / (||A|| × ||B||)
```

| Score | Meaning |
|---|---|
| 1.0 | Identical orientation — perfect match |
| 0.5 | Partial alignment — moderate match |
| 0.0 | Orthogonal vectors — no shared characteristics |

Cosine similarity focuses strictly on **direction**, not magnitude. A user with 3 skills and a job role with 12 skills can still score 1.0 — if their skills point in the same direction.

---

## Cold Start Problem

A new user with no prior data produces a **zero vector**. Any dot product against a zero vector returns 0 — the system cannot make recommendations.

**Detection in this project:**
```python
if np.all(scores == 0):
    # Cold start detected — no vocabulary overlap
```

**Mitigations implemented:**
- Surfaces known vocabulary terms to guide the user toward valid inputs
- Minimum 3-skill requirement ensures sufficient data density before scoring runs

---

## Concepts Demonstrated

- **Content-Based Filtering** — Item attributes drive recommendations, independent of other users
- **TF-IDF Vectorization** — Statistical feature extraction with specificity weighting
- **Cosine Similarity** — Magnitude-invariant angular distance metric
- **Shared Vocabulary Space** — User profile and item corpus mapped to identical feature dimensions
- **IPO Pipeline** — Ingestion → Scoring → Sorting → Filtering
- **Cold Start Handling** — Zero-vector detection with graceful fallback
- **Top-N Truncation** — Prevents choice overload by limiting output

---

## Project Structure

```
decodelabs_ai_project3/
├── recommender.py   # Main recommendation engine — fully documented
├── raw_skills.csv   # Dataset — 20 job roles with skill tags
└── README.md        # This file
```

---

## Dataset: `raw_skills.csv`

20 job roles across AI, DevOps, Web, Data, and Security domains. Each role is represented as a space-separated string of skill tags — the format required for TF-IDF tokenization.

| Job Role | Sample Skills |
|---|---|
| Data Scientist | python, machine_learning, sql, statistics, pandas |
| DevOps Engineer | docker, kubernetes, aws, ci_cd, terraform |
| NLP Engineer | python, nlp, transformers, huggingface, pytorch |
| Cybersecurity Analyst | networking, security, penetration_testing, linux |
| Blockchain Developer | solidity, ethereum, smart_contracts, web3 |

---

*DecodeLabs | Batch 2026 | Artificial Intelligence Track*
