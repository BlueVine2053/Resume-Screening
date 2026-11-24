"""Deterministic baseline matchers. No model API or hiring decisions."""
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"(?u)\b[a-z0-9]+\b")
MAX_TEXT_CHARS = 100_000


def tokens(text):
    return set(TOKEN_RE.findall(str(text).lower()))


def keyword_score(resume, job, vocabulary):
    terms = tokens(job) & vocabulary
    if not terms:
        return None  # Not scoreable, never silently count as a zero
    return len(tokens(resume) & terms) / len(terms)


def validate_text(text, field):
    if not isinstance(text, str):
        raise ValueError(f"{field} must be nonempty text")

    if not text.strip():
        raise ValueError(f"{field} must be nonempty text")

    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"{field} must not exceed {MAX_TEXT_CHARS:,} characters"
        )

    if not tokens(text):
        raise ValueError(
            f"{field} contains no recognizable English letters or digits"
        )


class Matcher:
    def __init__(self, training_jobs, training_resumes=None, max_keywords=200):
        jobs = [str(s) for s in training_jobs]
        resumes = [str(s) for s in (training_resumes if training_resumes is not None else [])]
        if not any(s.strip() for s in jobs):
            raise ValueError("Nonempty training job descriptions are required")
        counts = Counter(t for job in jobs for t in tokens(job)
                         if t not in ENGLISH_STOP_WORDS and len(t) > 1)
        # Stable ordering for ties; vocabulary is learned from TRAIN jobs only.
        self.vocabulary = {word for word, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_keywords]}
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                                          token_pattern=r"(?u)\b\w+\b")
        self.vectorizer.fit(jobs + resumes)  # Training texts only; no validation/test text

    def score(self, resume, job):
        validate_text(resume, "Resume")
        validate_text(job, "Job description")

        vectors = self.vectorizer.transform([resume, job])
        if vectors[0].nnz == 0 or vectors[1].nnz == 0:
            v2 = 0.0
        else:
            v2 = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
        return {'v1': keyword_score(resume, job, self.vocabulary),
                'v2': float(np.clip(v2, 0.0, 1.0))}

    def score_pairs(self, resumes, jobs):
        if len(resumes) != len(jobs):
            raise ValueError("Resumes and jobs must have the same number of entries")
        return [self.score(r, j) for r, j in zip(resumes, jobs)]
