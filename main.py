
"""
Intelligent Resume Screening and Job Matching System

A simplified demonstration of resume-to-job matching.
"""

import json
import re


def load_config():
    """Load application settings from config.json."""
    with open("config.json", "r", encoding="utf-8") as file:
        return json.load(file)


def keyword_match(resume, job_description):
    """Calculate a simple keyword overlap score."""
    resume_words = set(re.findall(r"\b\w+\b", resume.lower()))
    job_words = set(re.findall(r"\b\w+\b", job_description.lower()))

    if not job_words:
        return 0.0

    matched_words = resume_words.intersection(job_words)
    return round(len(matched_words) / len(job_words) * 100, 2)


def main():
    config = load_config()

    print("Project:", config["project_name"])
    print("Version:", config["version"])

    resume = "Python developer with SQL and data analysis experience"
    job_description = "Looking for Python developer with SQL skills"

    score = keyword_match(resume, job_description)

    print("Resume Matching Score:", score, "%")


if __name__ == "__main__":
    main()
