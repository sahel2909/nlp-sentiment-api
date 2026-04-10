"""
Topic Modeling using LDA (API Ready Version)
"""

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def topic_modeling(texts: list[str], n_topics: int = 3) -> list[dict]:
    """
    Perform LDA topic modeling and return structured output.
    """

    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words="english"
    )

    term_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

    lda.fit(term_matrix)

    feature_names = vectorizer.get_feature_names_out()

    topics_output = []

    for topic_idx, topic in enumerate(lda.components_):
        top_words = topic.argsort()[-8:]
        keywords = [feature_names[i] for i in top_words]

        topics_output.append({
            "topic_id": topic_idx + 1,
            "keywords": keywords
        })

    return topics_output


if __name__ == "__main__":
    df = pd.read_csv("data/clean/cleaned_texts.csv")
    print(topic_modeling(df["clean_text"].astype(str).tolist()))
