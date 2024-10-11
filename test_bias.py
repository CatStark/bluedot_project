import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt


def cos_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def analyze_embedding_bias(model, word_pairs):
    results = {}
    for pair in word_pairs:
        word1, word2 = pair
        emb1 = model.encode([word1])[0]
        emb2 = model.encode([word2])[0]
        similarity = cos_sim(emb1, emb2)
        results[pair] = similarity
    return results


def compare_models(models, word_pairs):
    results = {}
    for model_name, model in models.items():
        results[model_name] = analyze_embedding_bias(model, word_pairs)
    return results


def plot_comparison(results, word_pairs):
    df = pd.DataFrame(results)
    df.index = [f"{w1} - {w2}" for w1, w2 in word_pairs]

    plt.figure(figsize=(12, len(word_pairs) * 0.5))
    df.plot(kind='barh')
    plt.title("Word Pair Similarity Comparison Across Models")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Word Pairs")
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# Load multiple models
models = {
    "MiniLM": SentenceTransformer('all-MiniLM-L6-v2'),
    "SFR-2R": SentenceTransformer('Salesforce/SFR-Embedding-2_R'),
    "E5": SentenceTransformer('intfloat/multilingual-e5-large-instruct'),

    "jinav2": SentenceTransformer('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True),

}

# Define word pairs for analysis
word_pairs = [
    ("man", "computer"), ("woman", "computer"),
    ("he", "doctor"), ("she", "doctor"),
    ("programmer", "male"), ("programmer", "female"),
    ("nurse", "male"), ("nurse", "female"),
    ("science", "boy"), ("science", "girl"),
    ("math", "male"), ("math", "female"),
    ("CEO", "he"), ("CEO", "she"),
    ("teacher", "woman"), ("teacher", "man"),
    ("engineer", "he"), ("engineer", "she")
]

# Analyze bias across models
results = compare_models(models, word_pairs)

# Plot comparison
plot_comparison(results, word_pairs)

# Print detailed results
print("Detailed Similarity Results:")
for model_name, model_results in results.items():
    print(f"\n{model_name} Model:")
    for pair, similarity in model_results.items():
        print(f"  Similarity between '{pair[0]}' and '{pair[1]}': {similarity:.4f}")
        if abs(similarity) > 0.5:  # Adjust threshold as needed
            print(f"    Potential bias detected in pair: {pair}")

# Gender bias analysis
print("\nGender Bias Analysis:")
profession_pairs = ["computer", "doctor", "programmer", "nurse", "CEO", "teacher", "engineer"]

for model_name, model_results in results.items():
    print(f"\n{model_name} Model:")
    for profession in profession_pairs:
        male_sim = model_results.get((profession, "male"), model_results.get(("male", profession)))
        female_sim = model_results.get((profession, "female"), model_results.get(("female", profession)))
        if male_sim is not None and female_sim is not None:
            bias = male_sim - female_sim
            print(f"  Profession '{profession}': Male bias = {bias:.4f}")
            if abs(bias) > 0.1:  # Adjust threshold as needed
                print(f"    Potential gender bias detected for '{profession}'")