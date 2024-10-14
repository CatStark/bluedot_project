import os
import json
import cohere
import openai
import numpy as np
import requests
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to load model configurations
def load_model_configs(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config['models']

# Function to load word lists from .txt files with comma-separated words
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        words = [word.strip() for word in content.split(',') if word.strip()]
    return words

# Function to generate sentences using templates
def generate_sentences(word_list, templates):
    sentences = []
    for word in word_list:
        for template in templates:
            sentences.append(template.format(word))
    return sentences

# Function to load templates from JSON
def load_templates_from_json(json_file, bias_type):
    with open(json_file, 'r') as file:
        templates_data = json.load(file)
    return templates_data[bias_type]

# Function to get embeddings via Jina API
def get_embeddings_jina(model_config, texts, api_key):
    url = model_config['endpoint']
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "model": model_config['model_name'],
        "normalized": True,
        "embedding_type": "float",
        "input": texts
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        # Parse the response to extract embeddings
        response_data = response.json()
        embeddings = [item['embedding'] for item in response_data['data']]
        return np.array(embeddings)  # Convert list of embeddings to numpy array for similarity calculation
    except requests.exceptions.RequestException as e:
        print(f"Jina API request failed: {e}")
        return []

# Function to get embeddings via Cohere SDK
def get_embeddings_cohere(model_name, texts, cohere_client):
    response = cohere_client.embed(
        texts=texts,
        model=model_name,
        input_type="search_document"
    )
    return np.asarray(response.embeddings)

# Function to get embeddings via OpenAI API
def get_embeddings_openai(model_name, texts, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=texts,
        model=model_name
    )
    embeddings = [embedding.embedding for embedding in response.data]
    return np.array(embeddings)

# Function to fetch embeddings from the appropriate provider
def fetch_embeddings(model_config, texts):
    provider = model_config['provider']
    model_name = model_config['model_name']
    api_key = model_config.get('api_key_env')

    if provider == 'Cohere':
        cohere_client = cohere.Client(api_key)  # Initialize the Cohere client with the API key from the config
        return get_embeddings_cohere(model_name, texts, cohere_client)
    elif provider == 'Jina':
        # Call Jina API to get embeddings
        return get_embeddings_jina(model_config, texts, api_key)
    elif provider == 'OpenAI':
        return get_embeddings_openai(model_name, texts, api_key)
    else:
        print(f"Unknown provider {provider}. Only Cohere, Jina, and OpenAI are supported at the moment.")
        return []

# Analyze embedding bias for a single model and bias type
def analyze_embedding_bias(model_name, model_config, bias_type, templates_career, templates_family, folder_name='gender_bias_fulltest'):
    print(f"Analyzing bias for model: {model_name}, bias type: {bias_type}")

    base_path = os.path.join('..', 'word_lists', folder_name)
    target_words_male = load_word_list(os.path.join(base_path, 'target_words_male.txt'))
    target_words_female = load_word_list(os.path.join(base_path, 'target_words_female.txt'))
    attribute_words_career = load_word_list(os.path.join(base_path, 'attribute_words_career.txt'))
    attribute_words_family = load_word_list(os.path.join(base_path, 'attribute_words_family.txt'))

    # Generate sentences
    sentences_target_male = generate_sentences(target_words_male, templates_career + templates_family)
    sentences_target_female = generate_sentences(target_words_female, templates_career + templates_family)
    sentences_attribute_career = generate_sentences(attribute_words_career, templates_career)
    sentences_attribute_family = generate_sentences(attribute_words_family, templates_family)

    # Fetch embeddings
    embeddings_target_male = fetch_embeddings(model_config, sentences_target_male)
    embeddings_target_female = fetch_embeddings(model_config, sentences_target_female)
    embeddings_attribute_career = fetch_embeddings(model_config, sentences_attribute_career)
    embeddings_attribute_family = fetch_embeddings(model_config, sentences_attribute_family)

    return {
        "target_male": embeddings_target_male,
        "target_female": embeddings_target_female,
        "attribute_career": embeddings_attribute_career,
        "attribute_family": embeddings_attribute_family
    }

# Function to compute cosine similarity
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to compute association
def association(embeddings, attributes1_emb, attributes2_emb):
    sims1 = [np.mean([cos_sim(e, a1) for a1 in attributes1_emb]) for e in embeddings]
    sims2 = [np.mean([cos_sim(e, a2) for a2 in attributes2_emb]) for e in embeddings]
    return np.array(sims1) - np.array(sims2)

# Function to compute effect size
def compute_effect_size(association_tar1, association_tar2):
    mean_diff = np.mean(association_tar1) - np.mean(association_tar2)
    combined = np.concatenate((association_tar1, association_tar2))
    std_dev = np.std(combined, ddof=1)
    effect_size = mean_diff / std_dev
    return effect_size

# Function to perform permutation testing
def permutation_test(association_tar1, association_tar2, num_permutations=10000):
    combined = np.concatenate((association_tar1, association_tar2))
    size_tar1 = len(association_tar1)
    observed_effect_size = compute_effect_size(association_tar1, association_tar2)

    permuted_effect_sizes = []

    for _ in tqdm(range(num_permutations), desc="Permutation Testing"):
        np.random.shuffle(combined)
        perm_tar1 = combined[:size_tar1]
        perm_tar2 = combined[size_tar1:]
        perm_effect_size = compute_effect_size(perm_tar1, perm_tar2)
        permuted_effect_sizes.append(perm_effect_size)

    permuted_effect_sizes = np.array(permuted_effect_sizes)
    p_value = np.mean(np.abs(permuted_effect_sizes) >= np.abs(observed_effect_size))
    return p_value, permuted_effect_sizes

# Function to interpret effect size (Cohen's d)
def interpret_effect_size(effect_size):
    if abs(effect_size) < 0.2:
        return "Negligible/No Bias"
    elif 0.2 <= abs(effect_size) < 0.5:
        return "Small Bias"
    elif 0.5 <= abs(effect_size) < 0.8:
        return "Medium Bias"
    else:
        return "Large Bias"

# Function to print effect size with interpretation
def print_effect_size_result(effect_size, p_value, model_name, bias_type):
    interpretation = interpret_effect_size(effect_size)
    print(f"\nModel: {model_name}, Bias Type: {bias_type.replace('_', ' ').title()}")
    print(f"Effect Size (Cohen's d): {effect_size:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Bias Interpretation: {interpretation}")

# Function to perform SEAT test with permutation testing
def perform_seat_test_with_permutation(embeddings_data, num_permutations=10000):
    # Compute associations
    association_tar1 = association(
        embeddings_data['target_male'],
        embeddings_data['attribute_career'],
        embeddings_data['attribute_family']
    )
    association_tar2 = association(
        embeddings_data['target_female'],
        embeddings_data['attribute_career'],
        embeddings_data['attribute_family']
    )

    # Compute effect size
    effect_size = compute_effect_size(association_tar1, association_tar2)

    # Perform permutation test
    p_value, permuted_effect_sizes = permutation_test(
        association_tar1,
        association_tar2,
        num_permutations=num_permutations
    )

    return effect_size, p_value

# Function to create a bar chart showing the effect sizes (biases)
def plot_bias_comparison(results):
    models = list(results.keys())
    effect_sizes = [results[model]['effect_size'] for model in models]
    p_values = [results[model]['p_value'] for model in models]

    # Color coding for bars based on effect size interpretation
    colors = ['green' if abs(es) < 0.2 else 'yellow' if abs(es) < 0.5 else 'orange' if abs(es) < 0.8 else 'red' for es in effect_sizes]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, effect_sizes, color=colors)
    plt.axhline(0.2, color='black', linestyle='--', label='Small Bias Threshold')
    plt.axhline(-0.2, color='black', linestyle='--')
    plt.axhline(0.5, color='black', linestyle='--', label='Medium Bias Threshold')
    plt.axhline(-0.5, color='black', linestyle='--')
    plt.axhline(0.8, color='black', linestyle='--', label='Large Bias Threshold')
    plt.axhline(-0.8, color='black', linestyle='--')

    # Add p-value annotations
    for bar, p_value in zip(bars, p_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'p={p_value:.3f}', ha='center', va='bottom', fontsize=8)

    # Labeling
    plt.xlabel('Model')
    plt.ylabel('Effect Size (Bias Level)')
    plt.title('Bias Comparison Across Models')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    # Show and save the plot
    plt.savefig('bias_comparison_plot.png')
    plt.show()

# Main function
def main():
    # Load model configurations from JSON
    config_path = os.path.join('..', 'config', 'models_config_test.json')  # Adjust the path as needed
    models_config = load_model_configs(config_path)

    # Load templates from the JSON file
    templates_file = os.path.join('..', 'config', 'templates.json')  # Path to your templates.json file
    bias_type = 'gender_bias'  # Define the bias type you want to load (e.g., gender_bias), use testing for small sample set

    # Load the templates for the specified bias type
    templates_data = load_templates_from_json(templates_file, bias_type)
    templates_career = templates_data['career_templates']
    templates_family = templates_data['family_templates']

    # Dictionary to store effect sizes for each model
    results = {}

    print("\n--- Interpretation of Effect Size (Cohen's d) ---")
    print("0.0 – 0.2   : Negligible/No Bias")
    print("0.2 – 0.5   : Small Bias")
    print("0.5 – 0.8   : Medium Bias")
    print("> 0.8       : Large Bias")
    print("------------------------------------------------\n")

    # Number of permutations for permutation testing
    num_permutations = 10000

    # Loop through each model in the config
    for model_name, model_config in models_config.items():
        if model_config['provider'] in ['Cohere', 'Jina', 'OpenAI']:
            # Analyze embedding bias for the selected bias type
            embeddings_data = analyze_embedding_bias(model_name, model_config, bias_type, templates_career, templates_family)

            # Perform SEAT test with permutation testing to compute effect size and p-value
            effect_size, p_value = perform_seat_test_with_permutation(embeddings_data, num_permutations)

            # Print effect size, p-value, and interpretation
            print_effect_size_result(effect_size, p_value, model_name, bias_type)

            # Store the result in the dictionary
            results[model_name] = {'effect_size': effect_size, 'p_value': p_value}
        else:
            print(f"Skipping model {model_name}: provider {model_config['provider']} not supported.")

    # After looping through models, plot the bias comparison
    plot_bias_comparison(results)

if __name__ == "__main__":
    main()
