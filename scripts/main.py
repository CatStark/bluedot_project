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
def analyze_embedding_bias(model_name, model_config, bias_type, attribute_templates, attribute_pairs, folder_name='gender_bias_testing'):
    print(f"Analyzing bias for model: {model_name}, bias type: {bias_type}")

    base_path = os.path.join('..', 'word_lists', folder_name)
    target_words_male = load_word_list(os.path.join(base_path, 'target_words_male.txt'))
    target_words_female = load_word_list(os.path.join(base_path, 'target_words_female.txt'))

    # Load attribute words
    attribute_words = {}
    for attribute_name in attribute_templates.keys():
        file_name = f'attribute_words_{attribute_name}.txt'
        file_path = os.path.join(base_path, file_name)
        attribute_words[attribute_name] = load_word_list(file_path)

    # Generate sentences for target words using all templates
    all_templates = []
    for templates in attribute_templates.values():
        all_templates.extend(templates)

    sentences_target_male = generate_sentences(target_words_male, all_templates)
    sentences_target_female = generate_sentences(target_words_female, all_templates)

    # Generate sentences for attribute words
    sentences_attribute = {}
    for attribute_name, words in attribute_words.items():
        templates = attribute_templates[attribute_name]
        sentences_attribute[attribute_name] = generate_sentences(words, templates)

    # Fetch embeddings for target sentences
    embeddings_target_male = fetch_embeddings(model_config, sentences_target_male)
    embeddings_target_female = fetch_embeddings(model_config, sentences_target_female)

    # Fetch embeddings for attribute sentences
    embeddings_attribute = {}
    for attribute_name, sentences in sentences_attribute.items():
        embeddings_attribute[attribute_name] = fetch_embeddings(model_config, sentences)

    return {
        "target_male": embeddings_target_male,
        "target_female": embeddings_target_female,
        "embeddings_attribute": embeddings_attribute
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
    direction = "Male terms are more associated with the first attribute group." if effect_size > 0 else "Female terms are more associated with the first attribute group."
    print(f"\nModel: {model_name}, Bias Type: {bias_type.replace('_', ' ').title()}")
    print(f"Effect Size (Cohen's d): {effect_size:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Bias Interpretation: {interpretation}")
    print(f"Direction of Bias: {direction}")

# Function to perform SEAT test with permutation testing
def perform_seat_test_with_permutation(embeddings_data, attribute_pairs, num_permutations=10000):
    results = {}

    # Iterate over attribute pairs
    for attr_pair in attribute_pairs:
        attr1_name, attr2_name = attr_pair

        # Compute associations for male and female targets
        association_male = association(
            embeddings_data['target_male'],
            embeddings_data['embeddings_attribute'][attr1_name],
            embeddings_data['embeddings_attribute'][attr2_name]
        )
        association_female = association(
            embeddings_data['target_female'],
            embeddings_data['embeddings_attribute'][attr1_name],
            embeddings_data['embeddings_attribute'][attr2_name]
        )

        # Compute effect size and p-value
        effect_size = compute_effect_size(association_male, association_female)
        p_value, _ = permutation_test(association_male, association_female, num_permutations)

        # Store results
        bias_pair = f"{attr1_name} vs {attr2_name}"
        results[bias_pair] = {
            'effect_size': effect_size,
            'p_value': p_value
        }

    return results

# Function to create a bar chart showing the effect sizes (biases)
def plot_bias_comparison(results):
    models = list(results.keys())
    bias_pairs = list(next(iter(results.values())).keys())

    # Number of models and attribute pairs
    n_models = len(models)
    n_pairs = len(bias_pairs)

    # Set up the bar width and positions
    bar_width = 0.2
    index = np.arange(n_pairs)

    # Create the bar plot
    plt.figure(figsize=(14, 8))

    for i, (model_name, results) in enumerate(results.items()):
        effect_sizes = [results[bias_pair]['effect_size'] for bias_pair in bias_pairs]
        p_values = [results[bias_pair]['p_value'] for bias_pair in bias_pairs]

        # Offset the positions of the bars for each model
        plt.bar(index + i * bar_width, effect_sizes, bar_width, label=f'{model_name}')

        # Add p-value annotations for each model
        for j, (es, p_value) in enumerate(zip(effect_sizes, p_values)):
            plt.text(index[j] + i * bar_width, es, f'p={p_value:.3f}', ha='center', va='bottom', fontsize=8)

    # Labeling
    plt.xlabel('Attribute Pairs')
    plt.ylabel('Effect Size (Bias Level)')
    plt.title('Bias Comparison for All Models')
    plt.xticks(index + bar_width * (n_models - 1) / 2, bias_pairs, rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.tight_layout()

    # Show and save the plot
    results_dir = os.path.join('..', 'results/')
    plt.savefig(results_dir + 'combined_bias_comparison_plot.png')
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
    attribute_templates = {
        key.replace('_templates', ''): value
        for key, value in templates_data.items()
        if key.endswith('_templates')
    }

    # Define attribute pairs to analyze
    attribute_pairs = [
        ('career', 'family'),
        ('leadership', 'support'),
        #('agentic', 'communal'),
        #('logical', 'emotional'),
        #('physical_appearance', 'intelligence'),
        # Add more pairs as needed
    ]

    print("\n--- Interpretation of Effect Size (Cohen's d) ---")
    print("0.0 – 0.2   : Negligible/No Bias")
    print("0.2 – 0.5   : Small Bias")
    print("0.5 – 0.8   : Medium Bias")
    print("> 0.8       : Large Bias")
    print("------------------------------------------------\n")

    # Number of permutations for permutation testing
    num_permutations = 10000

    # Store all results for each model
    all_results = {}

    # Loop through each model in the config
    for model_name, model_config in models_config.items():
        if model_config['provider'] in ['Cohere', 'Jina', 'OpenAI']:
            # Analyze embedding bias for the selected bias type
            embeddings_data = analyze_embedding_bias(
                model_name,
                model_config,
                bias_type,
                attribute_templates,
                attribute_pairs
            )

            # Perform bias analysis
            results = perform_seat_test_with_permutation(embeddings_data, attribute_pairs, num_permutations=num_permutations)

            # Store the results for plotting
            all_results[model_name] = results

            # Print results
            for bias_pair, result in results.items():
                effect_size = result['effect_size']
                p_value = result['p_value']
                print_effect_size_result(effect_size, p_value, model_name, bias_pair)


        else:
            print(f"Skipping model {model_name}: provider {model_config['provider']} not supported.")

    # Plot bias comparison for the model
    plot_bias_comparison(all_results)

if __name__ == "__main__":
    main()
