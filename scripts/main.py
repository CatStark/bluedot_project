import os
import json
import cohere
import openai
import voyageai
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

# Function to get embeddings via Voyage API
def get_embeddings_voyage(model_name, texts, voyage_client, batch_size=128):
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size  # Calculate the number of batches

    for i in range(num_batches):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        response = voyage_client.embed(texts=batch_texts, model=model_name, input_type="document")
        all_embeddings.extend(response.embeddings)  # Accumulate embeddings from all batches

    return np.asarray(all_embeddings)

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
    elif provider == 'Voyage':
        voyage_client = voyageai.Client(api_key)  # Initialize the Voyage client with the API key
        return get_embeddings_voyage(model_name, texts, voyage_client)
    else:
        print(f"Unknown provider {provider}. Only Cohere, Jina, OpenAI and Voyage are supported at the moment.")
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
        p_value, permuted_effect_sizes = permutation_test(association_male, association_female, num_permutations)

        # Store results
        bias_pair = f"{attr1_name} vs {attr2_name}"
        results[bias_pair] = {
            'effect_size': effect_size,
            'p_value': p_value,
            'permuted_effect_sizes': permuted_effect_sizes  # Save permutation test results
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
            plt.text(index[j] + i * bar_width, es, f'p={p_value:.4f}', ha='center', va='bottom', fontsize=8)

    # Labeling
    plt.xlabel('Effect Size (Cohensd) \n Positive: Bias Towards Males with Career; Negative: Bias Towards Females with Career')
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

# Function to create a diverging bar chart showing the effect sizes (biases)
def plot_diverging_bias_comparison(results):
    models = list(results.keys())
    bias_pairs = list(next(iter(results.values())).keys())

    # Number of models and attribute pairs
    n_models = len(models)
    n_pairs = len(bias_pairs)

    # Set up figure size and subplots
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 6 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]  # Ensure axes is always iterable

    # Loop through each attribute pair to create diverging bar charts
    for i, attr_pair in enumerate(bias_pairs):
        effect_sizes = []
        model_labels = []
        p_values = []

        # Gather effect sizes, p-values, and corresponding model names
        for model_name, model_results in results.items():
            effect_size = model_results[attr_pair]['effect_size']
            p_value = model_results[attr_pair].get('p_value', None)  # Assuming p-values are included
            effect_sizes.append(effect_size)
            p_values.append(p_value)
            model_labels.append(model_name)

        # Convert to numpy array for easier processing
        effect_sizes = np.array(effect_sizes)

        # Set bar colors based on the direction of the bias
        bar_colors = []
        for es in effect_sizes:
            if abs(es) >= 0.8:
                bar_colors.append('red')  # Large Bias
            elif abs(es) >= 0.5:
                bar_colors.append('yellow')  # Medium Bias
            elif abs(es) >= 0.2:
                bar_colors.append('green')  # Small Bias
            else:
                bar_colors.append('tab:blue')  # Negligible/No Bias

        # Plot the diverging bars
        axes[i].barh(model_labels, effect_sizes, color=bar_colors, edgecolor='black')

        # Emphasize the zero line (no bias)
        axes[i].axvline(0, color='blue', linewidth=2)  # Draw the central axis with a thicker blue line

        # Add vertical threshold lines
        axes[i].axvline(0.2, color='black', linestyle='--', linewidth=1, label='Small Bias Threshold')
        axes[i].axvline(-0.2, color='black', linestyle='--', linewidth=1)
        axes[i].axvline(0.5, color='black', linestyle='-.', linewidth=1, label='Medium Bias Threshold')
        axes[i].axvline(-0.5, color='black', linestyle='-.', linewidth=1)
        axes[i].axvline(0.8, color='black', linestyle=':', linewidth=1, label='Large Bias Threshold')
        axes[i].axvline(-0.8, color='black', linestyle=':', linewidth=1)

        # Update the title based on the attribute pair
        axes[i].set_title(
            f"Positive Values → Bias Associating Males with 'Career' over 'Family' \n Negative Values → Bias Associating Females with 'Career' over 'Family'",
            fontsize=14)

        # Add labels and title for leadership/support
        if "leadership" in attr_pair.lower():
            axes[i].set_title(
                f"Negative Values → Bias Associating Females with 'Leadership' over 'Support '  \n Negative Values → Bias Associating Females with 'Leadership' over 'Support' ",
                fontsize=14)



        # Set limits for better visualization
        axes[i].set_xlim([-1.0, 1.0])

        # Add effect size and p-value labels on top of each bar
        for j, (effect_size, p_value) in enumerate(zip(effect_sizes, p_values)):
            p_val_str = f", p = {p_value:.4f}" if p_value is not None else ""
            label = f'd = {effect_size:.4f}{p_val_str}'
            axes[i].text(effect_size, j, label, va='center', ha='left' if effect_size < 0 else 'right',
                         fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Add legend
    axes[-1].legend(loc='upper right')

    # Add footnote explaining effect sizes
    fig.text(0.5, -0.02, "Effect sizes are interpreted as: 0.2 = Small Bias, 0.5 = Medium Bias, 0.8 = Large Bias.",
             ha='center', fontsize=12)

    # Adjust layout and add padding to prevent word cutoff
    plt.tight_layout(pad=3.0)

    results_dir = os.path.join('..', 'results/')
    plt.savefig(results_dir + 'diverging_bias_comparison_plot.png', bbox_inches='tight')  # Ensure words are not cut
    plt.show()


# Function to create a plot with point estimates and 95% confidence intervals (error bars)
def plot_point_estimates_with_error_bars(results):
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a plot for each attribute pair (e.g., Career vs Family, Leadership vs Support)
    for bias_pair in next(iter(results.values())).keys():
        fig, ax = plt.subplots(figsize=(8, 6))

        # Initialize lists to store data
        effect_sizes = []
        confidence_intervals = []
        model_labels = []

        # Extract effect sizes and calculate confidence intervals for each model
        for model_name, model_results in results.items():
            result = model_results[bias_pair]
            effect_size = result['effect_size']
            p_value = result['p_value']  # Assuming p-value was already calculated

            # Assuming a fixed standard error for CI calculation
            # Replace with real standard error if available from permutation test
            confidence_interval = 1.96 * 0.1  # Replace 0.1 with the real standard error if available

            effect_sizes.append(effect_size)
            confidence_intervals.append(confidence_interval)
            model_labels.append(model_name)

        # Convert lists to numpy arrays for easier plotting
        effect_sizes = np.array(effect_sizes)
        confidence_intervals = np.array(confidence_intervals)

        # Plotting the point estimates with error bars (95% CI)
        ax.errorbar(model_labels, effect_sizes, yerr=confidence_intervals, fmt='o', capsize=5, capthick=2, color='blue',
                    label="Effect Size (Cohen's d)")

        # Horizontal line at y = 0 for visual reference (no bias)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5)

        # Labels and title
        ax.set_ylabel("Effect Size (Cohen's d)")
        ax.set_xlabel("Models")
        ax.set_title(f"{bias_pair.replace('_', ' ').title()} Bias: Point Estimates with 95% Confidence Intervals")

        # Rotate model labels on x-axis for better readability
        plt.xticks(rotation=45, ha='right')

        # Display grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save and show the plot
        plt.tight_layout()
        plt.savefig(f'../results/{bias_pair}_combined_point_estimate_plot.png')
        plt.show()




# Helper function to convert numpy arrays to lists for JSON serialization
def convert_numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_numpy_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_list(item) for item in data]
    else:
        return data


# Function to save results to a JSON file
def save_results_to_json(results, file_name='bias_analysis_results.json', output_dir='../results/'):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert numpy arrays in the results to lists
    results_converted = convert_numpy_to_list(results)

    # Define the full path to the output file
    output_file = os.path.join(output_dir, file_name)

    # Save the converted results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results_converted, json_file, indent=4)

    print(f"Results saved to {output_file}")

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
        #('physical_appearance', 'intelligence'),
        #('agentic', 'communal'),
        #('logical', 'emotional'),
        # Add more pairs as needed
    ]

    print("\n--- Interpretation of Effect Size (Cohen's d) ---")
    print("0.0 – 0.2   : Negligible/No Bias")
    print("0.2 – 0.5   : Small Bias")
    print("0.5 – 0.8   : Medium Bias")
    print("> 0.8       : Large Bias")
    print("------------------------------------------------\n")

    # Number of permutations for permutation testing
    #num_permutations = 10000
    num_permutations = 100 # testing

    # Store all results for each model
    all_results = {}

    # Loop through each model in the config
    for model_name, model_config in models_config.items():
        if model_config['provider'] in ['Voyage']:#'Cohere', 'Voyage', 'OpenAI']:
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

    # Save all results to a JSON file
    save_results_to_json(all_results)

    # Plot bias comparison for the model
    #plot_bias_comparison(all_results)
    plot_diverging_bias_comparison(all_results)
    plot_point_estimates_with_error_bars(all_results)


if __name__ == "__main__":
    main()