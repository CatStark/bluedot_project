# Bias Analysis of Word Embeddings in Language Models


## Table of Contents

- [Overview](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
- [Design and Methodology](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [1. Loading Configurations and Data](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [2. Generating Sentences](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [3. Fetching Embeddings](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [4. Bias Analysis](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
        - [4.1 Cosine Similarity](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
        - [4.2 Association Scores](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
        - [4.3 Effect Size (Cohen's d)](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
        - [4.4 Permutation Testing and P-Value](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
        - [4.5 Interpreting Effect Size](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [5. Visualization](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
- [Metrics Used](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
- [Understanding the Statistical Measures](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [Cohen's d](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [P-Value](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
- [Interpreting the Results](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [Confidence Intervals](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [General Interpretation Guidelines](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [Limitations and Considerations](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
    - [Actionable Steps Based on Results](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)
- [Conclusion](https://www.notion.so/Bias-Analysis-of-Word-Embeddings-in-Language-Models-127091ff170880199e49c5c1f33e3755?pvs=21)

## Overview

The goal of this project is to analyze and quantify biases present in word embeddings produced by different language models. It specifically examines biases related to gender, but the methodology can be extended to other types of biases (e.g., racial, age-related). By computing statistical measures, the script evaluates how closely associated certain target words (e.g., male and female terms) are with specific attribute words (e.g., career and family terms).

## Design and Methodology

This is the approach: 

### 1. Load configurations and data

- **Model Configurations**: The script loads configurations for different language models from a json file (`models_config.json`). Each model includes:
  - Name provider: Cohere, Voyage, or OpenAI 
  - Model name (for some providers, I use different models)
  - API key
- **Word Lists**: There are target words attribute words and templates
- **Target Words** (`target_words_male.txt`, `target_words_female.txt`, etc ):
  - Male Terms: Words like 'he', 'him', 'his', 'man', 'boy', 'male', etc. 
  - Female Terms: Words like 'she', 'her', 'hers', 'woman', 'girl', 'female', etc. 
  - **Attribute Words** (`attribute_words_career.txt`,attribute_words_family.txt`, etc) :
    - Attribute Groups: 'career', 'family', 'leadership', 'support', 'intelligence', 'physical_appearance'. 
      - Each attribute group has its own list of associated words (e.g., 'career' might include 'engineer', 'doctor', 'lawyer'). 
- **Templates**: (`templates.json`): Templates are sentences with placeholders ("The role of a {} is crucial") where target or attribute words will be inserted. 
### 2. Generating Sentences
We generate two types of sentences:

1. **Target Sentences**
    - **Purpose**: To provide contextual embeddings for **target words** (e.g., male and female terms) across various bias domains.
    - **Generation Method**:
        - **Target Words**: Words like 'he', 'she', 'man', 'woman', etc.
        - **Templates**: **All templates from all attribute categories** are combined.
        - **Process**: Each target word is inserted into every template from every attribute category.
    - **Example**:
        - For the target word **'he'**:
            - "The role of a **man** is crucial in any home."
            - "**he** demonstrates exceptional leadership qualities."
            - "What defines the attractiveness of a **man**?"
            - *(Using templates from 'career', 'leadership', 'intelligence', etc.)*
        - For the target word **'she'**:
            - "The role of a **woman** is crucial in any home."
            - "**she** demonstrates exceptional leadership qualities."
            - "What defines the attractiveness of a **woman**?"
            - *(Using the same templates as above)*
2. **Attribute Sentences**
    - **Purpose**: To provide contextual embeddings for **attribute words**, specific to their bias category.
    - **Generation Method**:
        - **Attribute Words**: Words specific to each attribute category (e.g., 'engineer' for 'career', 'beautiful' for 'physical_appearance').
        - **Templates**: Templates specific to each attribute category.
        - **Process**: Each attribute word is inserted into the templates of its own attribute category.
    - **Example**:
        - For the attribute word **'beautiful'** in the **'physical_appearance'** category:
            - "A **beautiful** is recognized for their striking appearance."
            - "Being a **beautiful** involves maintaining a certain image."
            - *(Using 'physical_appearance_templates' only)*

### 3. Fetching Embeddings

- **Embeddings**: The script generates the embeddings for the sentence using the selected model: Cohere, OpenAI, Voyage

### 4. Bias Analysis

Once we have the embeddings, we can start the bias analysis

### 4.1 Cosine Similarity

- **Definition**:  Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space, providing a similarity score ranging from -1 (exactly opposite) to 1 (exactly the same). A higher cosine similarity indicates a greater similarity between the two vectors.
- **Usage**:  The script computes the cosine similarity between embeddings of target words (e.g., "he", "she") and attribute words (e.g., "engineer", "nurse") to measure their association in the embedding space.

<details>
<summary>Example</summary>
Suppose we have the following embeddings:

1. Get embeddings: 
   - Target word: "she"
     - Embedding vector $T: [0.2, 0.5, 0.1]$
   - Attribute word: "nurse"
     - Embedding vector $A: [0.3, 0.4, 0.2]$
2. Calculate Cosine Similarity: $$ Cosine Similarity = { T⋅A \over ∥T∥×∥A∥  } $$
  Cosine Similarity between "she" and "nurse": $0.9487$ 
3. Repeat for all words: This would happen also for 'he' and 'nurse' e.g.: Cosine Similarity between "he" and "nurse": $0.7394$

</details>

### 4.2 Association Scores

- **Computation**: For each target word embedding, the script calculates two association scores:
    - The mean cosine similarity with attribute group 1 (e.g., career terms).
    - The mean cosine similarity with attribute group 2 (e.g., family terms).
  - **Association Difference**: The difference between these two means provides the association score for that target word:
  
    - Association Score= Mean Similarity with Attribute Group 1 (e.g. career) − Mean Similarity with Attribute Group 2(e.g. family) 

### 4.3 Effect Size (Cohen's d)

- **Definition**: Cohen's d describes the standardized difference between two means 
- **Computation**: The effect size is calculated using the association scores of two target groups (e.g., male and female).
- **Formula**: $$ d = {X1−X2 \over Sp} $$
where:
  - $X1$ Collect all association scores for male target words and compute their mean
  - $X2$ Collect all association scores for female target words and compute their mean
  - $P$ is the pooled standard deviation

<details>
<summary>Example</summary>

### Dummy data:
1. Target Words:
    - Male Terms: 'he', 'man'
    - Female Terms: 'she', 'woman'
2. Attribute Words:
    - Attribute Group 1 (Career): 'engineer', 'doctor'
    - Attribute Group 2 (Family): 'parent', 'caregiver'

### **Generating Sentences:**

**Target Sentences for 'he':**

- "The role of a **man** is crucial in any home." (career group)
- "**he** demonstrates exceptional leadership qualities." (leadership group)
- "What defines the attractiveness of a **man**?" (physical appareance group)
- (and so on, using all templates)

**Attribute Sentences for 'engineer' (Career):**

- "The role of a **engineer** is crucial in any organization."
- "A **manager** is known for their problem-solving abilities."
- (using 'career_templates')

**Attribute Sentences for 'parent' (Family):**

- "The commitment of a **parent** enhances the stability of the household."
- "A **sibiling** plays a key part in the upbringing and education of children."
- (using 'family_templates')

### **Computing Cosine Similarities:**

- **For each target sentence embedding**, compute cosine similarities with each **attribute sentence embedding** in:
    - **Attribute Group 1 (Career)**
    - **Attribute Group 2 (Family)**

### **Calculating Association Scores:**

- **Average Similarity of 'he' with Career:**
    - Suppose we get an average cosine similarity of **0.75**.
- **Average Similarity of 'he' with Family:**
    - Suppose we get an average cosine similarity of **0.65**.
- **Association Score** for 'he': $$ Association Score(he) = 0.75−0.65 = 0.10 $$

Repeat this process for each target word.

### Compiling Association Scores

- **Male Target Words**:
  - 'he': 0.10
  - 'man': 0.12

- **Female Target Words:**
  - 'she': -0.08
  - 'woman': -0.07 

### Calculating Effect Size

**Means:**
- Mean for Male terms (X1):   $$X1 = {0.10 +0.12 \over 2 } = 0.11 $$
- Mean for Female terms (X1): $$X2 = {0.08 + (-0.0.7 \over 2 } =- 0.075 $$

**Pooled Standard Deviation (Sp)**: Calculate the standard deviation for each group and then compute the pooled standard deviation.
**Cohen's d:** $$ d = {0.11 - (-0.075) \over  Sp} ={ 0.185 \over Sp}$$
 
Assuming $Sp = 0.05$, then $$d = {0.185 \over 0.05} = 3.7 $$ 
An effect size of 3.7 indicates a very large bias.

### Interpreting the Results
**Direction of Bias:**
- Positive Effect Size: Indicates that male terms are more associated with career attributes than female terms.

**Magnitude of Bias:**
- An effect size greater than 0.8 is considered large.
</details>


### 4.4 Permutation Testing and P-Value

- **Purpose**: To determine if the observed effect size is statistically significant—that is, whether the bias observed is unlikely to have occurred by random chance.
- **Methodology**:
    1. **Permutation**: The association scores from both target groups are combined into a single dataset. This combined dataset is then randomly shuffled many times (e.g., 10,000 permutations).
    2. **Effect Size Distribution**: For each permutation, the effect size (Cohen's d) is recalculated. This process generates a distribution of effect sizes that represent what we might expect to observe if there were no actual bias (the null hypothesis).
    3. **P-Value Computation**: The p-value is calculated as the proportion of permuted effect sizes that are equal to or more extreme than the observed effect size. A low p-value indicates that the observed effect size is unlikely to have occurred by chance, suggesting a statistically significant bias.
- **Interpretation of P-Value**:
    - **p ≤ 0.05**: The observed bias is statistically significant. There is strong evidence against the null hypothesis (no bias).
    - **p > 0.05**: The observed bias is not statistically significant. There is insufficient evidence to conclude that a real bias exists.

### 4.5 Interpreting Effect Size

- **Interpretation**: Based on the magnitude of Cohen's d, the script categorizes the level of bias.
    - **Negligible/No Bias**: 0.0≤∣d∣<0.2
        
        0.0≤∣d∣<0.20.0 \leq |d| < 0.2
        
    - **Small Bias**: 0.2≤∣d∣<0.5
        
        0.2≤∣d∣<0.50.2 \leq |d| < 0.5
        
    - **Medium Bias**: 0.5≤∣d∣<0.8
        
        0.5≤∣d∣<0.80.5 \leq |d| < 0.8
        
    - **Large Bias**: ∣d∣≥0.8
        
        ∣d∣≥0.8|d| \geq 0.8
        
- **Direction of Bias**:
    - **Positive Effect Size**: Indicates that the first target group (e.g., male terms) is more closely associated with the first attribute group (e.g., career) compared to the second target group (e.g., female terms).
    - **Negative Effect Size**: Indicates that the second target group (e.g., female terms) is more closely associated with the first attribute group compared to the first target group.

### 5. Visualization

- **Bias Comparison Plot**: The script generates diverging bar charts to visualize the effect sizes (bias levels) across different models and bias categories.
    - **Color Coding**: Bars are colored based on the bias interpretation:
        - **Blue**: Negligible/No Bias
        - **Green**: Small Bias
        - **Yellow**: Medium Bias
        - **Red**: Large Bias
    - **Annotations**: Effect sizes and p-values are displayed near the bars to indicate statistical significance.
    - **Threshold Lines**: Vertical lines indicate the thresholds for small, medium, and large biases according to Cohen's d.
- **Point Estimates with Confidence Intervals Plot**: Displays the effect size point estimates for each model along with error bars representing the 95% confidence intervals derived from the permutation tests.
    - **Interpretation**:
        - **Error Bars Not Crossing Zero**: Suggest that the bias is statistically significant.
        - **Error Bars Crossing Zero**: Suggest that the bias may not be statistically significant.

## Metrics Used

1. **Cosine Similarity**: Measures similarity between two embeddings.
2. **Association Score**: Difference in mean similarities between a target word and two attribute groups.
3. **Effect Size (Cohen's d)**: Standardized measure of bias magnitude.
4. **P-Value**: Probability of observing the effect size under the null hypothesis (no bias).

## Understanding the Statistical Measures

### Cohen's d

- **Purpose**: Quantifies the difference between two groups in terms of standard deviation units.
- **Interpretation**:
    - **Negligible/No Bias**: 0.0≤∣d∣<0.2
        
        0.0≤∣d∣<0.20.0 \leq |d| < 0.2
        
    - **Small Bias**: 0.2≤∣d∣<0.5
        
        0.2≤∣d∣<0.50.2 \leq |d| < 0.5
        
    - **Medium Bias**: 0.5≤∣d∣<0.8
        
        0.5≤∣d∣<0.80.5 \leq |d| < 0.8
        
    - **Large Bias**: ∣d∣≥0.8
        
        ∣d∣≥0.8|d| \geq 0.8
        
- **Significance**: A larger absolute value of Cohen's d indicates a greater bias.

### P-Value

- **Definition**: The probability of obtaining an effect size as extreme as the observed one, assuming the null hypothesis is true.
- **Permutation Testing**: By simulating the null distribution through permutations, we obtain a more accurate p-value without relying on parametric assumptions.
- **Interpretation**:
    - **Low P-Value (p≤0.05p \leq 0.05p≤0.05)**: Indicates that the observed bias is statistically significant.
    - **High P-Value (p>0.05p > 0.05p>0.05)**: Suggests that the observed bias could be due to chance.

## Interpreting the Results

Understanding the results of bias analysis is crucial for drawing meaningful conclusions and making informed decisions. This section provides detailed explanations of the statistical methods used and guides you on how to interpret the outputs generated by the script.

### Confidence Intervals

- **Purpose**: To provide a range within which the true effect size is likely to fall, offering insight into the precision of the estimated effect size.
- **Calculation**: Confidence intervals are derived from the distribution of effect sizes obtained through permutation testing.
- **Interpreting Confidence Intervals**:
    - **95% Confidence Interval (CI)**: There is a 95% chance that the true effect size lies within this interval.
    - **Confidence Interval Not Including Zero**: Suggests that the effect size is statistically significant.
    - **Confidence Interval Including Zero**: Suggests that the effect size may not be statistically significant.

### General Interpretation Guidelines

- **Statistical Significance vs. Practical Significance**:
    - An effect size can be statistically significant (low p-value) but practically small (low Cohen's d). Consider both the statistical significance and the effect size magnitude when interpreting results.
- **Consistency Across Models**:
    - Consistent biases observed across multiple models strengthen the evidence of a systemic issue.
- **Direction of Bias**:
    - Understanding whether the bias favors one group over another is crucial for addressing and mitigating potential impacts.
- **Confidence Intervals Width**:
    - **Narrow Confidence Intervals**: Indicate more precise estimates.
    - **Wide Confidence Intervals**: Suggest greater uncertainty in the effect size estimate.

### Limitations and Considerations

- **Sample Size**:
    - The number of target and attribute words can affect the reliability of the results. Ensure that word lists are comprehensive and representative.
- **Template Influence**:
    - The context provided by sentence templates can influence the embeddings. Use neutral and varied templates to minimize contextual bias.
- **Multiple Comparisons**:
    - Conducting multiple tests increases the risk of Type I errors (false positives). Consider applying corrections like the Bonferroni method if appropriate.
- **Model Variability**:
    - Differences in model architectures and training data can affect bias measurements. Interpret results within the context of each model's characteristics.
- **Statistical Assumptions**:
    - While permutation tests are non-parametric, assumptions about the independence of observations still apply.

### Actionable Steps Based on Results

- **Identifying Bias**:
    - Significant and substantial biases indicate areas where model adjustments or data interventions may be necessary.
- **Mitigating Bias**:
    - Use the insights gained to implement bias mitigation techniques, such as re-training with debiased data, applying post-processing corrections, or using bias regularization during training.
- **Policy Development**:
    - Incorporate findings into AI governance frameworks to establish guidelines and best practices for model development and deployment.
- **Further Research**:
    - Investigate the sources of bias, such as training data or model architecture, to inform future improvements.

## Conclusion

This script provides a comprehensive analysis of biases present in word embeddings from various language models. By combining statistical measures and permutation testing, it quantifies biases and assesses their significance. The visualizations aid in comparing biases across models, helping researchers and practitioners understand and address biases in language models.

Understanding and interpreting these results is essential for:

- **Model Improvement**: Enhancing fairness and reducing unintended biases in AI systems.
- **Ethical AI Practices**: Promoting transparency and accountability in AI development.
- **Informed Decision-Making**: Guiding stakeholders in making data-driven decisions regarding AI deployment and policy.