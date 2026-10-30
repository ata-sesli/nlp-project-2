
# Baseline Models and HMM-Viterbi Analysis

## 1. Project Overview

This project focuses on Named Entity Recognition (NER) using BIO tagging. The objective is to identify personal and financial information entities in text sequences.

Two baseline approaches were implemented and evaluated:

1. **Most Frequent Baseline:** Predicts the most frequent tag observed for each token.
2. **HMM with Viterbi Decoding:** Uses a first-order Hidden Markov Model (HMM) and the Viterbi algorithm to predict the most likely tag sequence.

Both models were evaluated using token-level and entity-level metrics.

## 2. Dataset Preparation

The dataset was prepared from the English subset of the `gretelai/synthetic_pii_finance_multilingual` dataset.

The preparation process includes:

- Converting character-level entity annotations into token-level BIO tags.
- Using the BIO tagging scheme to represent entity boundaries.
- Splitting the training documents into training and validation sets before chunking.
- Preserving the original test source split.
- Removing duplicate training documents that overlap with the test split.
- Dividing long sequences into chunks with a maximum length of 200 tokens.
- Saving the processed data in compressed JSONL format.

The resulting dataset contains:

| Split | Number of chunks |
|---|---:|
| Train | 30,545 |
| Validation | 3,389 |
| Test | 3,903 |

The dataset contains 17 labels: one outside-entity label (`O`) and 16 BIO labels representing eight entity categories.

The entity categories are:

- NAME
- COMPANY
- ADDRESS
- DATE
- EMAIL
- PHONE
- ACCOUNT
- ID

The dataset preparation process is documented in `dataset/data/REPORT.md`.

## 3. BIO Tagging Scheme

The BIO scheme represents entity boundaries using three types of tags:

- `B-TYPE`: Beginning of an entity.
- `I-TYPE`: Continuation of an entity.
- `O`: Token that does not belong to an entity.

For example:

| Token | BIO tag |
|---|---|
| John | B-NAME |
| Smith | I-NAME |
| works | O |
| at | O |
| Acme | B-COMPANY |
| Corporation | I-COMPANY |

The BIO scheme is important because identifying an entity requires both recognizing its type and determining its exact boundaries.

## 4. Evaluation Methodology

The models were evaluated at two different levels.

### 4.1 Token-Level Evaluation

Token-level evaluation compares the predicted BIO tag with the corresponding gold BIO tag for every token.

The following metrics were calculated:

- **Accuracy:** Proportion of correctly predicted tokens.
- **Precision:** Proportion of predicted positive labels that are correct.
- **Recall:** Proportion of actual positive labels that are correctly identified.
- **Macro F1:** Average F1 score across labels, giving each label equal weight.
- **Weighted F1:** Average F1 score weighted by the number of instances belonging to each label.

Accuracy and weighted F1 must be interpreted carefully because the outside-entity label (`O`) dominates the dataset. A model can achieve high token accuracy while still performing poorly at identifying complete entities.

For this reason, token-level metrics alone are not sufficient to evaluate NER performance.

### 4.2 Entity-Level Evaluation

Entity-level evaluation measures whether the model correctly identifies complete entities.

An entity is considered correct only when all three conditions match:

1. Entity type.
2. Start token position.
3. End token position.

For example, predicting the correct entity type but an incorrect boundary is counted as an error.

The evaluation uses exact-span matching. Entity spans are extracted from BIO sequences and compared between predictions and gold annotations.

Invalid `I-TYPE` tags that do not continue an entity of the same type are interpreted as the beginning of a new entity of that type. This provides a consistent method for handling malformed BIO sequences.

The evaluator also checks that prediction and gold records have matching identifiers and sequence lengths. Record identifiers are included in entity matching to prevent entities from different records from being matched accidentally.

The following metrics are reported:

- **Entity Precision:** Correctly predicted entities divided by all predicted entities.
- **Entity Recall:** Correctly predicted entities divided by all gold entities.
- **Entity F1:** Harmonic mean of entity precision and recall.

Entity-level evaluation provides a more meaningful measure of NER performance because it considers complete entity boundaries rather than individual token predictions.

## 5. Most Frequent Baseline

The Most Frequent baseline is a simple statistical approach.

During training, the model calculates the most frequently observed tag for each lowercased token.

During prediction:

1. The token is converted to lowercase.
2. The model looks up the most frequent tag associated with that token.
3. If the token has not been observed during training, the model predicts the globally most frequent tag.

This approach provides a simple reference point for evaluating more advanced models.

However, it does not consider the surrounding context or relationships between consecutive tags. Therefore, it may struggle to identify entity boundaries and distinguish between different uses of the same token.

## 6. Hidden Markov Model (HMM)

A Hidden Markov Model is a probabilistic model that represents a sequence using hidden states and observable outputs.

In this project:

- Hidden states represent BIO tags.
- Observations represent input tokens.
- Transition probabilities model relationships between consecutive tags.
- Emission probabilities model the likelihood of observing a token given a tag.
- Initial probabilities model the likelihood of starting a sequence with a particular tag.

The model assumes that the current hidden state depends on the previous hidden state and that the current observation depends on the current hidden state.

### 6.1 Initial Probabilities

Initial probabilities represent the likelihood of a tag appearing at the beginning of a sequence.

For example, a sequence is more likely to begin with `O` or a `B-TYPE` tag than with an `I-TYPE` tag.

### 6.2 Transition Probabilities

Transition probabilities represent the likelihood of one tag following another.

For example:

- `B-NAME` can be followed by `I-NAME`.
- `I-NAME` can be followed by `I-NAME` or another valid tag.
- `O` can be followed by `O` or a `B-TYPE` tag.

The model applies BIO transition constraints to prevent invalid transitions between entity types.

### 6.3 Emission Probabilities

Emission probabilities represent the likelihood of a token being generated by a particular tag.

For example, a token frequently observed with the `B-NAME` tag will have a higher emission probability for that tag.

Tokens are lowercased before emission lookup. An unknown-token bucket is used to handle tokens that were not observed during training.

### 6.4 Laplace Smoothing

Laplace smoothing is applied with an alpha value of 0.1.

Smoothing prevents unseen events from receiving zero probability. Without smoothing, a single zero-probability transition or emission could make an entire sequence impossible under the model.

## 7. Viterbi Decoding

The Viterbi algorithm is a dynamic programming algorithm used to find the most probable sequence of hidden states.

In this project, it identifies the most likely BIO tag sequence for a given token sequence.

Instead of evaluating every possible tag sequence independently, Viterbi decoding stores the best score for each possible current tag at each position.

The algorithm consists of three main stages:

1. **Initialization:** Calculate the initial scores for the first token.
2. **Recursion:** Calculate the best possible path to each tag at every subsequent position.
3. **Backtracking:** Recover the most likely complete tag sequence.

The HMM and Viterbi algorithm work together: the HMM provides the probability distributions, while Viterbi finds the most probable sequence according to those distributions.

## 8. Token-Level Results

The following table presents the token-level evaluation results.

| Model | Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Most Frequent | Validation | 0.9479 | 0.6814 | 0.3618 | 0.4397 | 0.9402 |
| Most Frequent | Test | 0.9493 | 0.6345 | 0.3438 | 0.4166 | 0.9419 |
| HMM + Viterbi | Validation | 0.9506 | 0.5007 | 0.5137 | 0.4600 | 0.9526 |
| HMM + Viterbi | Test | 0.9532 | 0.5260 | 0.5096 | 0.4688 | 0.9550 |

### 8.1 Token-Level Interpretation

The HMM model achieves higher test accuracy and macro F1 than the Most Frequent baseline.

The test macro F1 increases from approximately 0.4166 to 0.4688.

The HMM also achieves a higher macro recall, indicating that it identifies a larger proportion of instances across labels.

The Most Frequent baseline has higher macro precision, meaning that its positive predictions are more selective. However, its lower recall indicates that it misses a larger proportion of actual labels.

The difference between accuracy and macro F1 is particularly important. Because the `O` label is dominant, accuracy can remain high even when performance on less frequent entity labels is limited.

## 9. Entity-Level Results

Entity-level evaluation provides a stricter assessment of NER performance.

| Model | Split | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Most Frequent | Validation | 0.2207 | 0.2928 | 0.2517 |
| HMM + Viterbi | Validation | 0.5379 | 0.4811 | 0.5079 |
| Most Frequent | Test | 0.2318 | 0.3100 | 0.2653 |
| HMM + Viterbi | Test | 0.5469 | 0.4971 | 0.5208 |

### 9.1 Entity-Level Interpretation

The HMM model substantially outperforms the Most Frequent baseline in exact-span entity evaluation.

On the test set:

- Most Frequent entity F1: 0.2653
- HMM entity F1: 0.5208

This corresponds to an improvement of approximately 25.55 percentage points.

The HMM model also identifies more correct entities:

| Metric | Most Frequent | HMM + Viterbi |
|---|---:|---:|
| True Positives | 4,818 | 7,725 |
| False Positives | 15,968 | 6,401 |
| False Negatives | 10,723 | 7,816 |

The HMM model produces substantially fewer false positives while identifying more correct entities.

This suggests that modeling relationships between consecutive tags helps the model produce more coherent entity sequences.

## 10. Per-Entity-Type Test F1

The following table compares entity-level F1 scores for each entity category on the test set.

| Entity Type | Most Frequent F1 | HMM + Viterbi F1 |
|---|---:|---:|
| ACCOUNT | 0.0591 | 0.0826 |
| ADDRESS | 0.0235 | 0.4939 |
| COMPANY | 0.2309 | 0.4493 |
| DATE | 0.4051 | 0.5687 |
| EMAIL | 0.4142 | 0.3804 |
| ID | 0.0819 | 0.0814 |
| NAME | 0.2843 | 0.6183 |
| PHONE | 0.2462 | 0.3486 |

### 10.1 Per-Type Interpretation

The HMM model improves F1 for most entity categories.

The largest improvements are observed for:

- ADDRESS
- NAME
- COMPANY
- DATE

NAME achieves the highest HMM entity-level F1 score at approximately 0.6183.

However, performance remains limited for ACCOUNT and ID. These categories may be more difficult to identify because of their token patterns and contextual characteristics.

The HMM model also performs slightly worse than the Most Frequent baseline for EMAIL and ID. This indicates that the HMM approach does not improve every entity category equally.

These results suggest that further improvements may require better token representations, richer contextual features, or more advanced sequence-labeling models.

## 11. Limitations

The implemented models provide useful baselines, but they have several limitations.

### Most Frequent Baseline

- Does not consider token context.
- Does not model relationships between consecutive tags.
- Relies heavily on token frequency.
- Can produce incorrect entity boundaries.

### HMM + Viterbi

- Uses a first-order Markov assumption.
- Has limited contextual representation.
- Relies on token-level emission probabilities.
- May struggle with rare or previously unseen entity patterns.
- Laplace smoothing helps with unseen events but does not solve data sparsity completely.

Both models are statistical approaches and do not have the contextual representation capabilities of modern neural sequence-labeling models.

## 12. Conclusion

The experiments demonstrate that the HMM with Viterbi decoding performs better than the Most Frequent baseline in overall token-level and entity-level evaluation.

Although both models achieve high token accuracy, entity-level evaluation reveals a substantial difference in their ability to identify complete entities correctly.

The HMM model achieves a test entity-level F1 score of approximately 0.5208, compared with 0.2653 for the Most Frequent baseline.

This improvement indicates that modeling transition relationships between BIO tags is valuable for sequence labeling tasks.

Nevertheless, performance varies across entity categories. ACCOUNT and ID remain particularly challenging, while EMAIL shows a small decrease compared with the Most Frequent baseline.

Overall, the results establish the HMM with Viterbi decoding as a stronger statistical baseline for this NER task, while also highlighting the limitations of traditional probabilistic models for contextual entity recognition.
