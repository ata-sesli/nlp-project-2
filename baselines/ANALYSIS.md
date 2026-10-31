
# Baseline Models and HMM-Viterbi Analysis

## 1. Project Overview

This project focuses on Named Entity Recognition (NER) using BIO tagging. The objective is to identify personal and financial information entities in text sequences.

Two baseline approaches were implemented and evaluated:

1. **Most Frequent Baseline:** Predicts the most frequent tag observed for each token.
2. **HMM with Viterbi Decoding:** Uses a first-order Hidden Markov Model (HMM), emission and transition probabilities, word-shape features, and the Viterbi algorithm to predict the most likely tag sequence.

Both models were evaluated using token-level and entity-level metrics. The final model comparison uses the project's shared benchmark evaluator with strict IOB2 entity-level evaluation.

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

The models are evaluated using token-level and entity-level metrics.

### 4.1 Token-Level Evaluation

Token-level evaluation compares the predicted BIO tag with the corresponding gold BIO tag for every token.

The following metrics are used:

- **Accuracy:** Proportion of correctly predicted tokens.
- **Precision:** Proportion of predicted positive labels that are correct.
- **Recall:** Proportion of actual positive labels that are correctly identified.
- **Macro F1:** Average F1 score across labels, giving each label equal weight.
- **Weighted F1:** Average F1 score weighted by the number of instances belonging to each label.

Accuracy must be interpreted carefully because the outside-entity label (`O`) dominates the dataset. A model can achieve high token accuracy while still performing poorly at identifying complete entities.

For this reason, token-level metrics alone are not sufficient to evaluate NER performance.

### 4.2 Entity-Level Evaluation

Entity-level evaluation measures whether the model correctly identifies complete entities.

An entity is considered correct only when all three conditions match:

1. Entity type.
2. Start token position.
3. End token position.

For example, predicting the correct entity type but an incorrect boundary is counted as an error.

The shared benchmark evaluator uses strict IOB2 entity-level evaluation through the `seqeval` library. This ensures that entity boundaries and entity types are evaluated consistently across all models.

The evaluator also checks prediction and gold records, including their identifiers and sequence lengths, to ensure that predictions are compared with the corresponding reference sequences.

The following metrics are reported:

- **Entity Precision:** Correctly predicted entities divided by all predicted entities.
- **Entity Recall:** Correctly predicted entities divided by all gold entities.
- **Entity F1:** Harmonic mean of entity precision and recall.
- **Missed PII:** Proportion of gold entities that were not correctly identified.

Entity-level evaluation provides a more meaningful measure of NER performance because it considers complete entity boundaries rather than individual token predictions.

### 4.3 Error Analysis

The shared benchmark also categorizes entity-level errors:

- **Correct:** Entity is identified with the correct type and boundaries.
- **Boundary:** Entity type is correct, but its boundaries are incorrect.
- **Type:** Entity boundaries match, but the predicted entity type is incorrect.
- **Boundary and type:** Both entity boundaries and type are incorrect.
- **Missed:** A gold entity is not identified.
- **Invented:** A predicted entity does not correspond to a gold entity.

These categories help explain why a model may have a relatively high token accuracy but a lower entity-level F1 score.

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
- Observations represent input tokens and their derived word-shape features.
- Transition probabilities model relationships between consecutive tags.
- Emission probabilities model the likelihood of observing a token feature given a tag.
- Initial probabilities model the likelihood of starting a sequence with a particular tag.

The model uses a first-order Markov assumption: the current hidden state depends on the previous hidden state. The observation is modeled according to the current hidden state.

### 6.1 Initial Probabilities

Initial probabilities represent the likelihood of a tag appearing at the beginning of a sequence.

For example, a sequence is more likely to begin with `O` or a `B-TYPE` tag than with an `I-TYPE` tag.

BIO constraints are applied to prevent invalid entity continuation tags at the beginning of a sequence.

### 6.2 Transition Probabilities

Transition probabilities represent the likelihood of one tag following another.

For example:

- `B-NAME` can be followed by `I-NAME`.
- `I-NAME` can be followed by `I-NAME` or another valid tag.
- `O` can be followed by `O` or a `B-TYPE` tag.

The model applies BIO transition constraints to prevent invalid transitions between entity types.

### 6.3 Emission Probabilities

Emission probabilities represent the likelihood of observing a token feature given a particular tag.

For example, a token frequently observed with the `B-NAME` tag will contribute to a higher emission probability for that tag.

The model uses lowercased tokens and word-shape features to represent observations. This allows it to use both lexical information and patterns in token structure.

### 6.4 Word-Shape Features

Word-shape features were introduced to improve the handling of rare and previously unseen tokens.

Tokens that occur frequently enough in the training data retain their lexical representation. Rare or unseen tokens are represented using word-shape categories.

The implemented feature extraction identifies patterns such as:

- Email-like strings.
- IBAN-like strings.
- Phone-number-like strings.
- Numeric sequences.
- URL-like strings.
- Uppercase, lowercase, and title-case patterns.
- General character-level patterns.

For example, an email address can be represented by an email-related feature rather than being treated as an entirely unrelated unknown word.

This approach helps the HMM generalize beyond exact token occurrences. It is particularly relevant for financial and personal information, where many entities contain structured numeric or alphanumeric patterns.

The feature extraction is based on the input token itself and does not use validation or test labels.

### 6.5 Additive Smoothing

Additive smoothing is applied with an alpha value of 0.1.

Smoothing prevents unseen events from receiving zero probability. Without smoothing, a single zero-probability transition or emission could make an entire sequence impossible under the model.

The smoothing is applied to emission probabilities so that unobserved tag-feature combinations can still receive nonzero probabilities.

## 7. Viterbi Decoding

The Viterbi algorithm is a dynamic programming algorithm used to find the most probable sequence of hidden states.

In this project, it identifies the most likely BIO tag sequence for a given token sequence.

Instead of evaluating every possible tag sequence independently, Viterbi decoding stores the best score for each possible current tag at each position.

The algorithm consists of three main stages:

1. **Initialization:** Calculate the initial scores for the first token.
2. **Recursion:** Calculate the best possible path to each tag at every subsequent position.
3. **Backtracking:** Recover the most likely complete tag sequence.

The HMM and Viterbi algorithm work together: the HMM provides the probability distributions, while Viterbi finds the most probable sequence according to those distributions.

## 8. Shared Benchmark Results

The final comparison was performed using the shared benchmark evaluator. All models were evaluated using the same test data and entity-level evaluation methodology.

### 8.1 Overall Test Results

| Model | Precision | Recall | Entity F1 | Macro F1 | Token Accuracy | Missed PII |
|---|---:|---:|---:|---:|---:|---:|
| Most Frequent | 0.397 | 0.279 | 0.327 | 0.251 | 0.949 | 0.721 |
| HMM + Viterbi | 0.630 | 0.564 | 0.595 | 0.528 | 0.960 | 0.436 |
| SVM | 0.802 | 0.728 | 0.763 | 0.780 | 0.980 | 0.272 |
| CRF | 0.859 | 0.789 | 0.822 | 0.825 | 0.981 | 0.211 |
| BiLSTM-CRF | 0.864 | 0.835 | 0.849 | 0.847 | 0.985 | 0.165 |
| BERT | 0.860 | 0.877 | 0.868 | 0.885 | 0.987 | 0.123 |

### 8.2 Overall Results Interpretation

The HMM model achieves a test entity F1 score of 0.5950, compared with 0.3274 for the Most Frequent baseline.

This represents an improvement of approximately 26.76 percentage points in entity F1.

The HMM also improves both precision and recall:

- Precision increases from 0.397 to 0.630.
- Recall increases from 0.279 to 0.564.
- Missed PII decreases from 0.721 to 0.436.

These results indicate that modeling relationships between consecutive BIO tags and using word-shape features provide a substantial improvement over the token-frequency baseline.

The HMM achieves 0.960 token accuracy, but its entity F1 is 0.595. This difference demonstrates why token accuracy alone is not sufficient for NER evaluation.

The HMM performs below SVM, CRF, BiLSTM-CRF, and BERT. These models use richer feature representations or more advanced contextual modeling approaches, which can help them identify complex entity patterns.

### 8.3 Validation and Test Entity F1

| Model | Validation Entity F1 | Test Entity F1 |
|---|---:|---:|
| Most Frequent | 0.3079 | 0.3274 |
| HMM + Viterbi | 0.5795 | 0.5950 |
| SVM | 0.7556 | 0.7631 |
| CRF | 0.8175 | 0.8221 |
| BiLSTM-CRF | 0.8457 | 0.8492 |
| BERT | 0.8612 | 0.8683 |

The HMM achieves a slightly higher entity F1 on the test set than on the validation set. The small difference indicates that its overall performance is relatively consistent across these two splits.

## 9. Per-Entity-Type Test F1

The following table compares entity-level F1 scores for each entity category on the test set.

| Entity Type | Most Frequent F1 | HMM + Viterbi F1 | SVM F1 | CRF F1 | BiLSTM-CRF F1 | BERT F1 |
|---|---:|---:|---:|---:|---:|---:|
| ACCOUNT | 0.060 | 0.236 | 0.725 | 0.714 | 0.789 | 0.874 |
| ADDRESS | 0.041 | 0.553 | 0.682 | 0.864 | 0.895 | 0.889 |
| COMPANY | 0.274 | 0.487 | 0.633 | 0.740 | 0.789 | 0.825 |
| DATE | 0.474 | 0.628 | 0.813 | 0.846 | 0.862 | 0.869 |
| EMAIL | 0.417 | 0.699 | 0.940 | 0.941 | 0.918 | 0.927 |
| ID | 0.082 | 0.236 | 0.784 | 0.784 | 0.783 | 0.918 |
| NAME | 0.320 | 0.713 | 0.808 | 0.831 | 0.862 | 0.877 |
| PHONE | 0.338 | 0.668 | 0.854 | 0.881 | 0.876 | 0.900 |

### 9.1 Per-Type Interpretation

The HMM improves over the Most Frequent baseline in all eight entity categories.

The strongest HMM results are observed for:

- NAME: 0.713
- EMAIL: 0.699
- PHONE: 0.668
- DATE: 0.628

The weakest results are observed for:

- ACCOUNT: 0.236
- ID: 0.236
- COMPANY: 0.487

ACCOUNT and ID remain particularly challenging. These categories can contain long numeric or alphanumeric sequences, and their recognition may require more detailed pattern representations and contextual information.

The results also show that word-shape features do not eliminate the limitations of the HMM. Although the model improves substantially over the Most Frequent baseline, it remains behind the more advanced models in every entity category.

## 10. Error Analysis

The shared benchmark reports the following test-set error counts:

| Model | Correct | Boundary | Type | Boundary and Type | Missed | Invented |
|---|---:|---:|---:|---:|---:|---:|
| Most Frequent | 4,332 | 4,002 | 54 | 329 | 6,824 | 1,186 |
| HMM + Viterbi | 8,764 | 3,275 | 61 | 521 | 2,920 | 1,734 |
| SVM | 11,312 | 1,516 | 80 | 115 | 2,518 | 1,019 |
| CRF | 12,257 | 876 | 82 | 132 | 2,194 | 982 |
| BiLSTM-CRF | 12,981 | 752 | 118 | 113 | 1,577 | 1,111 |
| BERT | 13,632 | 677 | 55 | 91 | 1,086 | 1,375 |

### 10.1 HMM Error Interpretation

The HMM correctly identifies 8,764 entities.

Its main error categories are:

- Boundary errors: 3,275
- Missed entities: 2,920
- Invented entities: 1,734
- Boundary and type errors: 521
- Type errors: 61

Boundary errors are the most frequent error category among the reported HMM errors.

This suggests that the model often recognizes relevant entity information but does not always determine the exact start and end positions correctly.

The HMM also misses 2,920 entities. This is consistent with its recall of 0.564 and missed PII rate of 0.436.

Compared with the Most Frequent baseline, the HMM produces substantially fewer missed entities and more correct predictions. However, its boundary and invented-entity errors show that the model still has difficulty producing exact entity spans.

## 11. Limitations

The implemented models provide useful baselines, but they have several limitations.

### Most Frequent Baseline

- Does not consider token context.
- Does not model relationships between consecutive tags.
- Relies heavily on token frequency.
- Can produce incorrect entity boundaries.
- Cannot generalize effectively to many unseen token patterns.

### HMM + Viterbi

- Uses a first-order Markov assumption.
- Has limited contextual representation.
- Relies on token-level emission and word-shape features.
- May struggle with rare or previously unseen entity patterns.
- Additive smoothing helps with unseen events but does not solve data sparsity completely.
- Word-shape features cannot fully represent the semantic and contextual differences between entities.
- Exact entity boundaries remain a major source of error.

These limitations help explain why the HMM performs better than the Most Frequent baseline but below the more advanced sequence-labeling models.

## 12. Conclusion

The experiments demonstrate that the HMM with Viterbi decoding performs substantially better than the Most Frequent baseline in entity-level NER evaluation.

The HMM achieves a test entity F1 score of 0.5950, compared with 0.3274 for the Most Frequent baseline. It also reduces the missed PII rate from 0.721 to 0.436.

The addition of word-shape features provides the model with information about structured tokens, including email-like, numeric, phone-like, and IBAN-like patterns. This improves its ability to handle rare and previously unseen tokens.

However, the HMM still struggles with ACCOUNT and ID entities, and boundary errors remain its most frequent reported error category.

The comparison with SVM, CRF, BiLSTM-CRF, and BERT shows that more advanced models achieve higher entity-level F1 scores on this dataset.

Overall, the HMM with Viterbi decoding provides a meaningful statistical baseline. It demonstrates the value of probabilistic sequence modeling and BIO transition constraints while highlighting the limitations of first-order HMMs and relatively simple observation representations for complex NER tasks.
