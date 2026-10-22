
# Baseline Models and Analysis

## 1. Project Overview

This project focuses on detecting Personally Identifiable Information (PII)
in financial documents using token-level Named Entity Recognition (NER).

The task is formulated as a sequence-labeling problem. Each token receives
a BIO tag indicating whether it is outside an entity, at the beginning of
an entity, or inside an entity.

Two baseline models were implemented and evaluated:

1. Most Frequent baseline
2. Hidden Markov Model (HMM) with Viterbi decoding

## 2. Dataset Preparation

The dataset is derived from the
`gretelai/synthetic_pii_finance_multilingual` dataset.

English-language records were selected. PII labels were mapped into
eight categories:

- NAME
- COMPANY
- ADDRESS
- DATE
- EMAIL
- PHONE
- ACCOUNT
- ID

The data was tokenized and converted into BIO format.

The prepared dataset contains the following splits:

| Split | Chunks |
|---|---:|
| Train | 30,545 |
| Validation | 3,389 |
| Test | 3,903 |

Each chunk contains a token sequence and a corresponding tag sequence.
Chunks contain at most 200 tokens.

The training data was divided into training and validation subsets.
The original test split was preserved for final evaluation.

## 3. Evaluation Methodology

Both models were evaluated using token-level classification metrics.

The following metrics were reported:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Per-label precision, recall, and F1-score

Macro F1 assigns equal weight to each evaluated class, making it useful
for examining performance across imbalanced labels.

Weighted F1 accounts for the number of true instances in each class.
Accuracy and weighted F1 must be interpreted carefully because the
outside-entity label (`O`) dominates the dataset.

## 4. Model 1: Most Frequent Baseline

The Most Frequent model learns the most common training label for each
lowercased token.

If a token was not observed during training, the model assigns the
globally most frequent training label.

This is a simple lexical baseline. It does not explicitly model the
relationships between neighboring labels.

### Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9493 |
| Macro Precision | 0.6345 |
| Macro Recall | 0.3438 |
| Macro F1 | 0.4166 |
| Weighted F1 | 0.9419 |

The model achieves high accuracy, but its macro recall is considerably
lower. This indicates that the model does not identify all PII classes
equally well.

## 5. Model 2: HMM + Viterbi

The Hidden Markov Model estimates:

- Initial tag probabilities
- Tag transition probabilities
- Token emission probabilities

Laplace smoothing is used to handle unseen events. Token strings are
lowercased before emission statistics are calculated.

The Viterbi algorithm finds the highest-scoring tag sequence using
dynamic programming.

BIO transition constraints prevent invalid transitions into an
inside-entity tag. For example, `I-EMAIL` cannot directly follow `O`.

### Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9532 |
| Macro Precision | 0.5260 |
| Macro Recall | 0.5096 |
| Macro F1 | 0.4688 |
| Weighted F1 | 0.9550 |

The HMM model improves macro F1 and macro recall compared with the
Most Frequent baseline. However, its macro precision is lower, indicating
that some of its additional positive predictions are incorrect.

## 6. Overall Model Comparison

| Split | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Validation | Most Frequent | 0.9479 | 0.6814 | 0.3618 | 0.4397 | 0.9402 |
| Validation | HMM + Viterbi | 0.9506 | 0.5007 | 0.5137 | 0.4600 | 0.9526 |
| Test | Most Frequent | 0.9493 | 0.6345 | 0.3438 | 0.4166 | 0.9419 |
| Test | HMM + Viterbi | 0.9532 | 0.5260 | 0.5096 | 0.4688 | 0.9550 |

On the test set, HMM + Viterbi improves macro F1 from 0.4166 to 0.4688,
an increase of approximately 5.22 percentage points.

Test accuracy increases by 0.39 percentage points, while weighted F1
increases by approximately 1.31 percentage points.

The macro F1 improvement is particularly relevant because it indicates
better aggregate performance across the evaluated classes, rather than
performance being dominated by the majority class.

## 7. Per-Label Test F1 Comparison

| Label | Most Frequent F1 | HMM F1 | Difference |
|---|---:|---:|---:|
| B-ACCOUNT | 0.0674 | 0.0926 | +0.0252 |
| B-ADDRESS | 0.5439 | 0.7196 | +0.1757 |
| B-COMPANY | 0.5724 | 0.6531 | +0.0806 |
| B-DATE | 0.6574 | 0.7322 | +0.0748 |
| B-EMAIL | 0.4169 | 0.3804 | -0.0366 |
| B-ID | 0.0819 | 0.0814 | -0.0005 |
| B-NAME | 0.6243 | 0.7168 | +0.0925 |
| B-PHONE | 0.6148 | 0.3722 | -0.2426 |
| I-ACCOUNT | 0.0800 | 0.0112 | -0.0688 |
| I-ADDRESS | 0.4832 | 0.7489 | +0.2658 |
| I-COMPANY | 0.4475 | 0.6309 | +0.1834 |
| I-DATE | 0.5477 | 0.7182 | +0.1705 |
| I-EMAIL | 0.0000 | 0.0000 | 0.0000 |
| I-ID | 0.0000 | 0.0000 | 0.0000 |
| I-NAME | 0.5151 | 0.7626 | +0.2475 |
| I-PHONE | 0.4526 | 0.3720 | -0.0806 |

### Observations

The largest improvements occur in:

- I-ADDRESS
- I-NAME
- B-ADDRESS
- I-DATE
- I-COMPANY

These results suggest that sequence modeling helps the HMM identify
continuations of multi-token entities and some entity boundaries.

However, the HMM performs worse on B-PHONE, I-PHONE, and B-EMAIL.
ACCOUNT and ID detection also remains weak.

The results indicate that sequence modeling alone does not solve the
challenges associated with rare or difficult entity types.

## 8. Limitations

1. The Most Frequent baseline relies on token-label frequencies and
   does not explicitly model sequence structure.

2. The HMM uses unigram token emissions and first-order label transitions.
   It does not directly model richer contextual representations.

3. Lowercasing removes capitalization information that may help identify
   names and organizations.

4. Class imbalance affects the interpretation of aggregate metrics.

5. Some entity labels have very low support, making their F1-scores
   unstable or difficult to interpret.

6. Token-level metrics do not directly measure exact entity-span
   precision and recall.

## 9. Conclusion

HMM + Viterbi outperforms the Most Frequent baseline on test accuracy,
macro recall, macro F1, and weighted F1.

The macro F1 improvement from 0.4166 to 0.4688 suggests that explicitly
modeling tag transitions is beneficial for this token-level PII detection
task.

Nevertheless, performance remains uneven across entity categories.
Future work should focus on improving phone, email, account, and ID
detection, while evaluating changes using the validation split.

The test set should remain reserved for final model comparison rather
than repeated hyperparameter tuning.
