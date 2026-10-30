# Ata Sesli: contribution

**Role:** benchmark and repository. I built the shared evaluation that scores all six models
the same way, wrote the main README and kept the repository together. My work is in
`benchmark/`, `README.md` and `LICENSE`.

## What I did

1. **Planned the project** (project plan v2): the task (PII detection as BIO sequence
   labeling), the dataset, the 8 PII types and 17 tags, the six models, the shared data and
   prediction formats, the metric choice and the work split.
2. **Wrote the shared evaluation** (`benchmark/evaluate.py`). It reads any model's prediction
   file, checks that every chunk has a prediction of the right length with valid tags, and
   computes:
   - entity precision, recall and micro F1 with seqeval in strict IOB2 mode, the ranking
     metric (an entity counts only if its boundaries and its type are both right);
   - token accuracy, reported to show why it is not used for ranking;
   - the missed PII rate (1 − recall);
   - F1 per PII type;
   - error types, following the two NER difficulties from the lecture: boundary error, type
     error, boundary and type error, missed entity, invented entity.
3. **Made the scores comparable.** Before this, each model computed its own metrics, and the
   two baselines used a lenient entity rule (an `I-` tag without a `B-` started a new
   entity) while the other four used strict IOB2. The shared script scores all six with the
   same rule.
4. **Wrote the benchmark runner** (`benchmark/run_benchmark.py`). It scores all six models
   on val and test and writes the result tables (`summary.csv`, `per_type_f1.csv`,
   `error_types.csv`, `results.json`, `results.md`) and three charts (F1 by model, F1 per
   type, errors by type) to `benchmark/results/`. It maps each model's prediction file
   names, so no one had to rename their files.
5. **Tested the evaluation** on small hand-made tag sequences where the right answer is known
   (precision, recall, token accuracy and each error type), and checked that missing,
   wrong-length and invalid predictions are rejected.
6. **Wrote the main README:** the task and its two difficulties, the data, the models
   (architecture and specific model listed separately), the information sources each model
   uses, the metric choice, the results with their interpretation, the limitations and how to
   run everything.
7. **Repository:** reviewed and merged the team's pull requests, added the Apache-2.0
   `LICENSE`, added the missing dependencies to `requirements.txt`, and removed
   `baselines/entity_evaluation.py` and `baselines/compare_models.py`, which the shared
   benchmark replaces.

I used Claude, an AI assistant, to help write and check the code and the documentation.

## Where my work is

- `benchmark/evaluate.py`, `benchmark/run_benchmark.py`
- `README.md`, `LICENSE`, `requirements.txt`
- This file.
