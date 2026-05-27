# PSA Sequence Modeling

Deep learning pipeline for modeling PSA (Prostate-Specific Antigen) trajectories using an attention-based LSTM with Time2Vec embeddings. The pipeline systematically evaluates multiple sequence lengths (**N**) and PSA **cutoff** thresholds to assess model performance across different configurations.

## Tasks

- **next_psa** — predict whether the next PSA crosses a cutoff
- **multi** — predict if/when PSA crosses (never / within 12 months / after 12 months)
- **cancer** — predict prostate cancer occurrence

## Install

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib tqdm openpyxl
```

## Input

Place `data.xlsx` (or `.xls` / `.csv`) in the repo root with these columns:

- `patient_unique_key`
- `test_time`
- `test_value`
- `cancer_ind`

## Usage

```bash
python psa_sequence_modeling.py
```

## Output

Results are saved to a timestamped folder containing trained models, metrics (CSV/XLSX), plots, and raw plot data for later regeneration.
