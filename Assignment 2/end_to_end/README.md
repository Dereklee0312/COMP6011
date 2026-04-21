# Transcript Suicide-Risk PoC

Single-file academic proof-of-concept for transcript-based suicide-risk classification:

`transcript -> prompt -> local Ollama model -> risk label`

This is not a clinical or production tool.

## Setup

```bash
uv sync
```

## Run

Edit the configuration block at the top of `run_eval.py` to set:

- `DATA_PATH` to your real workbook path
- `SHEET_NAME` if your worksheet name changes
- `MODEL_NAME` to your local Ollama model
- `MODE` to `"zeroshot"` or `"fewshot"`

Then run:

```bash
uv run python run_eval.py
```

## Expected Workbook

- Workbook file: your `.xlsx` file, for example `Student Assignment 10 Cases.xlsx`
- Worksheet: `Assignment_Cases`
- Required columns:
  - `Case ID`
  - `Paraphrased Dialogue`
  - `Risk Level`

## Outputs

The script writes results to the hardcoded `OUTPUT_DIR`:

- `predictions.csv`
- `metrics.json`
- `raw_outputs.jsonl`
- `confusion_matrix.csv`

## Where To Change Things Later

- Swap in your real workbook by editing `DATA_PATH` in `run_eval.py`
- Change the sheet name by editing `SHEET_NAME`
- Change the prompt, label definitions, or few-shot examples inside `run_eval.py`
- Switch between zero-shot and few-shot by editing `MODE`
