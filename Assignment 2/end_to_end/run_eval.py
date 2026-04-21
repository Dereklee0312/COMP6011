import json
from pathlib import Path

import pandas as pd
from openai import BadRequestError, OpenAI
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


DATA_PATH = Path("Student Assignment 10 Cases.xlsx")
SHEET_NAME = "Assignment_Cases"
ID_COL = "Case ID"
TRANSCRIPT_COL = "Paraphrased Dialogue"
LABEL_COL = "Risk Level"

MODE = "fewshot"  # "zeroshot" or "fewshot"
MODEL_NAME = "llama3.1:8b"
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
OUTPUT_DIR = Path("outputs")

TEMPERATURE = 0.0
SEED = 7

LABELS = ["attempt", "behavior", "ideation", "indicator", "safe"]

LABEL_DEFINITIONS = {
    "attempt": "the dialogue indicates a suicide attempt or a clearly described past/current attempt.",
    "ideation": "the dialogue indicates suicidal thoughts, desire to die, or wanting life to end, but no clear attempt.",
    "behavior": "the dialogue indicates suicide-related or self-harm-related behavior, gestures, planning, rehearsal, or preparatory acts without a confirmed attempt.",
    "indicator": "the dialogue contains indirect warning signs or concerning markers of suicide risk, but does not clearly reach ideation, behavior, or attempt.",
    "safe": "no clear evidence of suicide risk in the dialogue.",
}

FEWSHOT_EXAMPLES = [
    {
        "dialogue": "Speaker says they took a large number of pills last night because they wanted to die and then woke up in hospital.",
        "label": "attempt",
        "rationale": "The dialogue describes a suicide attempt by overdose.",
    },
    {
        "dialogue": "Speaker says they have been collecting pills and writing goodbye notes but deny actually trying to die.",
        "label": "behavior",
        "rationale": "The dialogue shows preparatory suicide-related behavior without a confirmed attempt.",
    },
    {
        "dialogue": "Speaker says they keep thinking everyone would be better off if they were dead and wish they would not wake up.",
        "label": "ideation",
        "rationale": "The dialogue expresses suicidal thoughts and a desire to die without a described attempt.",
    },
    {
        "dialogue": "Speaker says they feel trapped, hopeless, and like life is pointless, but they do not mention wanting to die or making plans.",
        "label": "indicator",
        "rationale": "The dialogue contains indirect warning signs without clear ideation, behavior, or attempt.",
    },
    {
        "dialogue": "Speaker discusses exam stress and poor sleep but also says they want support to manage workload and make a study plan.",
        "label": "safe",
        "rationale": "The dialogue does not provide clear evidence of suicide risk.",
    },
]


def normalize_label(value):
    text = str(value).strip().lower()
    if text not in LABELS:
        raise ValueError(f"Unsupported label: {value!r}. Expected one of {LABELS}.")
    return text


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name=SHEET_NAME)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Only .xlsx and .csv files are supported.")

    required = [ID_COL, TRANSCRIPT_COL, LABEL_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "case_id": str(row[ID_COL]).strip(),
                "dialogue": str(row[TRANSCRIPT_COL]).strip(),
                "ground_truth": normalize_label(row[LABEL_COL]),
            }
        )
    return records


def build_system_message():
    return (
        "You are performing a research classification task on paraphrased conversational "
        "dialogues for suicide-risk screening evaluation."
    )


def build_user_message(dialogue):
    lines = [
        "Classify the provided dialogue into exactly one suicide-risk label.",
        "Use only the content of the provided dialogue.",
        "Do not invent facts beyond the transcript.",
        "Do not provide treatment advice.",
        "",
        "Labels:",
    ]
    for label in LABELS:
        lines.append(f"- {label}: {LABEL_DEFINITIONS[label]}")

    if MODE == "fewshot":
        lines.extend(["", "Examples:"])
        for example in FEWSHOT_EXAMPLES:
            lines.append(f"Dialogue: {example['dialogue']}")
            lines.append(
                "JSON: "
                + json.dumps(
                    {"label": example["label"], "rationale": example["rationale"]},
                    ensure_ascii=True,
                )
            )
            lines.append("")

    lines.extend(
        [
            "Return JSON only using this schema:",
            '{"label":"attempt|behavior|ideation|indicator|safe","rationale":"short explanation"}',
            "Keep the rationale short and at most 2 sentences.",
            "",
            "Dialogue to classify:",
            dialogue,
        ]
    )
    return "\n".join(lines)


def call_model(client, dialogue):
    messages = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": build_user_message(dialogue)},
    ]
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
    }
    try:
        response = client.chat.completions.create(seed=SEED, **kwargs)
    except BadRequestError:
        response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def parse_model_output(raw_text):
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")

    data = json.loads(text[start : end + 1])
    label = normalize_label(data["label"])
    rationale = str(data.get("rationale", "")).strip()
    return {"label": label, "rationale": rationale}


def compute_metrics(rows):
    y_true = [row["ground_truth"] for row in rows]
    y_pred = [row["predicted_label"] if row["predicted_label"] in LABELS else "__parse_error__" for row in rows]

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for label in LABELS:
        y_true_bin = [1 if value == label else 0 for value in y_true]
        y_pred_bin = [1 if value == label else 0 for value in y_pred]
        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        per_class[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS + ["__parse_error__"],
    )
    confusion = {}
    for row_index, truth_label in enumerate(LABELS):
        confusion[truth_label] = {}
        for col_index, pred_label in enumerate(LABELS):
            confusion[truth_label][pred_label] = int(matrix[row_index][col_index])

    return {
        "total_cases": len(rows),
        "parse_error_count": sum(row["parse_error"] for row in rows),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(sum(precisions) / len(LABELS), 4),
        "macro_recall": round(sum(recalls) / len(LABELS), 4),
        "macro_f1": round(sum(f1s) / len(LABELS), 4),
        "per_class": per_class,
        "labels": LABELS,
        "confusion_matrix": confusion,
        "parse_error_by_true_label": {
            LABELS[row_index]: int(matrix[row_index][-1]) for row_index in range(len(LABELS))
        },
    }


def save_outputs(rows, raw_rows, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (OUTPUT_DIR / "raw_outputs.jsonl").open("w", encoding="utf-8") as f:
        for item in raw_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    confusion_rows = []
    for truth_label, preds in metrics["confusion_matrix"].items():
        row = {"ground_truth": truth_label}
        row.update(preds)
        confusion_rows.append(row)
    pd.DataFrame(confusion_rows).to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=False)


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    dataset = load_dataset(DATA_PATH)

    prediction_rows = []
    raw_output_rows = []

    for item in dataset:
        raw_text = call_model(client, item["dialogue"])
        parse_error = False
        predicted_label = ""
        rationale = ""

        try:
            parsed = parse_model_output(raw_text)
            predicted_label = parsed["label"]
            rationale = parsed["rationale"]
        except Exception:
            parse_error = True

        prediction_rows.append(
            {
                "case_id": item["case_id"],
                "ground_truth": item["ground_truth"],
                "predicted_label": predicted_label,
                "correct": int(item["ground_truth"] == predicted_label),
                "rationale": rationale,
                "parse_error": parse_error,
            }
        )

        raw_output_rows.append(
            {
                "case_id": item["case_id"],
                "ground_truth": item["ground_truth"],
                "raw_output": raw_text,
                "parse_error": parse_error,
            }
        )

    metrics = compute_metrics(prediction_rows)
    save_outputs(prediction_rows, raw_output_rows, metrics)

    print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
