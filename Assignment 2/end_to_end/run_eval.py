import json
from pathlib import Path

import pandas as pd
from openai import BadRequestError, OpenAI
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DATA_PATH = Path("Student Assignment 10 Cases.xlsx")
SHEET_NAME = "Assignment_Cases"
ID_COL = "Case ID"
TRANSCRIPT_COL = "Paraphrased Dialogue"
LABEL_COL = "Risk Level"

MODE = "fewshot"  # "zeroshot" or "fewshot"
MODEL_NAME = "gemma4:latest"
BASE_URL = "http://192.168.0.48:11434/v1"
API_KEY = "ollama"
OUTPUT_DIR = Path("outputs")

TEMPERATURE = 0.0
SEED = 7

LABELS = ["attempt", "behavior", "ideation", "indicator", "safe"]
EXPLICIT_RISK_LABELS = ["ideation", "behavior", "attempt"]

LABEL_DEFINITIONS = {
    "attempt": (
        "the dialogue describes a past, current, interrupted, or already-started suicide attempt, "
        "where the speaker has taken or begun a potentially lethal self-harm action with intent to die."
    ),
    "behavior": (
        "the dialogue describes suicide-related behavior, preparation, rehearsal, method selection, "
        "timing, location, tools, or self-harm behavior, but no confirmed suicide attempt has already started."
    ),
    "ideation": (
        "the dialogue contains explicit suicidal thoughts, wanting to die, wanting life to end, "
        "or imagining suicide, but no specific plan, preparation, rehearsal, self-harm act, or attempt."
    ),
    "indicator": (
        "the dialogue contains indirect warning signs such as hopelessness, burdensomeness, isolation, "
        "feeling trapped, emotional numbness, or severe distress, but no explicit wish to die, suicide plan, "
        "suicide behavior, or attempt."
    ),
    "safe": (
        "the dialogue contains no clear evidence of suicide risk, and may include ordinary stress, "
        "positive future plans, help-seeking, or neutral conversation."
    ),
}

FEWSHOT_EXAMPLES = [
    {
        "dialogue": (
            "Speaker says they swallowed a large amount of medication last night because they wanted to die, "
            "then woke up after someone found them."
        ),
        "label": "attempt",
        "rationale": "The dialogue describes a suicide attempt by overdose with intent to die.",
    },
    {
        "dialogue": (
            "Speaker says they put a belt around their neck and tightened it, then stopped only because "
            "someone came into the room."
        ),
        "label": "attempt",
        "rationale": "The dialogue describes an interrupted suicide attempt that had already started.",
    },
    {
        "dialogue": (
            "Speaker says they have chosen a bridge, picked a date, and written goodbye messages, "
            "but they have not gone there or harmed themselves yet."
        ),
        "label": "behavior",
        "rationale": "The dialogue shows planning and preparation without a confirmed attempt.",
    },
    {
        "dialogue": (
            "Speaker says they keep rehearsing how they would use a rope and have been testing whether "
            "the place they chose would hold their weight."
        ),
        "label": "behavior",
        "rationale": "The dialogue shows suicide-related rehearsal and method preparation without a confirmed attempt.",
    },
    {
        "dialogue": (
            "Speaker says they wish they would not wake up and believe everyone would be better off "
            "if they were dead, but they do not mention a method, plan, or action."
        ),
        "label": "ideation",
        "rationale": "The dialogue expresses suicidal thoughts without preparation, behavior, or attempt.",
    },
    {
        "dialogue": (
            "Speaker says they feel hopeless, trapped, exhausted, and like a burden, but they do not say "
            "they want to die or describe any suicide-related action."
        ),
        "label": "indicator",
        "rationale": "The dialogue contains indirect warning signs without explicit suicidal ideation or behavior.",
    },
    {
        "dialogue": (
            "Speaker talks about being stressed by deadlines but also mentions weekend plans, support from friends, "
            "and wanting help to manage their workload."
        ),
        "label": "safe",
        "rationale": "The dialogue describes ordinary stress and help-seeking without suicide-risk evidence.",
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

    lines.extend(
        [
            "",
            "Decision rules:",
            "- Use the highest-risk label supported by the dialogue.",
            "- Attempt requires evidence that a suicide attempt has already started, occurred, or was interrupted.",
            "- Behavior includes suicide-related planning, preparation, rehearsal, selected method, timing, location, or tools.",
            "- Ideation includes explicit desire to die or suicidal thoughts without plan, preparation, behavior, or attempt.",
            "- Indicator includes indirect warning signs only, without explicit suicidal desire or suicide-related behavior.",
            "- Safe means no clear suicide-risk evidence.",
            "- If the dialogue is ambiguous, do not infer facts that are not stated.",
        ]
    )

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


def safe_divide(num, den):
    # Prevent div by zero
    return round(float(num / den), 4) if den else 0.0


def compute_explicit_risk_recall(rows):
    true_explicit_risk = 0
    correctly_predicted_explicit_risk = 0

    for row in rows:
        true_label = row["ground_truth"]
        pred_label = row["predicted_label"]

        if true_label in EXPLICIT_RISK_LABELS:
            true_explicit_risk += 1
            if pred_label in EXPLICIT_RISK_LABELS:
                correctly_predicted_explicit_risk += 1

    return safe_divide(correctly_predicted_explicit_risk, true_explicit_risk)


def compute_metrics(rows):
    y_true = [row["ground_truth"] for row in rows]
    y_pred = [
        row["predicted_label"]
        if row["predicted_label"] in LABELS
        else "__parse_error__"
        for row in rows
    ]

    per_class = {}
    f1s = []

    for label in LABELS:
        y_true_bin = [1 if value == label else 0 for value in y_true]
        y_pred_bin = [1 if value == label else 0 for value in y_pred]

        precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        support = sum(y_true_bin)

        per_class[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(support),
        }

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

    parse_error_count = sum(row["parse_error"] for row in rows)

    return {
        "total_cases": len(rows),
        "parse_error_count": int(parse_error_count),
        "parse_error_rate": safe_divide(parse_error_count, len(rows)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(sum(f1s) / len(LABELS), 4),
        "explicit_risk_labels": EXPLICIT_RISK_LABELS,
        "explicit_risk_recall": compute_explicit_risk_recall(rows),
        "per_class": per_class,
        "labels": LABELS,
        "confusion_matrix": confusion,
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
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv", index=False
    )

    run_config = {
        "data_path": str(DATA_PATH),
        "sheet_name": SHEET_NAME,
        "id_col": ID_COL,
        "transcript_col": TRANSCRIPT_COL,
        "label_col": LABEL_COL,
        "mode": MODE,
        "model_name": MODEL_NAME,
        "base_url": BASE_URL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "labels": LABELS,
        "explicit_risk_labels": EXPLICIT_RISK_LABELS,
        "label_definitions": LABEL_DEFINITIONS,
    }

    with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)


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
