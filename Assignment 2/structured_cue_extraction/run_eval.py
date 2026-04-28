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

MODEL_NAME = "gemma4:latest"
BASE_URL = "http://192.168.0.48:11434/v1"
API_KEY = "ollama"
OUTPUT_DIR = Path("outputs")

TEMPERATURE = 0.0
SEED = 7

LABELS = ["attempt", "behavior", "ideation", "indicator", "safe"]
EXPLICIT_RISK_LABELS = ["ideation", "behavior", "attempt"]

CUE_DEFINITIONS = {
    "explicit_suicidal_expression": (
        "True if the dialogue explicitly mentions suicide, killing oneself, ending one's life, "
        "or a direct suicidal statement."
    ),
    "wish_to_die_or_not_exist": (
        "True if the speaker explicitly says they want to die, wish they would not wake up, "
        "want life to end, or do not want to exist."
    ),
    "suicide_method_mentioned": (
        "True if the speaker mentions a possible suicide method or means, such as pills, hanging, "
        "jumping, cutting, drowning, firearms, poison, ropes, belts, or similar."
    ),
    "specific_plan": (
        "True if the speaker describes a concrete plan for suicide, including what they intend to do "
        "or how they intend to do it."
    ),
    "timing_or_location": (
        "True if the speaker identifies a time, date, deadline, place, or location for suicide."
    ),
    "preparation_or_rehearsal": (
        "True if the speaker describes preparing, rehearsing, collecting means, writing goodbye notes, "
        "testing a method, arranging affairs, or similar pre-attempt behaviour."
    ),
    "self_harm_behavior": (
        "True if the speaker describes deliberate self-harm behavior, even if suicidal intent is unclear."
    ),
    "attempt_started_or_occurred": (
        "True if the speaker describes a current, recent, interrupted, or already-started suicide attempt."
    ),
    "past_suicide_attempt": (
        "True if the speaker explicitly mentions a previous suicide attempt, even if it was not recent."
    ),
    "indirect_warning_signs": (
        "List of indirect risk markers such as hopelessness, burdensomeness, isolation, feeling trapped, "
        "severe emotional pain, numbness, despair, or loss of meaning, without explicit suicidal expression."
    ),
    "protective_or_future_oriented_signs": (
        "List of protective or safety signs such as wanting help, future plans, social support, reasons for living, "
        "problem-solving, or explicit denial of suicidal thoughts."
    ),
    "evidence": (
        "Short transcript-grounded evidence snippets or paraphrases supporting the extracted cues."
    ),
}

RULE_PRIORITY = ["attempt", "behavior", "ideation", "indicator", "safe"]

MAPPING_RULES = {
    "attempt": "attempt_started_or_occurred OR past_suicide_attempt",
    "behavior": (
        "preparation_or_rehearsal OR specific_plan OR timing_or_location "
        "OR suicide_method_mentioned OR self_harm_behavior"
    ),
    "ideation": "explicit_suicidal_expression OR wish_to_die_or_not_exist",
    "indicator": "indirect_warning_signs is non-empty",
    "safe": "no suicide-risk cues are present",
}

BOOLEAN_CUE_FIELDS = [
    "explicit_suicidal_expression",
    "wish_to_die_or_not_exist",
    "suicide_method_mentioned",
    "specific_plan",
    "timing_or_location",
    "preparation_or_rehearsal",
    "self_harm_behavior",
    "attempt_started_or_occurred",
    "past_suicide_attempt",
]

LIST_CUE_FIELDS = [
    "indirect_warning_signs",
    "protective_or_future_oriented_signs",
    "evidence",
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
        "You are extracting structured suicide-risk cues from paraphrased conversational dialogues "
        "for a research proof-of-concept. You do not make clinical decisions and you do not provide "
        "treatment advice."
    )


def _cue_definitions_text():
    lines = ["Cue field definitions (set each field according to the dialogue):"]
    for name, desc in CUE_DEFINITIONS.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def build_user_message(dialogue):
    schema_obj = {
        "explicit_suicidal_expression": True,
        "wish_to_die_or_not_exist": True,
        "suicide_method_mentioned": True,
        "specific_plan": True,
        "timing_or_location": True,
        "preparation_or_rehearsal": True,
        "self_harm_behavior": True,
        "attempt_started_or_occurred": True,
        "past_suicide_attempt": True,
        "indirect_warning_signs": ["short marker strings"],
        "protective_or_future_oriented_signs": ["short marker strings"],
        "evidence": ["short quote or paraphrase from dialogue"],
    }
    parts = [
        "Task: Extract the structured suicide-risk cue fields from the dialogue below. "
        "Do not assign a final risk label. Do not provide treatment advice.",
        "",
        "Requirements:",
        "- Use only the provided dialogue. Do not invent facts.",
        "- Do not infer a suicide method, plan, or attempt unless it is clearly supported by the dialogue.",
        "- Do not provide treatment advice.",
        "- Do not assign the final label.",
        "- Return JSON only (one object), with no other text before or after.",
        "- Evidence must be short and grounded in the dialogue.",
        "",
        _cue_definitions_text(),
        "",
        "Output schema (booleans true/false; list fields are arrays of short strings):",
        json.dumps(schema_obj, indent=2),
        "",
        "Paraphrased dialogue to analyse:",
        dialogue,
    ]
    return "\n".join(parts)


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
    cues = {key: bool(data.get(key, False)) for key in BOOLEAN_CUE_FIELDS}
    for key in LIST_CUE_FIELDS:
        cues[key] = list(data.get(key, []))
    return cues


def map_cues_to_label(cues):
    if cues.get("attempt_started_or_occurred") or cues.get("past_suicide_attempt"):
        return "attempt"

    if (
        cues.get("preparation_or_rehearsal")
        or cues.get("specific_plan")
        or cues.get("timing_or_location")
        or cues.get("suicide_method_mentioned")
        or cues.get("self_harm_behavior")
    ):
        return "behavior"

    if cues.get("explicit_suicidal_expression") or cues.get("wish_to_die_or_not_exist"):
        return "ideation"

    if cues.get("indirect_warning_signs"):
        return "indicator"

    return "safe"


def build_rationale(label, cues):
    evidence = cues.get("evidence", [])
    evidence_text = (
        "; ".join(evidence[:2]) if evidence else "No specific evidence returned."
    )
    return f"Mapped to {label} using structured cue rules. Evidence: {evidence_text}"


def safe_divide(num, den):
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

    parse_error_count = sum(1 for row in rows if row["parse_error"])

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


def save_outputs(prediction_rows, raw_rows, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

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
        "model_name": MODEL_NAME,
        "base_url": BASE_URL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "labels": LABELS,
        "explicit_risk_labels": EXPLICIT_RISK_LABELS,
        "cue_definitions": CUE_DEFINITIONS,
        "mapping_rules": MAPPING_RULES,
        "rule_priority": RULE_PRIORITY,
        "boolean_cue_fields": BOOLEAN_CUE_FIELDS,
        "list_cue_fields": LIST_CUE_FIELDS,
        "prompt_design_summary": (
            "Approach 2 asks the LLM to extract structured suicide-risk cues only. "
            "Final label assignment is performed by deterministic Python rules using a "
            "highest-risk-supported priority order."
        ),
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
        cues = {k: False for k in BOOLEAN_CUE_FIELDS} | {k: [] for k in LIST_CUE_FIELDS}
        predicted_label = ""
        rationale = ""

        try:
            cues = parse_model_output(raw_text)
            predicted_label = map_cues_to_label(cues)
            rationale = build_rationale(predicted_label, cues)
        except Exception:
            parse_error = True

        pred_row = {
            "case_id": item["case_id"],
            "ground_truth": item["ground_truth"],
            "predicted_label": predicted_label,
            "correct": int(item["ground_truth"] == predicted_label),
            "rationale": rationale,
            "parse_error": parse_error,
        }
        for b in BOOLEAN_CUE_FIELDS:
            pred_row[b] = cues[b]
        for k in LIST_CUE_FIELDS:
            pred_row[k] = json.dumps(cues[k], ensure_ascii=False)
        prediction_rows.append(pred_row)

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
