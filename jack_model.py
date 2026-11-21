"""JACK's own closed-form statistical language model; no pretrained neural backbone.

Builds sparse word transition probabilities directly from attributed Patrick text.
History informs topic weights but cannot silently become Patrick's spoken voice.
Only Python standard library is used. No API access, optimizer, or tensor loading.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

SCHEMA = "JACK_ANALYTIC_MODEL_V1"
SOURCE_SCHEMA = "JACK_EVIDENCE_INDEX_V1"
WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENTENCE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
VOICE_ROLES = {"user_role_in_export": 1.0,
               "visual_role_marker_in_export": 0.65,
               "self_report": 0.85}
CONTEXT_FACTORS = {"third_party_record": 0.2,
                   "verified_public_account": 0.35,
                   "self_report": 0.5}


def tokenize(text):
    return WORD.findall(text.lower())


def _normalize(counts):
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {word: round(counts[word] / total, 10)
            for word in sorted(counts)}


def _transitions(counts):
    return {key: _normalize(next_counts)
            for key, next_counts in sorted(counts.items())}


def _model_weights(model):
    return {k: model[k] for k in (
        "unigram_counts", "lexicon_weights", "transition_weights",
        "backoff_weights", "start_weights", "history_weights")}


def digest_weights(model):
    data = json.dumps(_model_weights(model), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_parameters(evidence):
    if evidence.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unrecognized evidence index")
    unigram = Counter()
    tri = defaultdict(Counter)
    bi = defaultdict(Counter)
    starts = Counter()
    history = Counter()
    voice_count = 0
    for rec in evidence["records"]:
        role = rec["provenance"]
        category = rec["class"]
        if category == "conversation" and role in VOICE_ROLES:
            voice_count += 1
            factor = VOICE_ROLES[role]
            for sentence in SENTENCE.split(rec["text"]):
                words = tokenize(sentence)
                if not words:
                    continue
                starts[words[0]] += factor
                unigram.update({w: factor * count for w, count in Counter(words).items()})
                state = ("<bos>", "<bos>")
                for token in words + ["<eos>"]:
                    tri["|".join(state)][token] += factor
                    bi[state[1]][token] += factor
                    state = (state[1], token)
        elif category in ("history", "public") and role in CONTEXT_FACTORS:
            factor = CONTEXT_FACTORS[role]
            # Context words contribute topic weight only; never transition counts.
            for word, count in Counter(tokenize(rec["text"])).items():
                history[word] += factor * count
    if not voice_count or not unigram:
        raise ValueError("cannot build Patrick-attributed voice from missing evidence")
    model = {
        "schema": SCHEMA,
        "architecture": "weighted_word_trigram",
        "backbone": "none",
        "optimizer_steps": 0,
        "neural_weights": "none",
        "speaker_records": voice_count,
        "source_receipts": evidence.get("source_receipts", []),
        "style": evidence.get("style", {}),
        "unigram_counts": dict(sorted(unigram.items())),
        "lexicon_weights": _normalize(unigram),
        "transition_weights": _transitions(tri),
        "backoff_weights": _transitions(bi),
        "start_weights": _normalize(starts),
        "history_weights": _normalize(history),
        "weight_equations": {
            "transition": "w(next|prev2,prev1)=C(prev2,prev1,next)/sum_next C(prev2,prev1,next)",
            "lexicon": "w(token)=C_Patrick(token)/sum_token C_Patrick(token)",
            "context": "w(topic)=sum_history provenance_factor*count(topic)/sum_all_topics",
            "provenance": "PDF-visual roles 0.65, native user role 1.0, self report 0.85; third-party history 0.2"
        }
    }
    model["weights_sha256"] = digest_weights(model)
    return model


def validate_model(model):
    if model.get("schema") != SCHEMA or model.get("backbone") != "none":
        raise ValueError("not a JACK analytic model")
    if model.get("neural_weights") != "none" or model.get("optimizer_steps") != 0:
        raise ValueError("this release forbids neural training")
    if digest_weights(model) != model.get("weights_sha256"):
        raise ValueError("weight integrity mismatch")


def save_model(model, output, private_vault):
    validate_model(model)
    root = Path(private_vault).resolve()
    path = Path(output).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("model output outside private vault")
    if any((d / ".git").exists() for d in (root, *root.parents)):
        raise ValueError("private model vault cannot be inside Git")
    if path.exists():
        raise FileExistsError("model artifact already exists")
    root.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(model, file, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_model(model)
    return model


def _pick(rng, weights):
    items = sorted(weights)
    if not items:
        return "<eos>"
    return rng.choices(items, weights=[weights[w] for w in items], k=1)[0]


def generate(model, question, seed=0, max_tokens=45):
    validate_model(model)
    if not isinstance(question, str) or not question.strip():
        return ""
    rng = random.Random(seed)
    query = set(tokenize(question))
    # From the model's own first-word distribution, bias starts toward question
    # terms and historically recurrent topics, never insert history-only tokens.
    starts = {}
    history = model["history_weights"]
    for term, weight in model["start_weights"].items():
        topic_bonus = 1 + (0.5 if term in query else 0) + (
            0.3 * history.get(term, 0) if term in query else 0)
        starts[term] = weight * topic_bonus
    if not starts:
        return ""
    state = ("<bos>", "<bos>")
    output = []
    sentence_starts = 0
    for position in range(max(1, min(int(max_tokens), 160))):
        key = "|".join(state)
        choices = model["transition_weights"].get(key)
        if position == 0:
            word = _pick(rng, starts)
        else:
            choices = choices or model["backoff_weights"].get(state[-1])
            if not choices:
                choices = model["lexicon_weights"]
            word = _pick(rng, choices)
        if word == "<eos>":
            if output:
                break
            state = ("<bos>", "<bos>")
            sentence_starts += 1
            if sentence_starts > 2:
                break
            continue
        output.append(word)
        state = (state[-1], word)
    return " ".join(output).strip().capitalize() + ("." if output else "")


def main():
    parser = argparse.ArgumentParser(
        description="JACK's own source-calculated language model; no neural training")
    subs = parser.add_subparsers(dest="cmd", required=True)
    build = subs.add_parser("build")
    build.add_argument("--evidence", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--vault", required=True)
    sample = subs.add_parser("sample")
    sample.add_argument("--model", required=True)
    sample.add_argument("--question", required=True)
    sample.add_argument("--seed", type=int, default=1)
    sample.add_argument("--max-tokens", type=int, default=45)
    inspect = subs.add_parser("inspect")
    inspect.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
        result = build_parameters(evidence)
        sha = save_model(result, args.output, args.vault)
        print(json.dumps({"artifact_sha256": sha,
                          "weights_sha256": result["weights_sha256"],
                          "vocabulary": len(result["unigram_counts"]),
                          "transition_contexts": len(result["transition_weights"]),
                          "history_topics": len(result["history_weights"]),
                          "speaker_records": result["speaker_records"],
                          "backbone": "none", "neural_weights": "none"}))
    elif args.cmd == "inspect":
        model = load_model(args.model)
        print(json.dumps({"schema": model["schema"],
                          "weights_sha256": model["weights_sha256"],
                          "vocabulary": len(model["unigram_counts"]),
                          "history_topics": len(model["history_weights"]),
                          "backbone": model["backbone"],
                          "neural_weights": model["neural_weights"]}))
    else:
        model = load_model(args.model)
        print(generate(model, args.question, args.seed, args.max_tokens))


if __name__ == "__main__":
    main()
