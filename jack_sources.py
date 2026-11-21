"""JACK: provenance-aware evidence intake for a standalone analytic model.

No pretrained backbone, gradient training, remote embeddings or telemetry.
All personal source bytes remain in an operator-chosen private vault.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

SCHEMA = "JACK_EVIDENCE_INDEX_V1"
SOURCE_WEIGHT = {"conversation": 1.0, "history": 0.45, "public": 0.65}
PROVENANCE_FACTOR = {"user_role_in_export": 1.0,
                     "visual_role_marker_in_export": 0.65,
                     "self_report": 0.85, "third_party_record": 0.45,
                     "verified_public_account": 0.75}
TOKEN = re.compile(r"[A-Za-z0-9']+")
SUPPORTED = {"chatgpt_export", "pdf_text_speaker_export", "history_docx",
             "history_text", "voice_text", "public_text"}


def tokens(text):
    return TOKEN.findall(text.lower())


def chatgpt_messages(data):
    """Read user-role text only; role tags are evidence, not identity proof."""
    conversations = data if isinstance(data, list) else data.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError("expected conversations list")
    for conversation in conversations:
        if not isinstance(conversation, dict):
            continue
        messages = []
        mapping = conversation.get("mapping", {})
        if isinstance(mapping, dict) and mapping:
            current = conversation.get("current_node")
            if current is not None:
                # Use the selected ancestry, not every discarded conversation fork.
                lineage, seen = [], set()
                while current is not None:
                    if not isinstance(current, str) or current in seen:
                        lineage = []
                        break
                    seen.add(current)
                    node = mapping.get(current)
                    if not isinstance(node, dict):
                        lineage = []
                        break
                    lineage.append(node.get("message"))
                    current = node.get("parent")
                messages = list(reversed(lineage))
            else:
                messages = [v.get("message") for v in mapping.values()
                            if isinstance(v, dict)]
        elif isinstance(conversation.get("messages"), list):
            messages = conversation["messages"]
        for message in sorted((x for x in messages if isinstance(x, dict)),
                              key=lambda x: (x.get("create_time") or 0)):
            author = message.get("author", {})
            role = author.get("role") if isinstance(author, dict) else message.get("role")
            if role != "user":
                continue
            payload = message.get("content", {})
            parts = payload.get("parts", []) if isinstance(payload, dict) else []
            if not parts and isinstance(payload, str):
                parts = [payload]
            if not isinstance(parts, list):
                continue
            text = "\n".join(p.strip() for p in parts
                             if isinstance(p, str) and p.strip())
            if text.strip():
                yield text.strip()


def pdf_text_user_messages(raw):
    """Conservative user-only extraction from a timestamp-stamped PDF text export.

    This is weaker attribution than native ChatGPT JSON: visual role labels
    can be damaged by PDF extraction. Assistant blocks are never returned.
    """
    marker = re.compile(r"^\s*(You|ChatGPT)\s+\S\s+Nov\s+\d{1,2},\s+\d{4},", re.I)
    role, lines = None, []
    for line in raw.splitlines():
        match = marker.search(line)
        if match:
            if role == "you" and lines:
                yield "\n".join(lines).strip()
            role = match.group(1).lower()
            lines = []
            continue
        if role != "you":
            continue
        clean = line.strip()
        if (not clean or clean.startswith("=== PAGE") or
                clean == "Vera" or "moz-extension://" in clean or
                "This chat was exported with" in clean or
                clean.startswith("Export any chat to") or
                re.fullmatch(r"\d+ of \d+", clean)):
            continue
        lines.append(clean)
    if role == "you" and lines:
        yield "\n".join(lines).strip()


def docx_text(path):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("word/document.xml")
        if info.file_size > 12_000_000:
            raise ValueError("DOCX document.xml too large")
        root = ET.fromstring(archive.read(info))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for para in root.findall(".//w:p", ns):
        line = "".join(el.text or "" for el in para.findall(".//w:t", ns))
        if line.strip():
            paragraphs.append(line.strip())
    return "\n".join(paragraphs)


def chunks(text, limit=650):
    """Split at paragraph boundaries, without treating document text as commands."""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        while para:
            if len(para) <= limit:
                yield para
                break
            cut = para.rfind(" ", 200, limit + 1)
            cut = cut if cut > 0 else limit
            yield para[:cut].strip()
            para = para[cut:].strip()


def _vault(manifest_path, output_path):
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    vault = manifest_path.parent
    if not output_path.is_relative_to(vault) or output_path == manifest_path:
        raise ValueError("output must be within the private manifest vault")
    if any((p / ".git").exists() for p in (vault, *vault.parents)):
        raise ValueError("private vault may not be inside a Git repository")
    if output_path.exists():
        raise FileExistsError("refusing to overwrite an existing private index")
    return manifest_path, output_path


def build_index(manifest_path, output_path):
    manifest_path, output_path = _vault(manifest_path, output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "JACK_SOURCE_MANIFEST_V1":
        raise ValueError("manifest schema mismatch")
    records, receipts, seen_ids = [], [], set()
    for source in manifest.get("sources", []):
        sid = source["id"]
        kind = source["kind"]
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,90}", sid):
            raise ValueError("invalid source id")
        if sid in seen_ids or kind not in SUPPORTED:
            raise ValueError("duplicate source or unsupported type")
        seen_ids.add(sid)
        path = Path(source["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 50_000_000:
            raise ValueError("missing, linked or oversize source")
        if source.get("sensitivity") not in ("private", "restricted", "public"):
            raise ValueError("source sensitivity missing")
        proven = source.get("provenance")
        if proven not in PROVENANCE_FACTOR:
            raise ValueError("unverified source provenance")
        if kind == "chatgpt_export" and proven != "user_role_in_export":
            raise ValueError("chat exports require user-role attribution")
        if (kind == "pdf_text_speaker_export" and
                proven != "visual_role_marker_in_export"):
            raise ValueError("PDF speaker exports require visual marker provenance")
        if kind == "public_text" and proven != "verified_public_account":
            raise ValueError("public source account requires verified attribution")
        raw = path.read_bytes()
        if kind == "chatgpt_export":
            texts = list(chatgpt_messages(json.loads(raw.decode("utf-8-sig"))))
            group = "conversation"
        elif kind == "pdf_text_speaker_export":
            texts = list(pdf_text_user_messages(raw.decode("utf-8-sig")))
            group = "conversation"
        elif kind == "history_docx":
            texts = list(chunks(docx_text(path)))
            group = "history"
        else:
            texts = list(chunks(raw.decode("utf-8-sig")))
            group = "conversation" if kind == "voice_text" else (
                "public" if kind == "public_text" else "history")
        count = 0
        for i, passage in enumerate(texts):
            if len(passage.strip()) < 8:
                continue
            records.append({"id": f"{sid}:{i}", "source_id": sid,
                            "class": group, "provenance": proven,
                            "sensitivity": source["sensitivity"],
                            "text": passage, "tokens": tokens(passage)})
            count += 1
        receipts.append({"source_id": sid, "sha256": hashlib.sha256(raw).hexdigest(),
                         "kind": kind, "provenance": proven, "records": count,
                         "sensitivity": source["sensitivity"]})
    voice = [r["text"] for r in records if r["class"] == "conversation"
             and r["provenance"] in ("user_role_in_export", "visual_role_marker_in_export", "self_report")]
    word_counts = [len(tokens(x)) for x in voice]
    style = {"messages": len(voice),
             "avg_words": round(sum(word_counts) / max(1, len(voice)), 2),
             "question_ratio": round(sum("?" in x for x in voice) / max(1, len(voice)), 4),
             "exclamation_ratio": round(sum("!" in x for x in voice) / max(1, len(voice)), 4)}
    df = Counter(term for rec in records for term in set(rec["tokens"]))
    n = max(1, len(records))
    lexical_weights = {word: round(math.log(1 + (n - count + 0.5) /
                              (count + 0.5)), 9) for word, count in sorted(df.items())}
    calculated_digest = hashlib.sha256(json.dumps(lexical_weights, sort_keys=True,
                              separators=(",", ":")).encode("utf-8")).hexdigest()
    index = {"schema": SCHEMA, "backbone": "none",
             "method": "provenance_filtered_source_index",
             "neural_weights_updated": False, "weights": SOURCE_WEIGHT,
             "provenance_factors": PROVENANCE_FACTOR,
             "lexical_weights": lexical_weights, "calculated_weights_sha256": calculated_digest,
             "style": style, "source_receipts": receipts, "records": records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, ensure_ascii=False, separators=(",", ":"))
    return index


def retrieve(index, query, limit=5):
    if index.get("schema") != SCHEMA:
        raise ValueError("invalid index")
    query_terms = set(tokens(query))
    records = index["records"]
    n = max(1, len(records))
    lexical_weights = index["lexical_weights"]
    mean_len = max(1.0, sum(len(rec["tokens"]) for rec in records) / n)
    matches = []
    for rec in records:
        tf = Counter(rec["tokens"])
        length = max(1, len(rec["tokens"]))
        bm25 = 0.0
        for term in query_terms & tf.keys():
            idf = lexical_weights.get(term, 0.0)
            bm25 += idf * tf[term] * 2.2 / (
                tf[term] + 1.2 * (0.25 + 0.75 * length / mean_len))
        score = (bm25 * index["weights"][rec["class"]] *
                 index.get("provenance_factors", PROVENANCE_FACTOR)[rec["provenance"]])
        if score > 0:
            matches.append({**rec, "score": round(score, 7)})
    matches.sort(key=lambda r: (-r["score"], r["id"]))
    return matches[:max(0, min(int(limit), 8))]



def main():
    import argparse
    parser=argparse.ArgumentParser(description="JACK private evidence intake; no neural training")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    result=build_index(args.manifest,args.output)
    print(json.dumps({"records":len(result["records"]),
                      "source_groups":len(result["source_receipts"]),
                      "style":result["style"],
                      "output_sha256":hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
                      "neural_weights_updated":False}))

if __name__ == "__main__":
    main()
