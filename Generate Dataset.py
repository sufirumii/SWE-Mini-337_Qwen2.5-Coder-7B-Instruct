#!/usr/bin/env python3
"""
SWE-Mini Dataset Generator
=================================
Model   : Qwen/Qwen2.5-Coder-7B-Instruct (4-bit NF4)
Hardware: Kaggle 2x T4 (32GB VRAM) — device_map=auto
Output  : /kaggle/working/swe_decompose_dataset.jsonl + .parquet
Runs indefinitely. Saves every valid, syntax-checked, non-duplicate sample.
"""

import subprocess, sys, os, warnings

# Suppress all HuggingFace and transformers warnings before any imports
os.environ["HF_HUB_DISABLE_WARNINGS"]          = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"
os.environ["HF_HUB_VERBOSITY"]                 = "error"
warnings.filterwarnings("ignore")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "bitsandbytes>=0.43.0", "accelerate", "pandas", "pyarrow"])

import ast
import json
import random
import time
import uuid
import difflib
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()
import pandas as pd

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID              = "Qwen/Qwen2.5-Coder-7B-Instruct"
SAVE_DIR              = Path("/kaggle/working")
JSONL_PATH            = SAVE_DIR / "swe_decompose_dataset.jsonl"
PARQUET_PATH          = SAVE_DIR / "swe_decompose_dataset.parquet"
LOG_PATH              = SAVE_DIR / "swe_decompose_log.txt"
MAX_NEW_TOKENS        = 4096
TEMPERATURE           = 0.4
TOP_P                 = 0.95
TOP_K                 = 20
PARQUET_EVERY         = 20
MIN_IMPL_LENGTH       = 300   # reject sample if implementation shorter than this
DUPLICATE_THRESHOLD   = 0.85  # reject sample if task_title is this similar to an existing one
# ──────────────────────────────────────────────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ─── RESTORE BACKUP IF NEEDED ─────────────────────────────────────────────────
BACKUP_PATH = Path("/kaggle/input/datasets/rumiisufi/datasett/swe_decompose_dataset.jsonl")

if not JSONL_PATH.exists() and BACKUP_PATH.exists():
    import shutil
    shutil.copy(BACKUP_PATH, JSONL_PATH)
    print(f"Restored backup: {BACKUP_PATH} -> {JSONL_PATH}")
elif JSONL_PATH.exists():
    print(f"Existing file found at {JSONL_PATH} — continuing from there")
elif not BACKUP_PATH.exists():
    print("No backup found and no existing file — starting fresh")

DOMAINS = [
    "networking and HTTP clients",
    "data structures and algorithms",
    "file system and I/O operations",
    "concurrency and async programming",
    "database and ORM patterns",
    "caching systems",
    "authentication and security",
    "message queues and event systems",
    "CLI tools and argument parsing",
    "serialization and data formats",
    "testing frameworks and mocking",
    "logging and monitoring",
    "web scraping and parsing",
    "API design and REST patterns",
    "configuration management",
    "dependency injection",
    "stream processing",
    "graph algorithms",
    "string processing and parsing",
    "numerical computing",
]

# ─── LOGGING ──────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
log(f"Loading {MODEL_ID} in 4-bit NF4...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"GPU {i}: {used:.1f}GB / {total:.1f}GB")

log("Model ready. Generating indefinitely — press Stop to end.\n")

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert software engineer. Your task is to generate
a complex software engineering problem and write a COMPLETE, WORKING Python
implementation for it.

CRITICAL RULES:
- You MUST write the full Python implementation. Every line of code.
- NEVER say the implementation is omitted, too complex, or refer to external docs.
- NEVER write placeholder comments like # implement here or # TODO.
- The implementation field must contain real, runnable Python code.

After thinking, respond with ONLY a valid JSON object using this structure:

{
  "task_title": "short descriptive title",
  "task_description": "detailed requirements in one paragraph",
  "domain": "engineering domain",
  "complexity": "moderate or complex or expert",
  "decomposition_steps": [
    {"step": 1, "title": "step title", "description": "what this step does"},
    {"step": 2, "title": "step title", "description": "what this step does"}
  ],
  "implementation": "import ...\n\nclass ...:\n    def ...",
  "edge_cases": ["edge case 1", "edge case 2", "edge case 3"],
  "test_cases": [
    {"description": "test name", "input": "what you pass in", "expected": "what comes out"}
  ]
}

The implementation field must be real Python code. No exceptions."""

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def extract_json(raw):
    """Extract JSON object from model output with multiple fallback strategies."""
    raw = raw.strip()

    # Strategy 1: strip markdown code fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    # Strategy 2: find first { to last } and try parsing
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON found")

    candidate = raw[start:end]

    # Try direct parse first
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    # Strategy 3: try each } from the end until we find valid JSON
    pos = len(raw) - 1
    while pos >= start:
        pos = raw.rfind("}", start, pos)
        if pos == -1:
            break
        candidate = raw[start:pos+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pos -= 1
            continue

    raise ValueError("Could not extract valid JSON")


def is_valid_python(code):
    """Check that the implementation is at least syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def is_near_duplicate(title, seen_titles, threshold=DUPLICATE_THRESHOLD):
    """Check title similarity against previously seen titles."""
    title_lower = title.lower().strip()
    for seen in seen_titles:
        ratio = difflib.SequenceMatcher(None, title_lower, seen).ratio()
        if ratio >= threshold:
            return True
    return False


def load_seen_titles():
    """Load normalized task_title values from any existing samples, for dedup on resume."""
    titles = []
    if JSONL_PATH.exists():
        with open(JSONL_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    t = obj.get("task_title", "")
                    if t:
                        titles.append(t.lower().strip())
                except json.JSONDecodeError:
                    continue
    return titles


def save_jsonl(sample):
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_parquet():
    rows = []
    if JSONL_PATH.exists():
        with open(JSONL_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    if rows:
        pd.DataFrame(rows).to_parquet(PARQUET_PATH, index=False)
        log(f"Parquet checkpoint: {len(rows)} rows saved")


def count_saved():
    if not JSONL_PATH.exists():
        return 0
    with open(JSONL_PATH) as f:
        return sum(1 for line in f if line.strip())


# ─── GENERATION LOOP ──────────────────────────────────────────────────────────
total_saved  = count_saved()
seen_titles  = load_seen_titles()
domain_cycle = []
log(f"Resuming from {total_saved} saved samples ({len(seen_titles)} titles loaded for dedup).")

while True:
    # Reshuffle the domain order at the start of every full pass, so consecutive
    # samples aren't always generated in the same fixed sequence.
    if not domain_cycle:
        domain_cycle = DOMAINS.copy()
        random.shuffle(domain_cycle)

    domain = domain_cycle.pop()

    user_prompt = (
        f"Generate a complex software engineering task in the domain of: {domain}. "
        f"Write the complete Python implementation — all the code, no placeholders. "
        f"This is task number {total_saved + 1}."
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw        = tokenizer.decode(output_ids, skip_special_tokens=True)

        json_text = extract_json(raw)
        sample    = json.loads(json_text)

        # Validate required fields
        required = ["task_title", "task_description", "implementation",
                    "decomposition_steps", "edge_cases"]
        if not all(k in sample for k in required):
            log("Missing required fields — skipping")
            del inputs, outputs
            torch.cuda.empty_cache()
            continue

        # Reject samples where implementation is a placeholder
        impl = sample.get("implementation", "")
        if len(impl) < MIN_IMPL_LENGTH or "not included" in impl.lower():
            log(f"Implementation too short or missing ({len(impl)} chars) — skipping")
            del inputs, outputs
            torch.cuda.empty_cache()
            continue

        # Reject samples with invalid Python syntax
        if not is_valid_python(impl):
            log(f"Implementation failed syntax check — skipping ({sample.get('task_title','?')})")
            del inputs, outputs
            torch.cuda.empty_cache()
            continue

        # Reject near-duplicate tasks
        title = sample.get("task_title", "")
        if is_near_duplicate(title, seen_titles):
            log(f"Near-duplicate task title — skipping ({title})")
            del inputs, outputs
            torch.cuda.empty_cache()
            continue

        # Enrich with metadata
        sample["id"]         = f"swe_mini_{str(uuid.uuid4())[:8]}"
        sample["source"]     = MODEL_ID
        sample["created_at"] = datetime.now().isoformat()

        save_jsonl(sample)
        seen_titles.append(title.lower().strip())
        total_saved += 1

        log(f"[{total_saved}] {sample.get('task_title','?')} "
            f"| {sample.get('domain','?')} "
            f"| {sample.get('complexity','?')} "
            f"| impl: {len(impl)} chars")

        if total_saved % PARQUET_EVERY == 0:
            save_parquet()

        del inputs, outputs
        torch.cuda.empty_cache()

    except (json.JSONDecodeError, ValueError):
        torch.cuda.empty_cache()
        continue

    except torch.cuda.OutOfMemoryError:
        log("OOM — clearing cache and retrying in 5s")
        torch.cuda.empty_cache()
        time.sleep(5)
        continue

    except KeyboardInterrupt:
        log("Stopped by user.")
        break

    except Exception as e:
        log(f"Unexpected error: {e} — continuing in 3s")
        torch.cuda.empty_cache()
        time.sleep(3)
        continue

# Final save
save_parquet()
log(f"Final count: {total_saved} samples")
log(f"JSONL   : {JSONL_PATH}")
log(f"Parquet : {PARQUET_PATH}")
