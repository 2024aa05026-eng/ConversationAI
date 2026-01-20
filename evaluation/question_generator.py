import os

# ---------- MAC SAFE MODE ----------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------- MODEL ----------
MODEL_NAME = "google/flan-t5-small"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to("cpu")
model.eval()


# ---------- PATHS ----------
CORPUS_PATH = "data/corpus_chunks.json"
OUTPUT_PATH = "data/eval_questions.json"

NUM_QUESTIONS = 100


# ---------- LOAD CORPUS ----------
print("\nLoading corpus...")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

random.shuffle(corpus)

print("Generating evaluation questions...\n")


results = []


# ---------- SIMPLE PROMPTS ----------

QUESTION_PROMPT = (
    "Create a question based on this text:\n\n"
    "{context}\n\n"
    "Question:"
)

ANSWER_PROMPT = (
    "Answer the following question using the given text.\n\n"
    "Text:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)


# ---------- MAIN LOOP ----------

for item in tqdm(corpus):

    if len(results) >= NUM_QUESTIONS:
        break

    context = item["text"][:300]
    url = item["url"]

    # ----- Generate Question -----
    q_prompt = QUESTION_PROMPT.format(context=context)

    q_inputs = tokenizer(
        q_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        q_outputs = model.generate(
            q_inputs["input_ids"],
            max_new_tokens=50,
            do_sample=False
        )

    question = tokenizer.decode(q_outputs[0], skip_special_tokens=True).strip()

    # If model outputs statement — force it into question
    if not question.endswith("?"):
        question = question + "?"

    if len(question) < 12:
        continue


    # ----- Generate Answer -----
    a_prompt = ANSWER_PROMPT.format(
        context=context,
        question=question
    )

    a_inputs = tokenizer(
        a_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        a_outputs = model.generate(
            a_inputs["input_ids"],
            max_new_tokens=80,
            do_sample=False
        )

    answer = tokenizer.decode(a_outputs[0], skip_special_tokens=True).strip()

    if len(answer) < 12:
        continue


    results.append({
        "question": question,
        "answer": answer,
        "source_url": url
    })


# ---------- SAVE ----------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved", len(results), "questions to", OUTPUT_PATH)
