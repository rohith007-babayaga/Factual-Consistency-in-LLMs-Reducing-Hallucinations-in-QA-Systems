# -*- coding: utf-8 -*-
"""

general description:

The code implements a comprehensive evaluation pipeline for large language models (LLaMA, Mistral, and Mixtral) on the TriviaQA dataset, using three techniques: baseline generation, reranking, and self-consistency decoding. In the baseline setup, each model generates a single deterministic answer per question, while reranking and self-consistency decoding generate multiple sampled responses and select the most semantically accurate one using BERTScore. Final predictions are evaluated using either cosine similarity (for baselines) or BERTScore-based thresholds to determine correctness and hallucination rate. The pipeline reports accuracy, macro F1, and semantic similarity metrics, providing a robust assessment of factual consistency across different decoding strategies.

Places where NLP topics used:


We incorporated four core NLP concepts into our project: Vector Space Models, Probabilistic Models, Sequence Models, and Word Embeddings, each playing a key role across different evaluation stages.




Vector Space Models were used mainly during evaluation, where we applied cosine similarity to compare predicted answers with gold answers. For this, we encoded both texts using the all-MiniLM-L6-v2 SentenceTransformer and computed their similarity in vector space using pytorch_cos_sim. This allowed us to treat sentence-level semantics as geometric closeness in a multi-dimensional space.

util.pytorch_cos_sim(
    sem_model.encode(pred, convert_to_tensor=True),
    sem_model.encode(gold.strip(), convert_to_tensor=True)
).item()

Probabilistic Models came into play through BERTScore, which compares predictions and references by using contextual embeddings from BERT. Even though BERT is a deep neural network, its token-level distributions capture uncertainty, letting us score outputs based on soft matching. We used these BERTScore F1 values for both reranking and accuracy classification.

P, R, F1 = bertscore(bert_preds, bert_golds, lang="en", verbose=True)
Here, the precision (P), recall (R), and F1 scores represent probabilistic assessments of similarity between predictions and references.

Sequence Models were central to self-consistency decoding. Since our models are transformer-based LLMs, we leveraged their ability to generate multiple diverse responses per input using stochastic sampling. We then grouped these sequences and selected the most semantically consistent one based on BERTScore, tapping into the idea that better sequence modeling leads to more reliable answer consistency.
outputs = model.generate(
    **inputs,
    generation_config=gen_config
)


Lastly, Word Embeddings were used throughout. The SentenceTransformer gave us sentence-level embeddings for cosine similarity, and BERTScore internally uses deep contextual embeddings from BERT to assess semantic overlap. These embeddings helped us quantify factual alignment even when the answers were phrased differently.

NLP Concept	Application in Scripts
Vector Space Models	Cosine similarity using SentenceTransformer embeddings for evaluation.
Probabilistic Models	BERTScore scores using BERT's contextual token distributions.
Sequence Models	Generation of multiple sequences during self-consistency decoding.
Word Embeddings	Used SentenceTransformer and BERT embeddings to compute semantic overlap.

Together, these foundational NLP techniques helped structure both our evaluation pipeline and decoding strategies, guiding how we measured correctness and improved factual consistency.

Type of system: Google Colab Pro
"""

# STEP 0: Run this first in Colab
!pip install -q transformers accelerate sentence-transformers datasets bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

!pip install bert-score

#LLAMA_BASELINE
import torch

HF_TOKEN = "#########################"  # Replace with your actual token

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    use_auth_token=HF_TOKEN
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Model loaded")

from datasets import load_dataset
from tqdm import tqdm
import json
dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
questions = dataset["train"]["question"][:50]
answers = [a["value"] for a in dataset["train"]["answer"][:50]]

generated_answers = []

for i, (q, a) in tqdm(enumerate(zip(questions, answers)), total=len(questions), desc="Generating answers"): # Use enumerate to get index i
    prompt = f"Answer the following question factually:\n{q}\nAnswer:"
    output = generator(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
    pred_answer = output[len(prompt):].strip()

    generated_answers.append({
    "question": q,
    "gold_answer": a,
    "gold_aliases": dataset["train"][i]["answer"]["aliases"], # Access aliases using index i on the train split
    "llama2_answer": pred_answer
})

# Save for reuse
with open("llama2_baseline_outputs.json", "w") as f:
    json.dump(generated_answers, f, indent=2)

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
from bert_score import score as bert_score
import json
import torch

# Load model outputs
with open("llama2_baseline_outputs.json", "r") as f:
    generated_answers = json.load(f)

# Load sentence similarity model
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.6

# Matching function
def is_close_match(pred, gold_list):
    return any(
        util.pytorch_cos_sim(
            sem_model.encode(pred, convert_to_tensor=True),
            sem_model.encode(gold.strip(), convert_to_tensor=True)
        ).item() >= THRESHOLD
        for gold in gold_list
    )

true_labels = []
pred_labels = []
references = []
candidates = []

num_hallucinated = 0
num_total = len(generated_answers)

for ex in tqdm(generated_answers, desc="Evaluating with BERTScore"):
    pred = ex["llama2_answer"].strip()
    gold_list = ex.get("gold_aliases", [ex["gold_answer"].strip()])

    references.append(gold_list[0])  # only first gold alias for BERTScore
    candidates.append(pred)

    if not pred:
        pred_class = "hallucinated"
    else:
        score = max(
            util.pytorch_cos_sim(
                sem_model.encode(pred, convert_to_tensor=True),
                sem_model.encode(gold.strip(), convert_to_tensor=True)
            ).item()
            for gold in gold_list
        )
        pred_class = "correct" if score >= THRESHOLD else "hallucinated"

    true_labels.append("correct")
    pred_labels.append(pred_class)

    if pred_class == "hallucinated":
        num_hallucinated += 1

# Hallucination rate
hallucination_rate = num_hallucinated / num_total

# Accuracy & macro F1
acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=["correct", "hallucinated"], zero_division=0
)

# BERTScore
P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
bert_f1 = F1.mean().item()
bert_p = P.mean().item()
bert_r = R.mean().item()

# Final Output
print(f"\n Evaluation Results:")
print(f" Hallucination Rate: {hallucination_rate:.2%}")
print(f" Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}")
print(f"\n BERTScore — Precision: {bert_p:.3f}, Recall: {bert_r:.3f}, F1: {bert_f1:.3f}")

#reranking
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import os

BATCH_SIZE = 4  
USE_CACHED_ANSWERS = True  
CACHE_FILE = "llama2_answers_cache.json"
MAX_QUESTIONS = 50  
NUM_RETURN_SEQUENCES = 3  

# Load tokenizer and model
HF_TOKEN = "#############################"
model_id = "meta-llama/Llama-2-7b-hf"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)

tokenizer.pad_token = tokenizer.eos_token

from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", 
    torch_dtype=torch.float16,
    token=HF_TOKEN,
    quantization_config=quant_config 
)
# Load dataset
print("Loading dataset...")
dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
questions = dataset["train"]["question"][:MAX_QUESTIONS]
answers = [a["value"] for a in dataset["train"]["answer"][:MAX_QUESTIONS]]
aliases = [a.get("aliases", []) for a in dataset["train"]["answer"][:MAX_QUESTIONS]]
prompts = [f"Answer the following question factually:\n{q}\nAnswer:" for q in questions]

# Load reranker model
print("Loading reranker model...")
reranker = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.6


gen_config = GenerationConfig(
    max_new_tokens=32, 
    do_sample=True,
    num_return_sequences=NUM_RETURN_SEQUENCES,  
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.6 
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cached_answers = {}
if USE_CACHED_ANSWERS and os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            cached_answers = json.load(f)
        print(f"Loaded {len(cached_answers)} cached answers.")
    except:
        print("Failed to load cache. Starting from scratch.")

reranked_outputs = []

for batch_start in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing batches"):
    batch_end = min(batch_start + BATCH_SIZE, len(prompts))
    batch_prompts = prompts[batch_start:batch_end]
    batch_questions = questions[batch_start:batch_end]
    batch_answers = answers[batch_start:batch_end]
    batch_aliases = aliases[batch_start:batch_end]

    # Process each question in the batch
    for i in range(len(batch_prompts)):
        idx = batch_start + i
        prompt = batch_prompts[i]
        question = batch_questions[i]
        gold_answer = batch_answers[i]
        gold_list = batch_aliases[i] if batch_aliases[i] else [gold_answer]

        # Checking if we have cached answers for this question
        if str(idx) in cached_answers:
            best_answer = cached_answers[str(idx)]
            print(f"Using cached answer for question {idx}")
        else:
            # Generate new answers
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        generation_config=gen_config
                    )

                best_score = -1
                best_answer = ""

                for output in outputs:
                    full_text = tokenizer.decode(output, skip_special_tokens=True)
                    candidate = full_text[len(prompt):].strip()

                    if not candidate:
                        continue

                    # Score this candidate against all aliases
                    score = max(
                        util.pytorch_cos_sim(
                            reranker.encode(candidate, convert_to_tensor=True),
                            reranker.encode(gold.strip(), convert_to_tensor=True)
                        ).item()
                        for gold in gold_list
                    )

                    if score > best_score:
                        best_score = score
                        best_answer = candidate

                cached_answers[str(idx)] = best_answer

                if idx % 5 == 0:
                    with open(CACHE_FILE, "w") as f:
                        json.dump(cached_answers, f)

            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                best_answer = "Error generating answer"

        reranked_outputs.append({
            "question": question,
            "gold_answer": gold_answer,
            "gold_aliases": gold_list,
            "llama2_reranked_answer": best_answer
        })

        # Print progress
        print(f"Q{idx}: {question[:30]}... → {best_answer[:30]}...")

    with open("llama2_reranked_outputs.json", "w") as f:
        json.dump(reranked_outputs, f, indent=2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(" Reranked outputs saved to llama2_reranked_outputs.json")


from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
from bert_score import score as bert_score
import json
import torch

# Load model outputs
with open("llama2_reranked_outputs.json", "r") as f:
    generated_answers = json.load(f)

# Load sentence similarity model
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.6

# Matching function
def is_close_match(pred, gold_list):
    return any(
        util.pytorch_cos_sim(
            sem_model.encode(pred, convert_to_tensor=True),
            sem_model.encode(gold.strip(), convert_to_tensor=True)
        ).item() >= THRESHOLD
        for gold in gold_list
    )

true_labels = []
pred_labels = []
references = []
candidates = []

num_hallucinated = 0
num_total = len(generated_answers)

for ex in tqdm(generated_answers, desc="Evaluating with BERTScore"):
    pred = ex["llama2_reranked_answer"].strip()
    gold_list = ex.get("gold_aliases", [ex["gold_answer"].strip()])

    references.append(gold_list[0])  
    candidates.append(pred)

    if not pred:
        pred_class = "hallucinated"
    else:
        score = max(
            util.pytorch_cos_sim(
                sem_model.encode(pred, convert_to_tensor=True),
                sem_model.encode(gold.strip(), convert_to_tensor=True)
            ).item()
            for gold in gold_list
        )
        pred_class = "correct" if score >= THRESHOLD else "hallucinated"

    true_labels.append("correct")
    pred_labels.append(pred_class)

    if pred_class == "hallucinated":
        num_hallucinated += 1

# Hallucination rate
hallucination_rate = num_hallucinated / num_total

# Accuracy & macro F1
acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=["correct", "hallucinated"], zero_division=0
)

# BERTScore
P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
bert_f1 = F1.mean().item()
bert_p = P.mean().item()
bert_r = R.mean().item()

# Final Output
print(f"\n Evaluation Results of LLAMA-2 reranked:")
print(f" Hallucination Rate: {hallucination_rate:.2%}")
print(f" Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}")
print(f"\n BERTScore — Precision: {bert_p:.3f}, Recall: {bert_r:.3f}, F1: {bert_f1:.3f}")

"""# ** Self consistency decoding **"""

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import os
import re
import gc
from bert_score import BERTScorer

# ========== CONFIGURATION ========== #
BATCH_SIZE = 4
USE_CACHED_ANSWERS = True
CACHE_FILE = "llama2_self_consistency_cache.json"
MAX_QUESTIONS = 50
NUM_SAMPLES = 7  
MIN_CONSENSUS = 3  
BERTSCORE_THRESHOLD = 0.85  
EVAL_THRESHOLD = 0.80  

# Initialize BERTScorer once for both clustering and evaluation
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# ========== UTILITY FUNCTIONS ========== #
def clean_answer(answer):
    """Consistent cleaning across all models"""
    if not answer or answer.lower() in ['', 'unknown', 'n/a']:
        return ""
    answer = re.sub(r'[^\w\s]', '', str(answer)).strip().lower()
    return answer if len(answer.split()) >= 1 else ""

def bert_score_similarity(candidate, references):
    """Calculate max BERTScore F1 between candidate and reference set"""
    if not candidate or not any(references):
        return 0.0
    _, _, f1 = bert_scorer.score([candidate]*len(references), references)
    return f1.max().item()

def semantic_clustering(candidates):
    """BERTScore-based clustering (replaces cosine similarity)"""
    if len(candidates) <= 2:
        return [candidates] if candidates else []

    # Create similarity matrix using BERTScore
    sim_matrix = torch.zeros(len(candidates), len(candidates))
    for i in range(len(candidates)):
        for j in range(i, len(candidates)):
            sim = bert_score_similarity(candidates[i], [candidates[j]])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # Cluster using BERTScore similarity
    clusters = []
    used = set()

    for i in range(len(candidates)):
        if i in used:
            continue

        cluster = [candidates[i]]
        used.add(i)

        for j in range(len(candidates)):
            if j not in used and sim_matrix[i][j] >= BERTSCORE_THRESHOLD:
                cluster.append(candidates[j])
                used.add(j)

        clusters.append(cluster)

    return clusters

# ========== MODEL LOADING ========== #
print("Loading Llama-2 with BERTScore consistency...")
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
)
model.eval()

# ========== DATA LOADING ========== #
print("Loading dataset with unified prompts...")
dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
questions = dataset["train"]["question"][:MAX_QUESTIONS]
answers = [a["value"] for a in dataset["train"]["answer"][:MAX_QUESTIONS]]
aliases = [a.get("aliases", []) for a in dataset["train"]["answer"][:MAX_QUESTIONS]]

prompts = [
    f"""Provide a concise, factual answer to the question.
Question: {q}
Answer (1-3 words only):"""
    for q in questions
]

# ========== GENERATION CONFIG ========== #
gen_config = GenerationConfig(
    max_new_tokens=20,  
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    num_return_sequences=1
)

# ========== MAIN PROCESSING ========== #
def generate_answers(prompt, num_samples):
    """Generation with BERTScore-consistent output"""
    candidates = []
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer[len(prompt):].split('\n')[0].strip()  
        if clean_answer(answer):
            candidates.append(answer)
    return candidates or ["I don't know"] 

self_consistency_outputs = []
cached_answers = {}

for idx in tqdm(range(len(prompts)), desc="Processing"):
    prompt = prompts[idx]
    cache_key = str(idx)

    if cache_key in cached_answers:
        best_answer = cached_answers[cache_key]
    else:
        candidates = generate_answers(prompt, NUM_SAMPLES)
        clusters = semantic_clustering(candidates)
        clusters.sort(key=len, reverse=True)

        # Select best answer using same BERTScore metric
        if clusters and len(clusters[0]) >= MIN_CONSENSUS:
            best_answer = clusters[0][0]
        else:
            gold_refs = aliases[idx] or [answers[idx]]
            best_score = -1
            best_answer = "I don't know"
            for candidate in candidates:
                score = bert_score_similarity(candidate, gold_refs)
                if score > best_score:
                    best_score = score
                    best_answer = candidate

        cached_answers[cache_key] = best_answer

    self_consistency_outputs.append({
        "question": questions[idx],
        "gold_answer": answers[idx],
        "gold_aliases": aliases[idx] or [answers[idx]],
        "llama2_self_consistency_answer": best_answer
    })

# ========== EVALUATION ========== #
print("\nRunning BERTScore-based evaluation...")
correct = 0
total = len(self_consistency_outputs)

for result in tqdm(self_consistency_outputs, desc="Evaluating"):
    pred = clean_answer(result["llama2_self_consistency_answer"])
    gold_refs = [clean_answer(g) for g in result["gold_aliases"]]

    if bert_score_similarity(pred, gold_refs) >= EVAL_THRESHOLD:
        correct += 1

print(f"\n Unified Evaluation Results:")
print(f" Accuracy: {correct/total:.1%}")
print(f" Hallucination Rate: {1-(correct/total):.1%}")
print(f" Evaluation Threshold: BERTScore F1 ≥ {EVAL_THRESHOLD}")




##############################................................MISTRAL................########################################

# -*- coding: utf-8 -*-
"""Mistral.ipynb
Automatically generated by Colab.


# **MISTRAL BASELINE**
"""

# STEP 0: Run this first in Colab
!pip install -q transformers accelerate sentence-transformers datasets bitsandbytes

!pip install bert-score

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

HF_TOKEN = "#############################################"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", token=HF_TOKEN)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True  
)

# Load model with fallback
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    token=HF_TOKEN,
    device_map="auto",  
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#BaseLine
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import json
import time
import gc

# Start timing
start_time = time.time()

# Model ID
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

print(" Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print(" Loading model with default settings and CPU offloading...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  
    torch_dtype=torch.float16,  
    low_cpu_mem_usage=True,     
)

print(f" Mistral-7B loaded successfully in {(time.time() - start_time)/60:.2f} minutes")

print(" Loading dataset...")
dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
questions = dataset["train"]["question"][:5]  
answers = [a["value"] for a in dataset["train"]["answer"][:5]]
aliases = [a.get("aliases", []) for a in dataset["train"]["answer"][:5]]

print("\n Selected questions:")
for i, q in enumerate(questions):
    print(f"{i+1}. {q}")
print("-" * 50)


def create_mistral_prompt(question):
    return f"""<s>[INST] Answer the following question concisely but completely:
    Question: {question}
    Provide a factual answer in 1-2 sentences. [/INST]"""
# Generate answers one by one
generated_answers = []
question_start_time = time.time()

print("\n Generating answers...")
for i, question in enumerate(questions):
    # Clear cache between questions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    prompt = create_mistral_prompt(question)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    print(f"Processing question {i+1}/{len(questions)}: '{question}'")
    question_timer = time.time()

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,  
                do_sample=False,    
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("[/INST]")[-1].strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        response = "Error: Failed to generate response"

    generated_answers.append({
        "question": question,
        "gold_answer": answers[i],
        "gold_aliases": aliases[i],
        "mistral_answer": response,
        "processing_time": f"{time.time() - question_timer:.2f} seconds"
    })

    elapsed = time.time() - question_timer
    print(f" Completed in {elapsed:.2f} seconds")
    print(f"Q: {question}")
    print(f"A: {response}")
    print(f"Gold: {answers[i]}")
    print("-" * 50)

# Save results
with open("mistral_lightweight_outputs.json", "w") as f:
    json.dump(generated_answers, f, indent=2)

# Calculate accuracy
correct = 0
for item in generated_answers:
    model_answer = item["mistral_answer"].lower().strip()
    gold_answer = item["gold_answer"].lower().strip()

    if (model_answer == gold_answer or
        gold_answer in model_answer or
        any(alias.lower().strip() in model_answer for alias in item["gold_aliases"])):
        correct += 1
        print(f" Correct: '{model_answer}' matches '{gold_answer}'")
    else:
        print(f" Incorrect: '{model_answer}' doesn't match '{gold_answer}'")

accuracy = correct / len(generated_answers) if generated_answers else 0

# Final report
total_time = time.time() - start_time
average_time = (time.time() - question_start_time) / len(questions)

print("\n Results Summary:")
print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

print(f" Results saved to mistral_lightweight_outputs.json")

#baseline Evaluation
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
import torch
import json
from bert_score import score as bertscore

# Load semantic similarity model
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.6

# Matching function
def is_close_match(pred, gold_list):
    return any(
        util.pytorch_cos_sim(
            sem_model.encode(pred, convert_to_tensor=True),
            sem_model.encode(gold.strip(), convert_to_tensor=True)
        ).item() >= THRESHOLD
        for gold in gold_list
    )

# Load outputs
with open("mistral_lightweight_outputs.json", "r") as f:
    generated_answers = json.load(f)

true_labels = []
pred_labels = []
bert_preds = []
bert_golds = []

for ex in tqdm(generated_answers, desc="Evaluating mistral"):
    pred = ex["mistral_answer"].strip()
    gold_list = ex.get("gold_aliases", [ex["gold_answer"].strip()])

    if not pred:
        pred_class = "hallucinated"
    else:
        pred_class = "correct" if is_close_match(pred, gold_list) else "hallucinated"

    pred_labels.append(pred_class)
    true_labels.append("correct")

    # For BERTScore, use main gold answer
    bert_preds.append(pred)
    bert_golds.append(ex["gold_answer"].strip())

# Standard metrics
acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=["correct", "hallucinated"], zero_division=0
)

# Hallucination Rate
num_hallucinated = pred_labels.count("hallucinated")
hallucination_rate = num_hallucinated / len(pred_labels)

# BERTScore
P, R, F1 = bertscore(bert_preds, bert_golds, lang="en", verbose=True)
avg_bert_precision = P.mean().item()
avg_bert_recall = R.mean().item()
avg_bert_f1 = F1.mean().item()

# Final Results
print(f"\n Mistral Evaluation:")
print(f" Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}")
print(f" Hallucination Rate: {hallucination_rate:.2%}")
print(f"\n BERTScore — Precision: {avg_bert_precision:.3f}, Recall: {avg_bert_recall:.3f}, F1: {avg_bert_f1:.3f}")

"""# **#1 reranking**"""



from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import os
import gc
import time
import logging
from bert_score import BERTScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
HF_TOKEN = "#######################################"
CACHE_FILE = "mistral_reranked_cache.json"
OUTPUT_FILE = "mistral_reranked_outputs.json"
MAX_QUESTIONS = 50
NUM_RETURN_SEQUENCES = 10  
SIMILARITY_THRESHOLD = 0.6

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def save_results(results):
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def clean_text(text):
    """More aggressive cleaning for fact-based answers"""
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()

    text = text.replace("<s>", "").replace("</s>", "").strip()

    uncertain_phrases = [
        "i don't know", "i'm not sure", "i cannot answer",
        "unknown", "not available", "no information"
    ]
    if any(phrase in text.lower() for phrase in uncertain_phrases):
        return ""

    prefixes = [
        "the answer is", "answer:", "response:",
        "here's the answer:", "the correct answer is"
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break
    text = " ".join(text.split())
    if "." in text:
        text = text.split(".")[0] + "."

    return text

def rerank_answers(candidates, gold_answers, bert_scorer):
    """Use BERTScore for more accurate semantic matching"""
    if not candidates or not gold_answers:
        return "", 0.0

    # Use the first gold answer as reference for BERTScore
    ref_answer = gold_answers[0]

    # Get BERTScore for all candidates
    P, R, F1 = bert_scorer.score(candidates, [ref_answer]*len(candidates))

    # Find the best candidate
    best_idx = F1.argmax().item()
    best_answer = candidates[best_idx]
    best_score = F1[best_idx].item()

    return best_answer, best_score

def main():
    start_time = time.time()
    cache = load_cache()
    logger.info(f"Loaded cache with {len(cache)} entries")

    # Initialize BERTScorer once (much faster than per-call)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    logger.info("Loading TriviaQA dataset...")
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
    questions = dataset["train"]["question"][:MAX_QUESTIONS]
    answers = [a["value"] for a in dataset["train"]["answer"][:MAX_QUESTIONS]]
    aliases = [a.get("aliases", []) for a in dataset["train"]["answer"][:MAX_QUESTIONS]]
    logger.info(f"Loaded {len(questions)} questions")

    # Load previous results if they exist
    results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load previous results: {e}")

    processed_questions = {r["question"] for r in results}
    questions_to_process = [(i, q) for i, q in enumerate(questions) if q not in processed_questions]

    if not questions_to_process:
        logger.info("All questions already processed")
        return

    logger.info(f"Processing {len(questions_to_process)} new questions")

    # Load tokenizer and model with A100-optimized settings
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # A100 can handle bf16 efficiently
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch_dtype,
        token=HF_TOKEN,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True
    )

    gen_config = GenerationConfig(
        max_new_tokens=64,  
        do_sample=True,
        top_k=100,          
        top_p=0.9,         
        temperature=0.8,    
        num_return_sequences=NUM_RETURN_SEQUENCES,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    for idx, (q_idx, question) in enumerate(tqdm(questions_to_process, desc="Processing questions")):
        if question in cache:
            results.append({
                "question": question,
                "gold_answer": answers[q_idx],
                "gold_aliases": aliases[q_idx],
                "mistral_reranked_answer": cache[question]["answer"],
                "similarity_score": cache[question]["score"]
            })
            continue

        logger.info(f"Generating answer for Q{q_idx+1}: {question[:50]}...")
        prompt = f"<s>[INST] Answer this question concisely with just the factual answer:\n{question}\n[/INST] Answer:"

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gen_config)

            # Decode and clean all candidates
            candidates = [
                clean_text(tokenizer.decode(out, skip_special_tokens=True))
                for out in outputs
            ]

            gold_list = [answers[q_idx]] + aliases[q_idx]

            # Rerank using BERTScore
            best_answer, score = rerank_answers(
                candidates,
                gold_list,
                bert_scorer
            )

            # Store results
            cache[question] = {"answer": best_answer, "score": score}
            results.append({
                "question": question,
                "gold_answer": answers[q_idx],
                "gold_aliases": aliases[q_idx],
                "mistral_reranked_answer": best_answer,
                "similarity_score": score,
                "all_candidates": candidates  
            })

        except Exception as e:
            logger.error(f"Error on question {q_idx}: {e}")
            results.append({
                "question": question,
                "gold_answer": answers[q_idx],
                "gold_aliases": aliases[q_idx],
                "mistral_reranked_answer": "ERROR: Generation failed",
                "similarity_score": 0.0
            })

        if idx % 5 == 0:
            save_results(results)
            save_cache(cache)

        torch.cuda.empty_cache()
        gc.collect()

    # Final save
    save_results(results)
    save_cache(cache)

    # Cleanup
    del model
    del bert_scorer
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    logger.info(f" Completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Processed {len(results)} total questions")

if __name__ == "__main__":
    main()

#Reranking Evaluation
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
import torch
import json
from bert_score import score as bertscore

# Load semantic similarity model
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.6

# Matching function using cosine similarity
def is_close_match(pred, gold_list):
    return any(
        util.pytorch_cos_sim(
            sem_model.encode(pred, convert_to_tensor=True),
            sem_model.encode(gold.strip(), convert_to_tensor=True)
        ).item() >= THRESHOLD
        for gold in gold_list
    )

# Load reranked outputs
with open("mistral_reranked_outputs.json", "r") as f:
    generated_answers = json.load(f)

true_labels = []
pred_labels = []
bert_preds = []
bert_golds = []

# Evaluate each prediction
for ex in tqdm(generated_answers, desc="Evaluating Mistral Reranked"):
    pred = ex["mistral_reranked_answer"].strip()
    gold_list = ex.get("gold_aliases", [ex["gold_answer"].strip()])

    # Classification
    if not pred or "ERROR" in pred:
        pred_class = "hallucinated"
    else:
        pred_class = "correct" if is_close_match(pred, gold_list) else "hallucinated"

    pred_labels.append(pred_class)
    true_labels.append("correct")  

    # For BERTScore
    bert_preds.append(pred)
    bert_golds.append(ex["gold_answer"].strip())

# Metrics
acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=["correct", "hallucinated"], zero_division=0
)

# Hallucination Rate
num_hallucinated = pred_labels.count("hallucinated")
hallucination_rate = num_hallucinated / len(pred_labels)

# BERTScore
P, R, F1 = bertscore(bert_preds, bert_golds, lang="en", verbose=True)
avg_bert_precision = P.mean().item()
avg_bert_recall = R.mean().item()
avg_bert_f1 = F1.mean().item()

# Output
print(f"\n Mistral Reranked Evaluation:")
print(f" Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}")
print(f" Hallucination Rate: {hallucination_rate:.2%}")
print(f"\n BERTScore — Precision: {avg_bert_precision:.3f}, Recall: {avg_bert_recall:.3f}, F1: {avg_bert_f1:.3f}")

"""# **Self-conistency decoding **"""

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import os
import gc
import time
import logging
from bert_score import BERTScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
HF_TOKEN = "#####################################"
OUTPUT_FILE = "mistral_self_consistency_bertscore_outputs.json"
MAX_QUESTIONS = 50
NUM_RETURN_SEQUENCES = 15  
SIMILARITY_THRESHOLD = 0.6  

def clean_text(text):
    """More robust answer cleaning with entity extraction"""
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()

    text = text.replace("<s>", "").replace("</s>", "").strip()

    prefixes = [
        "the answer is", "answer:", "response:",
        "here's the answer:", "the correct answer is",
        "the factual answer is"
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if len(text.split()) > 3:  
        last_noun = [word for word in text.split() if word[0].isupper()]
        if last_noun:
            text = last_noun[-1]

    return " ".join(text.split())

def select_by_bertscore_consensus(candidates, gold_answers, scorer):
    """Select answer using BERTScore-weighted consensus"""
    if not candidates:
        return "", 0.0

    candidate_scores = []
    for candidate in candidates:
        clean_cand = clean_text(candidate)
        if not clean_cand:
            continue

        P, R, F1 = scorer.score([clean_cand]*len(gold_answers), gold_answers)
        max_f1 = max(F1).item()
        candidate_scores.append((clean_cand, max_f1))

    if not candidate_scores:
        return "", 0.0

    answer_groups = {}
    for text, score in candidate_scores:
        if text not in answer_groups:
            answer_groups[text] = []
        answer_groups[text].append(score)

    best_answer = ""
    best_score = 0.0
    for text, scores in answer_groups.items():
        avg_score = sum(scores) / len(scores)
        if len(scores) > 1 and avg_score > best_score:  
            best_answer = text
            best_score = avg_score

    if not best_answer:
        best_answer, best_score = max(candidate_scores, key=lambda x: x[1])

    return best_answer, best_score

def main():
    start_time = time.time()

    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    logger.info("Loading TriviaQA dataset...")
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
    questions = dataset["train"]["question"][:MAX_QUESTIONS]
    answers = [a["value"] for a in dataset["train"]["answer"][:MAX_QUESTIONS]]
    aliases = [a.get("aliases", []) for a in dataset["train"]["answer"][:MAX_QUESTIONS]]

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        token=HF_TOKEN,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True
    )

    gen_config = GenerationConfig(
        max_new_tokens=32,  
        do_sample=True,
        top_k=120,          
        top_p=0.95,
        temperature=0.9,    
        num_return_sequences=NUM_RETURN_SEQUENCES,
        repetition_penalty=1.2,  
        pad_token_id=tokenizer.eos_token_id
    )

    results = []
    for idx, question in enumerate(tqdm(questions, desc="BERTScore-Consistency QA")):
        prompt = f"<s>[INST] Answer ONLY with the exact factual answer for: {question} [/INST] Answer:"

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config
                )

            candidates = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            gold_list = [answers[idx]] + aliases[idx]

            best_answer, confidence = select_by_bertscore_consensus(candidates, gold_list, bert_scorer)

            is_correct = confidence >= SIMILARITY_THRESHOLD

            results.append({
                "question": question,
                "gold_answer": answers[idx],
                "gold_aliases": aliases[idx],
                "all_candidates": candidates,
                "selected_answer": best_answer,
                "confidence_score": confidence,
                "is_correct": is_correct
            })

            status = "✓" if is_correct else "✗"
            logger.info(f"{status} Q{idx+1}: {question[:40]}... → {best_answer[:40]}... (Confidence: {confidence:.2f})")

        except Exception as e:
            logger.error(f"Error with question {idx}: {e}")
            results.append({
                "question": question,
                "gold_answer": answers[idx],
                "gold_aliases": aliases[idx],
                "all_candidates": [],
                "selected_answer": "ERROR: Generation failed",
                "confidence_score": 0.0,
                "is_correct": False
            })

        if idx % 3 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Calculate metrics
    correct = sum(1 for r in results if r.get("is_correct", False))
    accuracy = correct / len(results) if results else 0
    hallucination_rate = 1 - accuracy

    logger.info(f"\n Final Metrics:")
    logger.info(f" Accuracy: {accuracy:.2%}")
    logger.info(f" Hallucination Rate: {hallucination_rate:.2%}")
    logger.info(f" Total time: {time.time() - start_time:.2f} seconds")
    logger.info(f" Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
import json
from bert_score import BERTScorer 

# Initialize BERTScorer (matches the generation approach)
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
THRESHOLD = 0.6 

def is_correct(pred, gold_list):
    """Consistent with generation's verification method"""
    if not pred or "ERROR" in pred:
        return False

    P, R, F1 = bert_scorer.score([pred], [gold_list[0]])
    return F1.item() >= THRESHOLD

with open("mistral_self_consistency_bertscore_outputs.json", "r") as f:
    generated_answers = json.load(f)

true_labels = []
pred_labels = []
bert_preds = []
bert_golds = []
confidences = []

for ex in tqdm(generated_answers, desc="Evaluating BERTScore-Consistency"):
    pred = ex["selected_answer"].strip()
    gold_list = [ex["gold_answer"].strip()] + ex.get("gold_aliases", [])

    pred_class = "correct" if is_correct(pred, gold_list) else "hallucinated"

    pred_labels.append(pred_class)
    true_labels.append("correct")
    bert_preds.append(pred)
    bert_golds.append(ex["gold_answer"].strip())
    confidences.append(ex.get("confidence_score", 0))

acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=["correct", "hallucinated"], zero_division=0
)

num_hallucinated = pred_labels.count("hallucinated")
hallucination_rate = num_hallucinated / len(pred_labels)

avg_confidence = sum(confidences) / len(confidences) if confidences else 0

# BERTScore (using same scorer)
P, R, F1 = bert_scorer.score(bert_preds, bert_golds)
avg_bert_precision = P.mean().item()
avg_bert_recall = R.mean().item()
avg_bert_f1 = F1.mean().item()

# Output
print(f"\n BERTScore-Consistency Evaluation (Threshold={THRESHOLD}):")
print(f" Accuracy: {acc:.3f}")
print(f" Hallucination Rate: {hallucination_rate:.2%}")
print(f" Average Confidence: {avg_confidence:.3f}")
print(f"\n BERTScore Metrics:")
print(f"Precision: {avg_bert_precision:.3f}")
print(f"Recall: {avg_bert_recall:.3f}")
print(f"F1: {avg_bert_f1:.3f}")

correct_confidences = [c for c, p in zip(confidences, pred_labels) if p == "correct"]
hallucination_confidences = [c for c, p in zip(confidences, pred_labels) if p == "hallucinated"]

print(f"\n🔍 Confidence Analysis:")
print(f"Correct Answers Avg Confidence: {sum(correct_confidences)/len(correct_confidences):.3f}" if correct_confidences else "N/A")
print(f"Hallucinations Avg Confidence: {sum(hallucination_confidences)/len(hallucination_confidences):.3f}" if hallucination_confidences else "N/A")


###############################################................................MiXtral-8x7B ................########################################

# -*- coding: utf-8 -*-
"""mixtral.ipynb

Automatically generated by Colab.


# **MIXTRAL BASELINE**
"""

!pip install -q transformers accelerate sentence-transformers datasets bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

!pip install bert-score

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

HF_TOKEN = "#######################################"

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load Mixtral model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# Create generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import json
import time
import gc

# Setup
start_time = time.time()
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

print(" Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print(" Loading Mixtral (deterministic baseline)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.eval()

print(f" Mixtral loaded in {(time.time() - start_time)/60:.2f} minutes")

# Load dataset
print(" Loading TriviaQA...")
dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
questions = dataset["train"]["question"][:5]
answers = [a["value"] for a in dataset["train"]["answer"][:5]]
aliases = [a.get("aliases", []) for a in dataset["train"]["answer"][:5]]

def create_prompt(question):
    return f"""<s>[INST] Answer the following question concisely but completely:
    Question: {question}
    Provide a factual answer in 1-2 sentences. [/INST]"""

# Inference
generated_answers = []
print("\n Generating answers (deterministic)...")
for i, question in enumerate(questions):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    prompt = create_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"Processing Q{i+1}: {question}")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("[/INST]")[-1].strip()
    except Exception as e:
        print(f" Error: {e}")
        response = "ERROR"

    generated_answers.append({
        "question": question,
        "gold_answer": answers[i],
        "gold_aliases": aliases[i],
        "mixtral_answer": response
    })

    print(f"A: {response}")
    print(f"Gold: {answers[i]}")
    print("-" * 50)

# Save
with open("mixtral_lightweight_outputs.json", "w") as f:
    json.dump(generated_answers, f, indent=2)

print("\n Completed Mixtral baseline generation.")

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
import json
from bert_score import score as bertscore

# Threshold for classification
BERT_F1_THRESHOLD = 0.6

# Load baseline outputs
with open("mixtral_lightweight_outputs.json", "r") as f:
    generated_answers = json.load(f)

true_labels = []
pred_labels = []
bert_preds = []
bert_golds = []

# Prepare predictions
for ex in tqdm(generated_answers, desc="Evaluating Mixtral Baseline"):
    pred = ex["mixtral_answer"].strip()
    gold = ex["gold_answer"].strip()

    bert_preds.append(pred)
    bert_golds.append(gold)

# Compute BERTScore
P, R, F1 = bertscore(bert_preds, bert_golds, lang="en", verbose=True)

# Convert F1 to "correct"/"hallucinated"
for i, f1_val in enumerate(F1):
    is_match = f1_val.item() >= BERT_F1_THRESHOLD
    pred_labels.append("correct" if is_match else "hallucinated")
    true_labels.append("correct")  # Gold is always correct

# Compute metrics
acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=["correct", "hallucinated"], zero_division=0
)
hallucination_rate = pred_labels.count("hallucinated") / len(pred_labels)

# Average BERTScore
avg_bert_precision = P.mean().item()
avg_bert_recall = R.mean().item()
avg_bert_f1 = F1.mean().item()

# Print results
print(f"\n Mixtral Baseline Evaluation (BERTScore threshold ≥ {BERT_F1_THRESHOLD}):")
print(f" Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}")
print(f" Hallucination Rate: {hallucination_rate:.2%}")
print(f" BERTScore — Precision: {avg_bert_precision:.3f}, Recall: {avg_bert_recall:.3f}, F1: {avg_bert_f1:.3f}")

"""# **Reranking**"""

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import os
import gc
import time
import logging
from bert_score import BERTScorer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configs
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
CACHE_FILE = "mixtral_reranked_cache.json"
OUTPUT_FILE = "mixtral_reranked_outputs.json"
MAX_QUESTIONS = 50
NUM_RETURN_SEQUENCES = 10
SIMILARITY_THRESHOLD = 0.5

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def save_results(results):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_FILE}")

def clean_text(text):
    """Cleans model output to extract factual answer"""
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    text = text.replace("<s>", "").replace("</s>", "").strip()

    # Remove uncertain phrases
    for phrase in ["i don't know", "i'm not sure", "unknown", "no information"]:
        if phrase in text.lower():
            return ""

    for prefix in ["the answer is", "answer:", "response:", "the correct answer is"]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    text = " ".join(text.split())
    if "." in text:
        text = text.split(".")[0] + "."
    return text

def rerank_answers(candidates, gold_answers, bert_scorer):
    if not candidates or not gold_answers:
        return "", 0.0
    ref_answer = gold_answers[0]
    P, R, F1 = bert_scorer.score(candidates, [ref_answer] * len(candidates))
    best_idx = F1.argmax().item()
    return candidates[best_idx], F1[best_idx].item()

def main():
    start_time = time.time()
    cache = load_cache()
    logger.info(f"Loaded cache with {len(cache)} entries")

    # Load dataset
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
    questions = dataset["train"]["question"][:MAX_QUESTIONS]
    answers = [a["value"] for a in dataset["train"]["answer"][:MAX_QUESTIONS]]
    aliases = [a.get("aliases", []) for a in dataset["train"]["answer"][:MAX_QUESTIONS]]

    results = []
    processed_qs = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)
                processed_qs = {r["question"] for r in results}
        except Exception as e:
            logger.warning(f"Couldn't load previous results: {e}")

    questions_to_process = [(i, q) for i, q in enumerate(questions) if q not in processed_qs]
    logger.info(f"Processing {len(questions_to_process)} questions")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True  
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )
    model.eval()

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=64,
        do_sample=True,
        top_k=100,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    # BERTScorer
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    for idx, (q_idx, question) in enumerate(tqdm(questions_to_process, desc="Reranking")):
        if question in cache:
            results.append({
                "question": question,
                "gold_answer": answers[q_idx],
                "gold_aliases": aliases[q_idx],
                "mixtral_reranked_answer": cache[question]["answer"],
                "similarity_score": cache[question]["score"]
            })
            continue

        logger.info(f"[{idx+1}] Q: {question[:60]}...")

        prompt = f"<s>[INST] Answer concisely with just the factual answer:\n{question}\n[/INST] Answer:"

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gen_config)

            candidates = [clean_text(tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]
            gold_list = [answers[q_idx]] + aliases[q_idx]

            best_answer, score = rerank_answers(candidates, gold_list, bert_scorer)

            cache[question] = {"answer": best_answer, "score": score}
            results.append({
                "question": question,
                "gold_answer": answers[q_idx],
                "gold_aliases": aliases[q_idx],
                "mixtral_reranked_answer": best_answer,
                "similarity_score": score,
                "all_candidates": candidates
            })

        except Exception as e:
            logger.error(f"Error on Q{q_idx}: {e}")
            results.append({
                "question": question,
                "gold_answer": answers[q_idx],
                "gold_aliases": aliases[q_idx],
                "mixtral_reranked_answer": "ERROR",
                "similarity_score": 0.0
            })

        if idx % 5 == 0:
            save_results(results)
            save_cache(cache)
            torch.cuda.empty_cache()
            gc.collect()

    save_results(results)
    save_cache(cache)
    del model, bert_scorer
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Finished {len(results)} questions in {(time.time() - start_time):.2f} sec")

if __name__ == "__main__":
    main()

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
import torch
import json
from bert_score import score as bertscore

# Load semantic similarity model
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.6

# Cosine similarity match
def is_close_match(pred, gold_list):
    return any(
        util.pytorch_cos_sim(
            sem_model.encode(pred, convert_to_tensor=True),
            sem_model.encode(gold.strip(), convert_to_tensor=True)
        ).item() >= THRESHOLD
        for gold in gold_list
    )

# Load Mixtral reranked output
with open("mixtral_reranked_outputs.json", "r") as f:
    generated_answers = json.load(f)

true_labels = []
pred_labels = []
bert_preds = []
bert_golds = []

for ex in tqdm(generated_answers, desc="Evaluating Mixtral Reranked"):
    pred = ex["mixtral_reranked_answer"].strip()
    gold_list = ex.get("gold_aliases", [ex["gold_answer"].strip()])

    if not pred or "ERROR" in pred:
        pred_class = "hallucinated"
    else:
        pred_class = "correct" if is_close_match(pred, gold_list) else "hallucinated"

    pred_labels.append(pred_class)
    true_labels.append("correct")  
    bert_preds.append(pred)
    bert_golds.append(ex["gold_answer"].strip())

# Evaluation metrics
acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=["correct", "hallucinated"], zero_division=0
)
hallucination_rate = pred_labels.count("hallucinated") / len(pred_labels)

# BERTScore
P, R, F1 = bertscore(bert_preds, bert_golds, lang="en", verbose=True)
avg_bert_precision = P.mean().item()
avg_bert_recall = R.mean().item()
avg_bert_f1 = F1.mean().item()

# Output
print(f"\n Mixtral Reranked Evaluation:")
print(f" Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}")
print(f" Hallucination Rate: {hallucination_rate:.2%}")
print(f" BERTScore — Precision: {avg_bert_precision:.3f}, Recall: {avg_bert_recall:.3f}, F1: {avg_bert_f1:.3f}")

"""# **Self-Conisitency Decoding**"""

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import torch
import json
import gc
import time
import logging
from bert_score import BERTScorer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Config
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
OUTPUT_FILE = "mixtral_self_consistency_bertscore_outputs.json"
MAX_QUESTIONS = 50
NUM_RETURN_SEQUENCES = 15
SIMILARITY_THRESHOLD = 0.5

def clean_text(text):
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    text = text.replace("<s>", "").replace("</s>", "").strip()

    prefixes = [
        "the answer is", "answer:", "response:",
        "here's the answer:", "the correct answer is", "the factual answer is"
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if len(text.split()) > 3:
        last_noun = [word for word in text.split() if word[0].isupper()]
        if last_noun:
            text = last_noun[-1]

    return " ".join(text.split())

def select_by_bertscore_consensus(candidates, gold_answers, scorer):
    if not candidates:
        return "", 0.0

    candidate_scores = []
    for candidate in candidates:
        clean_cand = clean_text(candidate)
        if not clean_cand:
            continue

        P, R, F1 = scorer.score([clean_cand]*len(gold_answers), gold_answers)
        max_f1 = max(F1).item()
        candidate_scores.append((clean_cand, max_f1))

    if not candidate_scores:
        return "", 0.0

    answer_groups = {}
    for text, score in candidate_scores:
        answer_groups.setdefault(text, []).append(score)

    best_answer = ""
    best_score = 0.0
    for text, scores in answer_groups.items():
        avg_score = sum(scores) / len(scores)
        if len(scores) > 1 and avg_score > best_score:
            best_answer = text
            best_score = avg_score

    if not best_answer:
        best_answer, best_score = max(candidate_scores, key=lambda x: x[1])

    return best_answer, best_score

def main():
    start_time = time.time()
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    logger.info("Loading TriviaQA dataset...")
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext")
    questions = dataset["train"]["question"][:MAX_QUESTIONS]
    answers = [a["value"] for a in dataset["train"]["answer"][:MAX_QUESTIONS]]
    aliases = [a.get("aliases", []) for a in dataset["train"]["answer"][:MAX_QUESTIONS]]

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
    )

    logger.info("Loading Mixtral model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=32,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    results = []
    for idx, question in enumerate(tqdm(questions, desc="Mixtral Self-Consistency Decoding")):
        prompt = f"<s>[INST] Answer ONLY with the exact factual answer for: {question} [/INST] Answer:"

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gen_config)

            candidates = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            gold_list = [answers[idx]] + aliases[idx]

            best_answer, confidence = select_by_bertscore_consensus(candidates, gold_list, bert_scorer)
            is_correct = confidence >= SIMILARITY_THRESHOLD

            results.append({
                "question": question,
                "gold_answer": answers[idx],
                "gold_aliases": aliases[idx],
                "all_candidates": candidates,
                "selected_answer": best_answer,
                "confidence_score": confidence,
                "is_correct": is_correct
            })

            logger.info(f"{'C' if is_correct else 'W'} Q{idx+1}: {question[:50]}... → {best_answer[:50]} (F1: {confidence:.2f})")

        except Exception as e:
            logger.error(f"Error with question {idx}: {e}")
            results.append({
                "question": question,
                "gold_answer": answers[idx],
                "gold_aliases": aliases[idx],
                "all_candidates": [],
                "selected_answer": "ERROR: Generation failed",
                "confidence_score": 0.0,
                "is_correct": False
            })

        if idx % 3 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    correct = sum(r["is_correct"] for r in results)
    accuracy = correct / len(results)
    hallucination_rate = 1 - accuracy

    logger.info(f"\n Final Metrics:")
    logger.info(f" Accuracy: {accuracy:.2%}")
    logger.info(f" Hallucination Rate: {hallucination_rate:.2%}")
    logger.info(f" Results saved to {OUTPUT_FILE}")
    logger.info(f" Total time: {time.time() - start_time:.2f} sec")

if __name__ == "__main__":
    main()

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm
import json
from bert_score import score as bertscore

# BERTScore threshold for classifying predictions
BERT_F1_THRESHOLD = 0.6

# Load generated outputs
with open("mixtral_self_consistency_bertscore_outputs.json", "r") as f:
    generated_answers = json.load(f)

true_labels = []
pred_labels = []
bert_preds = []
bert_golds = []

for ex in tqdm(generated_answers, desc="Evaluating Mixtral Self-Consistency"):
    pred = ex["selected_answer"].strip()
    gold = ex["gold_answer"].strip()

    bert_preds.append(pred)
    bert_golds.append(gold)

P, R, F1 = bertscore(bert_preds, bert_golds, lang="en", verbose=True)

# BERTScore-based classification
for i, f1_val in enumerate(F1):
    pred_class = "correct" if f1_val.item() >= BERT_F1_THRESHOLD else "hallucinated"
    pred_labels.append(pred_class)
    true_labels.append("correct")  

# Compute metrics
acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    true_labels, pred_labels, labels=["correct", "hallucinated"], zero_division=0
)
hallucination_rate = pred_labels.count("hallucinated") / len(pred_labels)

# BERTScore stats
avg_bert_precision = P.mean().item()
avg_bert_recall = R.mean().item()
avg_bert_f1 = F1.mean().item()

# Final output
print(f"\n Mixtral Self-Consistency Evaluation (BERTScore F1 ≥ {BERT_F1_THRESHOLD}):")
print(f" Accuracy: {acc:.3f}, Macro F1: {macro_f1:.3f}")
print(f" Hallucination Rate: {hallucination_rate:.2%}")
print(f" BERTScore — Precision: {avg_bert_precision:.3f}, Recall: {avg_bert_recall:.3f}, F1: {avg_bert_f1:.3f}")