# 📚 NLP Project: Evaluating Factual Consistency of LLMs on TriviaQA

This project evaluates the **factual consistency and hallucination rates** of various large language models (LLMs) — including **LLaMA-2**, **Mistral**, and **Mixtral** — using the **TriviaQA** dataset. We use **baseline decoding**, **reranking**, and **self-consistency decoding**, combining cosine similarity and BERTScore to assess semantic correctness.



## 🧠 Key Concepts Applied
| NLP Concept            | Application in Code |
|------------------------|---------------------|
| Vector Space Models    | Cosine similarity via SentenceTransformer |
| Probabilistic Models   | BERTScore (Precision, Recall, F1) |
| Sequence Models        | Decoding strategies using LLMs |
| Word Embeddings        | Used for both cosine and BERTScore similarity |

---

## 🧪 Evaluation Methods

We implemented three decoding strategies:

### 1. **Baseline Decoding**
- Deterministic generation (one output per prompt)
- Evaluated with cosine similarity and BERTScore

### 2. **Reranking**
- Multiple outputs generated using stochastic sampling
- Best output chosen based on semantic similarity (BERTScore and cosine)

### 3. **Self-Consistency Decoding**
- Multiple outputs clustered based on BERTScore similarity
- Most semantically consistent cluster chosen as the final answer

---

## 📊 Metrics Reported
- Accuracy
- Hallucination Rate
- Macro F1 Score
- BERTScore (Precision, Recall, F1)

---

## 🗂️ File Structure
```bash
.
├── FinalEvalCode.py                 # Full implementation
├── README.md                       # This file
├── requirements.txt                # All dependencies
├── llama2_baseline_outputs.json    # Sample output (LLaMA baseline)(All these outputs are generated once the FinalEvalCode.py executed)
├── llama2_reranked_outputs.json    # Reranked results
├── mistral_lightweight_outputs.json
├── mistral_reranked_outputs.json
├── mistral_self_consistency_bertscore_outputs.json
├── mixtral_lightweight_outputs.json
├── mixtral_reranked_outputs.json
└── mixtral_self_consistency_bertscore_outputs.json

📦 Installation & Usage
Step 1: Install Dependencies

pip install -r requirements.txt
Step 2: Run the Code
Due to the size of models like LLaMA-2 and Mixtral, we recommend running the code in Google Colab Pro with GPU or a local machine with A100/H100 GPU.

Make sure to add your Hugging Face token where applicable:



HF_TOKEN = "your_token_here"
📚 Dataset
TriviaQA (unfiltered, no context)

📈 Sample Output

Evaluation Results:
Hallucination Rate: 24.00%
Accuracy: 0.760, Macro F1: 0.733
BERTScore — Precision: 0.850, Recall: 0.843, F1: 0.846


✅ TODOs
 Add support for newer open models (Gemma, Phi-3)

 Visualize hallucination distribution

 Integrate with Hugging Face Spaces

📜 License
This project is for academic/educational purposes.

🙌 Acknowledgements
Hugging Face Transformers and Datasets

SentenceTransformers

BERTScore by T. Zhang et al.

LLaMA-2, Mistral, Mixtral model authors
