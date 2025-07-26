import os
import re
import random
import torch
import numpy as np
import logging
import csv
import time
from functools import lru_cache
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
import sacrebleu

# Configuration & Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0")
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RERANK_TOP = 3
CSV_LOG = "qa_log_final.csv"

# Logging & CSV Record
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_csv(question, answer, f1_score, bleu_score, retrieval_time, llm_time, f1_time, bleu_time, csv_file=CSV_LOG):
    exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["UserAsked", "ModelAnswer", "F1_Score", "BLEU_Score", "RetrievalTime(s)", "LLMTime(s)", "F1Time(s)", "BLEUTime(s)"])
        writer.writerow([question.strip(), answer.strip(), f1_score if f1_score is not None else '', bleu_score if bleu_score is not None else '', 
                        f"{retrieval_time:.4f}", f"{llm_time:.4f}", f"{f1_time:.4f}", f"{bleu_time:.4f}"])

def print_timings(retrieval_time, llm_time, f1_time, bleu_time):
    total_time = retrieval_time + llm_time + f1_time + bleu_time
    print("\nTiming Summary:")
    print("Component            | Time Taken (s)")
    print("-------------------- | --------------")
    print(f"Vector Retrieval    | {retrieval_time:.4f}")
    print(f"LLM Inference       | {llm_time:.4f}")
    print(f"F1 Score            | {f1_time:.4f}")
    print(f"BLEU Score          | {bleu_time:.4f}")
    print(f"Total per Query     | {total_time:.4f}")

# Text Cleaning
SYNONYM_MAP = {
    r"service\s*centre": "Service Center",
    r"dealer\s*network": "Service Center",
    r"nexa\s*network": "Service Center",
    r"is\s*there": "where is the",
    r"are\s*there": "where are the"
}

def normalize_terms(text: str) -> str:
    for pattern, rep in SYNONYM_MAP.items():
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)
    return text

# Load & Chunk
path = "GrandVitara1.md"
if not os.path.exists(path):
    logger.error(f"Manual not found: {path}")
    raise FileNotFoundError(path)

with open(path, "r", encoding="utf-8") as f:
    raw = f.read()
cleaned = normalize_terms(" ".join(raw.split()))

def custom_splitter(text):
    pattern = r'(?=##\s[A-Z]+(?:\s[A-Z]+)*\s-\s[A-Z]+.*)'
    chunks = re.split(pattern, text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    combined_chunks = []
    i = 0
    while i < len(chunks):
        if chunks[i].startswith("##"):
            combined_chunks.append(chunks[i])
            i += 1
        elif i + 1 < len(chunks):
            combined_chunks.append(chunks[i] + chunks[i + 1])
            i += 2
        else:
            i += 1
    return combined_chunks

raw_chunks = custom_splitter(cleaned)
chunk_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "(?<=\. )", " "]
)

final_chunks = []
for chunk in raw_chunks:
    if len(chunk) <= CHUNK_SIZE:
        final_chunks.append(chunk)
    else:
        sub_chunks = chunk_splitter.split_text(chunk)
        final_chunks.extend(sub_chunks)

texts, seen = [], set()
for t in final_chunks:
    txt = normalize_terms(t).strip()
    if txt and txt not in seen:
        seen.add(txt)
        texts.append(txt)

logger.info(f"Prepared {len(texts)} unique chunks")

# Embeddings & Vector Store
emb = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device}
)

store_dir = "./chroma_db_hybrid_openai_3"
if not os.path.exists(store_dir):
    docs = [Document(page_content=t, metadata={"id": i}) for i, t in enumerate(texts)]
    vs = Chroma.from_documents(
        documents=docs,
        embedding=emb,
        collection_name="grand_vitara",
        persist_directory=store_dir
    )
    vs.persist()
    logger.info("Vector store built.")
else:
    vs = Chroma(
        persist_directory=store_dir,
        collection_name="grand_vitara",
        embedding_function=emb
    )
    logger.info("Vector store loaded.")

# LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL, torch_dtype=torch.float16
).to(device)

# Generate Answer
def generate_answer(query: str, contexts: list[str]) -> tuple[str, float]:
    prompt = """You are a Grand Vitara Car Chatbot. Provide only one answer to the question and do not generate additional questions or context. Use the following information to answer the question concisely and directly. Do not include any additional details, explanations, or references to the information sources.
"""
    for i, c in enumerate(contexts, 1):
        prompt += f"Context {i}: {c}\n"
    prompt += f"\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    start_time = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.1,
        top_p=1.0,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )
    llm_time = time.time() - start_time
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("debug answer:", text)
    if "Answer:" in text:
        after_label = text.split("Answer:", 1)[1].strip()
        if after_label.startswith("You are a Grand Vitara Car Chatbot."):
            return "please refer to the manual", llm_time
        else:
            tokens = tokenizer(after_label, return_tensors="pt").input_ids
            if len(tokens[0]) < 15 and after_label.strip().endswith(":"):
                answer = after_label.strip()
            else:
                double_newline_index = after_label.find("\n\n")
                question_index = after_label.lower().find("question:")
                context_index = after_label.lower().find("context:")
                human_index = after_label.lower().find("human:")
                truncate_index = None
                if double_newline_index != -1:
                    truncate_index = double_newline_index
                if question_index != -1 and (truncate_index is None or question_index < truncate_index):
                    truncate_index = question_index
                if context_index != -1 and (truncate_index is None or context_index < truncate_index):
                    truncate_index = context_index
                if human_index != -1 and (truncate_index is None or human_index < truncate_index):
                    truncate_index = human_index
                if truncate_index is not None:
                    answer = after_label[:truncate_index].strip()
                else:
                    answer = after_label.strip()
            return answer, llm_time
    else:
        return "please refer to the manual", llm_time

# Vector Retrieval (No BM25, No Cross Encoder)
def vector_retrieval(query: str) -> tuple[list, float]:
    start_retrieval = time.time()
    q = normalize_terms(query)
    vect_and_scores = vs.similarity_search_with_score(q, k=RERANK_TOP)
    retrieval_time = time.time() - start_retrieval
    if not vect_and_scores:
        return [], retrieval_time
    return [doc for doc, _ in vect_and_scores], retrieval_time

# Compute F1 Score
def compute_f1(prediction, reference) -> tuple[float, float]:
    start_f1 = time.time()
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(target=reference, prediction=prediction)
    f1_time = time.time() - start_f1
    return scores["rouge1"].fmeasure, f1_time

# Compute BLEU Score
def compute_bleu(prediction, reference) -> tuple[float, float]:
    start_bleu = time.time()
    bleu = sacrebleu.corpus_bleu([prediction], [[reference]])
    bleu_time = time.time() - start_bleu
    return bleu.score, bleu_time

# Main QA Function
@lru_cache(maxsize=128)
def answer_question(query: str) -> tuple:
    if not query.strip():
        return "Please enter a valid question.", None, None, 0.0, 0.0, 0.0, 0.0

    normalized_query = normalize_terms(query)
    docs, retrieval_time = vector_retrieval(normalized_query)

    if not docs:
        log_to_csv(query, "No relevant information found.", None, None, retrieval_time, 0.0, 0.0, 0.0)
        print_timings(retrieval_time, 0.0, 0.0, 0.0)
        return "No relevant information found.", None, None, retrieval_time, 0.0, 0.0, 0.0

    ans, llm_time = generate_answer(normalized_query, [d.page_content for d in docs])

    reference_answer = "The service center for Mandovi Motors Pvt Ltd located at Gonikopall can be reached by phone number +91 9900064873."
    f1, f1_time = compute_f1(ans, reference_answer)
    bleu_score, bleu_time = compute_bleu(ans, reference_answer)

    log_to_csv(query, ans, f1, bleu_score, retrieval_time, llm_time, f1_time, bleu_time)
    print_timings(retrieval_time, llm_time, f1_time, bleu_time)

    return ans, f1, bleu_score, retrieval_time, llm_time, f1_time, bleu_time

# CLI
if __name__ == "__main__":
    print("Grand Vitara Assistant (Vector Search Only)")
    while True:
        q = input("Your question: ")
        if q.lower().strip() in {"exit", "quit"}:
            break
        ans, f1, bleu_score, retrieval_time, llm_time, f1_time, bleu_time = answer_question(q)
        print(f"Answer:\n{ans}")
        if f1 is not None and bleu_score is not None:
            print(f"F1 Score: {f1:.4f}")
            print(f"BLEU Score: {bleu_score:.4f}")