##with cross encoder and esembler

import os
import re
import random
import torch
import numpy as np
import logging
import csv
import time
from functools import lru_cache
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------
# Configuration & Seeds
# ----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL      = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL            = "Qwen/Qwen2.5-3B-Instruct"
CHUNK_SIZE           = 1000
CHUNK_OVERLAP        = 200
TOP_K                = 8
RERANK_TOP           = 3
CSV_LOG              = "qa_log_final.csv"

# ----------------------
# Logging & CSV Record
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_csv(question, answer, retrieval_time, cross_encoder_time, llm_time, csv_file=CSV_LOG):
    exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["UserAsked", "ModelAnswer", "RetrievalTime(s)", "CrossEncoderTime(s)", "LLMTime(s)"])
        writer.writerow([question.strip(), answer.strip(), f"{retrieval_time:.4f}", f"{cross_encoder_time:.4f}", f"{llm_time:.4f}"])

def print_timings(retrieval_time, cross_encoder_time, llm_time):
    total_time = retrieval_time + cross_encoder_time + llm_time
    print("\nTiming Summary:")
    print("Component            | Time Taken (s)")
    print("-------------------- | --------------")
    print(f"Hybrid Retrieval    | {retrieval_time:.4f}")
    print(f"Cross-Encoder       | {cross_encoder_time:.4f}")
    print(f"LLM Inference       | {llm_time:.4f}")
    print(f"Total per Query     | {total_time:.4f}")

# ----------------------
# Text Cleaning
# ----------------------
SYNONYM_MAP = {
    r"service\s*centre":       "Service Center",
    r"dealer\s*network":       "Service Center",
    r"nexa\s*network":         "Service Center",
    r"is\s*there":             "where is the",
    r"are\s*there":            "where are the"
}

def normalize_terms(text: str) -> str:
    for pattern, rep in SYNONYM_MAP.items():
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)
    return text

# ----------------------
# Load & Chunk
# ----------------------
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

# ----------------------
# Embeddings & Vector Store
# ----------------------
emb = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": str(device)}
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
    # Get the documents from the vector store for BM25Retriever
    docs = vs.get()['documents']
    docs = [Document(page_content=content, metadata={"id": i}) for i, content in enumerate(docs)]

# ----------------------
# BM25 Setup with LangChain
# ----------------------
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = TOP_K

# ----------------------
# Vector Retriever
# ----------------------
vector_retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

# ----------------------
# Ensemble Retriever
# ----------------------
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

# ----------------------
# Cross-Encoder & LLM
# ----------------------
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=str(device))
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL, torch_dtype=torch.float16
).to(device)

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
    print("debug answer: ", text)
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

# ----------------------
# Hybrid Retrieval with Ensemble and Cross-Encoder
# ----------------------
def hybrid_retrieval(query: str) -> tuple[list, float, float]:
    start_retrieval = time.time()
    normalized_query = normalize_terms(query)
    retrieved_docs = ensemble_retriever.get_relevant_documents(normalized_query)
    retrieval_time = time.time() - start_retrieval
    if not retrieved_docs:
        return [], retrieval_time, 0.0
    pairs = [(normalized_query, doc.page_content) for doc in retrieved_docs]
    start_cross_encoder = time.time()
    ce_scores = cross_encoder.predict(pairs)
    cross_encoder_time = time.time() - start_cross_encoder
    reranked = sorted(zip(retrieved_docs, ce_scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in reranked[:RERANK_TOP]], retrieval_time, cross_encoder_time

# ----------------------
# Main QA Function
# ----------------------
@lru_cache(maxsize=128)
def answer_question(query: str) -> str:
    if not query.strip():
        return "Please enter a valid question."
    normalized_query = normalize_terms(query)
    docs, retrieval_time, cross_encoder_time = hybrid_retrieval(normalized_query)
    if not docs:
        log_to_csv(query, "No relevant information found.", retrieval_time, cross_encoder_time, 0.0)
        print_timings(retrieval_time, cross_encoder_time, 0.0)
        return "No relevant information found."
    ans, llm_time = generate_answer(normalized_query, [d.page_content for d in docs])
    log_to_csv(query, ans, retrieval_time, cross_encoder_time, llm_time)
    print_timings(retrieval_time, cross_encoder_time, llm_time)
    return ans

# ----------------------
# CLI
# ----------------------
if __name__ == "__main__":
    print("Grand Vitara Assistant (Hybrid RAG with LangChain Ensemble)")
    while True:
        q = input("Your question: ")
        if q.lower().strip() in {"exit", "quit"}:
            break
        print("Answer:\n", answer_question(q))