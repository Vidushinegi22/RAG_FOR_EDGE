# RAG_FOR_EDGE
This repository contains a pipeline for building and evaluating Retrieval-Augmented Generation (RAG) systems, with a focus on car manual question answering (Grand Vitara). It includes data preprocessing, question-answer generation, fine-tuning, and multiple RAG implementations (hybrid, vector-only, etc.).

# Project Structure
RAG_FOR_EDGE/
├── FINETUNING/
│   ├── dataprocessing_part1.py
│   ├── dataprocessing_part2.py
│   ├── output.json
│   └── training.ipynb
└── RAG/
    ├── docling_command(preprocessing1) (1).txt
    ├── hybrid_search.py
    ├── only_vectorsearch (2).py
    └── truemainwithtime.py
# Directory & File Descriptions
# FINETUNING/
1.dataprocessing_part1.py: Extracts text and images from a PDF (e.g., car manual), saving each page's text and images to a CSV for downstream processing. Handles multi-frame/animated images.
2.dataprocessing_part2.py: Loads the CSV, uses a language model (Qwen2.5-3B-Instruct) to generate diverse question-answer pairs for each page's text, and saves the results as a JSON dataset.
3.output.json: The generated dataset containing page-wise text, images, and Q&A pairs, used for fine-tuning or evaluation.
training.ipynb: Jupyter notebook for fine-tuning a Vision-Language Model (VLM) on the generated dataset. Includes data formatting, training, and evaluation code. Uses 4.HuggingFace Transformers, TRL, PEFT, and related libraries.
# RAG/
1.truemainwithtime.py: Implements a hybrid RAG pipeline combining vector search (Chroma + HuggingFaceEmbeddings), BM25, and cross-encoder reranking. Includes logging, evaluation metrics (METEOR, BLEU, ROUGE), and a CLI for interactive QA.
2.hybrid_search.py: Similar to truemainwithtime.py but uses LangChain's EnsembleRetriever for hybrid retrieval (vector + BM25), cross-encoder reranking, and a CLI interface.
only_vectorsearch (2).py: Implements a vector search-only RAG pipeline (no BM25, no cross-encoder). Uses Chroma vector store and HuggingFaceEmbeddings. Includes evaluation metrics and CLI.
3.docling_command(preprocessing1) (1).txt: Likely contains command-line instructions or notes for preprocessing (not essential for running code).
# Getting Started
1. Data Preparation
Place your PDF manual (e.g., GrandVitara 1.pdf) in the working directory.
Run dataprocessing_part1.py to extract text and images to output_dataset/grand_vitara_dataset.csv.
Run dataprocessing_part2.py to generate question-answer pairs and save as output.json.
2. Fine-Tuning (Optional)
Use training.ipynb to fine-tune a Vision-Language Model (VLM) on the generated dataset.
Install required packages: bitsandbytes, peft, trl, transformers, datasets, etc.
3. RAG Pipelines
Hybrid RAG: Run truemainwithtime.py or hybrid_search.py for hybrid retrieval (vector + BM25 + cross-encoder reranking).
Vector Search Only: Run only_vectorsearch (2).py for a pure vector search pipeline.
All scripts provide a CLI for interactive question answering.
# Requirements
Python 3.8+
PyTorch (with CUDA for GPU acceleration)
HuggingFace Transformers, Datasets, TRL, PEFT
LangChain, ChromaDB, rank_bm25, sentence-transformers
NLTK, sacrebleu, rouge-score, PIL, fitz (PyMuPDF), pandas, numpy
Install dependencies (example):

pip install torch transformers datasets trl peft langchain chromadb rank_bm25 sentence-transformers nltk sacrebleu rouge-score pillow pymupdf pandas numpy
Usage Example
# Data extraction
python FINETUNING/dataprocessing_part1.py
python FINETUNING/dataprocessing_part2.py

# (Optional) Fine-tune VLM
jupyter notebook FINETUNING/training.ipynb

# Run Hybrid RAG
python RAG/truemainwithtime.py
# or
python RAG/hybrid_search.py

# Run Vector Search Only
python RAG/only_vectorsearch\ \(2\).py
# Notes
Ensure you have the required models downloaded (e.g., Qwen2.5-3B-Instruct, BAAI/bge-large-en-v1.5, cross-encoder/ms-marco-MiniLM-L-6-v2).
The pipeline is designed for car manual QA but can be adapted for other document types.
For large PDFs, ensure sufficient disk space for extracted images and intermediate files.
This repository contains a pipeline for building and evaluating Retrieval-Augmented Generation (RAG) systems, with a focus on car manual question answering (Grand Vitara). It includes data preprocessing, question-answer generation, fine-tuning, and multiple RAG implementations (hybrid, vector-only, etc.).

Project Structure
RAG_FOR_EDGE/
├── FINETUNING/
│   ├── dataprocessing_part1.py
│   ├── dataprocessing_part2.py
│   ├── output.json
│   └── training.ipynb
└── RAG/
    ├── docling_command(preprocessing1) (1).txt
    ├── hybrid_search.py
    ├── only_vectorsearch (2).py
    └── truemainwithtime.py
Directory & File Descriptions
FINETUNING/
dataprocessing_part1.py: Extracts text and images from a PDF (e.g., car manual), saving each page's text and images to a CSV for downstream processing. Handles multi-frame/animated images.
dataprocessing_part2.py: Loads the CSV, uses a language model (Qwen2.5-3B-Instruct) to generate diverse question-answer pairs for each page's text, and saves the results as a JSON dataset.
output.json: The generated dataset containing page-wise text, images, and Q&A pairs, used for fine-tuning or evaluation.
training.ipynb: Jupyter notebook for fine-tuning a Vision-Language Model (VLM) on the generated dataset. Includes data formatting, training, and evaluation code. Uses HuggingFace Transformers, TRL, PEFT, and related libraries.
RAG/
truemainwithtime.py: Implements a hybrid RAG pipeline combining vector search (Chroma + HuggingFaceEmbeddings), BM25, and cross-encoder reranking. Includes logging, evaluation metrics (METEOR, BLEU, ROUGE), and a CLI for interactive QA.
hybrid_search.py: Similar to truemainwithtime.py but uses LangChain's EnsembleRetriever for hybrid retrieval (vector + BM25), cross-encoder reranking, and a CLI interface.
only_vectorsearch (2).py: Implements a vector search-only RAG pipeline (no BM25, no cross-encoder). Uses Chroma vector store and HuggingFaceEmbeddings. Includes evaluation metrics and CLI.
docling_command(preprocessing1) (1).txt: Likely contains command-line instructions or notes for preprocessing (not essential for running code).
Getting Started
1. Data Preparation
Place your PDF manual (e.g., GrandVitara 1.pdf) in the working directory.
Run dataprocessing_part1.py to extract text and images to output_dataset/grand_vitara_dataset.csv.
Run dataprocessing_part2.py to generate question-answer pairs and save as output.json.
2. Fine-Tuning (Optional)
Use training.ipynb to fine-tune a Vision-Language Model (VLM) on the generated dataset.
Install required packages: bitsandbytes, peft, trl, transformers, datasets, etc.
3. RAG Pipelines
Hybrid RAG: Run truemainwithtime.py or hybrid_search.py for hybrid retrieval (vector + BM25 + cross-encoder reranking).
Vector Search Only: Run only_vectorsearch (2).py for a pure vector search pipeline.
All scripts provide a CLI for interactive question answering.
Requirements
Python 3.8+
PyTorch (with CUDA for GPU acceleration)
HuggingFace Transformers, Datasets, TRL, PEFT
LangChain, ChromaDB, rank_bm25, sentence-transformers
NLTK, sacrebleu, rouge-score, PIL, fitz (PyMuPDF), pandas, numpy
Install dependencies (example):

pip install torch transformers datasets trl peft langchain chromadb rank_bm25 sentence-transformers nltk sacrebleu rouge-score pillow pymupdf pandas numpy
Usage Example
#Data extraction
python FINETUNING/dataprocessing_part1.py
python FINETUNING/dataprocessing_part2.py

#(Optional) Fine-tune VLM
jupyter notebook FINETUNING/training.ipynb

#Run Hybrid RAG
python RAG/truemainwithtime.py
#or
python RAG/hybrid_search.py

# Run Vector Search Only
python RAG/only_vectorsearch\ \(2\).py
# Notes
Ensure you have the required models downloaded (e.g., Qwen2.5-3B-Instruct, BAAI/bge-large-en-v1.5, cross-encoder/ms-marco-MiniLM-L-6-v2).
The pipeline is designed for car manual QA but can be adapted for other document types.
For large PDFs, ensure sufficient disk space for extracted images and intermediate files.
# License
This project is for research and educational purposes. Please check the licenses of individual models and datasets used.
