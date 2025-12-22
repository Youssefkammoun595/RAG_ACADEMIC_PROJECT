# 📄 Advanced RAG System for PDF Document Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Models-orange)
![Gradio](https://img.shields.io/badge/Interface-Gradio-ff69b4)

## 🌟 Overview

An advanced **Retrieval-Augmented Generation (RAG)** system specialized in intelligent analysis of PDF documents. This system combines state-of-the-art NLP techniques, hybrid search, and multilingual language models to provide accurate, context-aware responses based on document content.

## ✨ Key Features

### 🔍 **Advanced PDF Processing**
- **Multi-method extraction**: PyPDF2 + pdfplumber with table and metadata extraction
- **Structural analysis**: Automatic section, title, and table detection
- **Enhanced metadata**: Page-level information, fonts, layout analysis
- **Multi-language support**: Native French/English processing

### 🧠 **Intelligent Chunking & Indexing**
- **Adaptive chunking**: Multiple strategies (semantic, sentence-based) with configurable overlap
- **Context preservation**: Smart overlap between chunks (250 characters by default)
- **Dual embedding models**: 
  - `paraphrase-multilingual-MiniLM-L12-v2` (multilingual)
  - `all-MiniLM-L6-v2` (English-optimized)
- **Hybrid search**: FAISS vector search + BM25 lexical search + entity-based search

### 🤖 **Advanced Generation**
- **Mistral-7B-Instruct**: 8-bit quantized for memory efficiency
- **Adaptive prompts**: Question-type specific instructions
- **Context-aware responses**: Structured answers with source citations
- **Cross-Encoder re-ranking**: Enhanced relevance scoring

### 📊 **Document Analysis**
- **Named Entity Recognition**: Person, organization, location detection
- **Sentiment analysis**: Document tone and polarity assessment
- **Reading level estimation**: Basic/Intermediate/Advanced classification
- **Keyword extraction**: Top 20 keywords with frequency

### 🎨 **User Interface**
- **Gradio web interface**: Intuitive and responsive
- **Minimalist design**: Clean, professional interface
- **Real-time feedback**: Visual processing status
- **Copy functionality**: Easy response sharing

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB for models
- **GPU**: Optional but recommended for faster inference

### Python Packages
  -**torch>=2.0.0
  -**transformers>=4.35.0
  -**sentence-transformers>=2.2.0
  -**faiss-cpu>=1.7.0
  -**gradio>=4.0.0
  -**pypdf2>=3.0.0
  -**pdfplumber>=0.10.0
  -**spacy>=3.7.0
  -**nltk>=3.8.0
  -**rank-bm25>=0.2.2
  -**langchain-text-splitters>=0.0.1
  -**accelerate>=0.24.0
  -**bitsandbytes>=0.41.0


## 🏗️ Processing Pipeline

### Document Processing Flow
1. **PDF Extraction** → Multi-method text extraction with metadata
2. **Document Analysis** → Structure, entities, sentiment analysis
3. **Intelligent Chunking** → Semantic + sentence-based segmentation
4. **Embedding Generation** → Dual-model embeddings for diversity
5. **Index Building** → FAISS + BM25 + entity indexes
6. **Hybrid Retrieval** → Combined semantic + lexical search
7. **Cross-Encoder Re-ranking** → Relevance optimization
8. **Context Construction** → Intelligent context assembly
9. **LLM Generation** → Mistral-7B with adaptive prompts
10. **Post-processing** → Answer refinement and formatting

### Architecture Diagram


## 🎯 Supported Question Types

| Type | Description | Examples |
|------|-------------|----------|
| **Factual** | Specific facts, numbers, dates | "What is the budget?" |
| **Analytical** | Analysis, causes, consequences | "Why did the project fail?" |
| **Comparative** | Comparisons, differences | "Compare the two approaches" |
| **Summarization** | Summaries, key points | "Summarize the conclusions" |
| **Extraction** | Lists, structured data | "List all participants" |
| **Evaluative** | Judgments, recommendations | "Evaluate the strategy" |

## ⚙️ Key Parameters

```python
# Chunking parameters
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 250      # Overlap between chunks
SENTENCE_CHUNK_SIZE = 384 # Token limit for sentence chunks

# Model configurations
EMBEDDING_MODELS = [
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/all-MiniLM-L6-v2'
]
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
LLM_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'
