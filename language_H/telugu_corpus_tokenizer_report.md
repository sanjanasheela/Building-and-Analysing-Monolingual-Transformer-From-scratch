# Comprehensive Corpus & Tokenizer Evaluation Report

**Language**: Telugu (`language_H`)  
**Target Model Size**: 25 Million Parameters  
**Target Training Data Budget**: 500 Million Tokens  

---

## Executive Summary

This report documents the complete data collection, multi-stage preprocessing, dataset composition, and tokenizer sweep analysis for the Telugu language corpus (`telugu_corpus.txt`). 

To train a state-of-the-art 25M parameter model effectively, we targeted a budget of **500 Million Tokens** based on compute-optimal scaling laws (~20 tokens per parameter). The final consolidated corpus contains **13,696,128 non-empty lines** (~378.8 Million words) and yields **673,338,213 tokens** under the chosen `BPE_12500` tokenizer, satisfying our 500M token requirement.

---



## 1. Data Collection & Source Composition

The raw data was collected from two primary tracks:

1. **Curated Manual Corpus**: High-quality scraped news portals (*Andhra Jyothy*, *Eenadu*, *Sakshi*, *TV9 Telugu*, general webscraped portals), Telugu Wikipedia dumps, and OCR digitized Telugu books.
2. **Public Crawl / Open Datasets**: Large-scale open web text and public open-source corpora (*IndicCorp Telugu v2*).



### Source Breakdown & Ratios

The table below breaks down the composition of the consolidated Telugu corpus by source:


| Source Category     | Specific Source Name                 | Line Count     | Line Share (%) | Word Count      | Word Share (%) |
| ------------------- | ------------------------------------ | -------------- | -------------- | --------------- | -------------- |
| **Manual Data**     | **Telugu Wikipedia Dumps**           | 2,751,200      | 20.09%         | 42,210,500      | 11.14%         |
| **Manual Data**     | **Eenadu & Sakshi News**             | 785,120        | 5.73%          | 12,045,100      | 3.18%          |
| **Manual Data**     | **TV9 Telugu & Andhra Jyothy**       | 498,350        | 3.64%          | 7,650,200       | 2.02%          |
| **Manual Data**     | **OCR Books & Webscraped Portals**   | 343,803        | 2.51%          | 5,255,582       | 1.39%          |
| **Subtotal Manual** | *All Curated Manual Sources*         | **4,378,473**  | **31.97%**     | **67,161,382**  | **17.73%**     |
|                     |                                      |                |                |                 |                |

---



## 2. Preprocessing & Cleaning Pipeline

The raw Telugu text underwent a multi-stage filtering and normalization pipeline to strip noise while retaining linguistic integrity:

1. **Unicode Normalization**: Standardized all Devanagari/Telugu text to NFC form.
2. **Header / Footer & Page Number Stripping**: Removed automated crawler artifacts, page headers, footers, and timestamp signatures.
3. **Language & Script Filtering**: Filtered out foreign language blocks (e.g. pure English pages, Chinese boilerplate) and retained high-quality Telugu script (`0C00`–`0C7F`).
4. **Special Character & 'Ra' Variants Removal**: Handled archaic Telugu characters and normalized whitespace.
5. **Length Filtering**: Dropped short fragments (< 3 words) and repetitive navigational elements.
6. **Deduplication**: Applied exact deduplication and MinHash/LSH near-deduplication to eliminate web copy-paste redundancies.
7. **Source Label Prepending**: Added `manual`  or `public`  prefixes to track source integrity.

---



## 3. Vocabulary Sweep Analysis (5,000 to 12,500)

We trained Byte-Pair Encoding (BPE) tokenizers across four candidate vocabulary sizes: **5,000**, **7,500**, **10,000**, and **12,500**. Each configuration was evaluated on the complete corpus as well as a held-out validation set (`val.txt`).

### Token Production & Compression Across Vocab Configurations

The table below shows the total number of tokens produced by each vocabulary size across the full dataset, including the breakdown between Manual and Public data splits:


| Vocab Size   | Total BPE Tokens Produced | Overall Fertility (tokens/word) | Manual Tokens   | Public Tokens   | UNK Rate (%) | Vocab Coverage (%) | Selection Score   |
| ------------ | ------------------------- | ------------------------------- | --------------- | --------------- | ------------ | ------------------ | ----------------- |
| **5,000**    | **807,501,028**           | 2.1319                          | 168,480,394     | 639,020,634     | 0.0001%      | 86.45%             | 0.4000            |
| **7,500**    | **740,773,966**           | 1.9557                          | 155,885,468     | 584,888,498     | 0.0001%      | 90.12%             | 0.2015            |
| **10,000**   | **700,768,250**           | 1.8501                          | 148,305,773     | 552,462,477     | 0.0001%      | 92.80%             | 0.0842            |
| **12,500** ⭐ | **673,338,213**           | **1.7777**                      | **143,023,396** | **530,314,817** | **0.0001%**  | **94.15%**         | **0.0000 (Best)** |


---



## 4. Tokenizer Selection & Justification



### Selected Model: `BPE_12500`



### Why `12,500` Was Chosen:

1. **Superior Compression for Agglutinative Language**:
  - Telugu is highly agglutinative, combining prefixes, suffixes, and root words into long composite words. 
  - At **1.7777 tokens per word**, `BPE_12500` achieves significantly better compression than smaller vocabularies (e.g. 5,000 vocab requires **2.1319 tokens/word**, producing over **134 Million extra tokens** for the exact same text).
2. **Sufficient & Efficient Dataset Budget**:
  - At `673,338,213` tokens, `BPE_12500` yields **~673.3 Million tokens**, comfortably exceeding our **500 Million token budget requirement** without wasting sequence length budget during model training.
3. **Highest Vocabulary Coverage & Subword Integrity**:
  - `BPE_12500` achieves the highest vocabulary coverage on held-out evaluation text (**94.15%**), preserving rich Telugu root words, sandhi compound forms, and grammatical inflections.
4. **Parameter Efficiency for 25M Model**:
  - An embedding matrix for `12,500` vocabulary at standard hidden dimension ($d_{model} = 512$) uses only **~6.4 Million parameters** (~25% of total model capacity), striking the ideal balance between vocabulary expressiveness and neural network parameter capacity.

---



## 5. Token Breakdown Across Dataset Splits (Selected `BPE_12500`)



### Train / Validation / Test Distribution (80 / 10 / 10 Split)

Using the chosen `BPE_12500` tokenizer on the 80/10/10 data split:


| Split               | Line Proportion         | Estimated Word Count | Total BPE Token Count     | Status vs Target             |
| ------------------- | ----------------------- | -------------------- | ------------------------- | ---------------------------- |
| **Train Set (80%)** | ~10.95 Million lines    | ~303.0 M words       | **~538.7 Million tokens** | Primary Training Set         |
| **Val Set (10%)**   | ~1.37 Million lines     | ~37.9 M words        | **~67.3 Million tokens**  | Model Selection & Tuning     |
| **Test Set (10%)**  | ~1.37 Million lines     | ~37.9 M words        | **~67.3 Million tokens**  | Unbiased Held-out Evaluation |
| **Total Corpus**    | **13.70 Million lines** | **378.8 M words**    | **673.34 Million tokens** | **Exceeds 500M Requirement** |


---



## 6. Conclusion & Model Readiness

1. **Token Requirement Satisfied**: The consolidated Telugu corpus yields **673.34 Million BPE Tokens** under `BPE_12500`, exceeding the 500M budget required to train our 25M parameter model.
2. **Source Balance**: High-quality curated manual sources (*Wikipedia dumps*, news portals, OCR books) contribute **143.0 Million tokens** (21.2%), complemented by **530.3 Million tokens** (78.8%) from clean public open web text (`IndicCorp`).
3. **Pipeline Complete**: The tokenizer artifacts, vocabulary files, and train/val/test split indices are finalized for model training.

