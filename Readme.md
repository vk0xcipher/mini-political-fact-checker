# Mini Political Fact-Checker (RAG + Bias Detection)

A lightweight **political news analysis tool** combining:  
- **Bias classification** (left/right/neutral)  
- **Mini RAG-style retrieval** for verifying claims against trusted sources  
- **Trust score computation** using embeddings  
- **Contradiction detection** using NLI (natural language inference)  

This project is perfect for **AI beginners** looking to showcase a practical, multi-source NLP pipeline.

---

## Features

1. **Political Bias Classification**  
   - Uses **Hugging Face zero-shot classification** to detect left-wing, right-wing, or neutral bias in headlines.  
   - No training required—pretrained model is used out-of-the-box.

2. **Claim Verification (Mini RAG)**  
   - Fetches related information from multiple sources:  
     - **Wikipedia** for general context  
     - **NewsAPI** for recent articles  
     - Optional scraping of fact-checking sites (Snopes, FactCheck)  
   - Uses **SentenceTransformers embeddings** to compute **semantic similarity** between the claim and source snippets.

3. **Contradiction Detection**  
   - Uses **NLI (Natural Language Inference) model** to detect contradictions between the claim and source text.  
   - Flags potentially misleading or false headlines.

4. **Trust Score**  
   - Computes average semantic similarity between claim and supporting sources.  
   - Roughly estimates how “supported” the claim is.

---

## 📦 Installation

1. **Clone the repo:**  
```bash
git clone https://github.com/yourusername/mini-political-fact-checker.git
cd mini-political-fact-checker
```

2. **Create a virtual environment:**  
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**  
```bash
pip install -r requirements.txt
```
Example `requirements.txt`:  
```
transformers
sentence-transformers
newsapi-python
wikipedia-api
requests
beautifulsoup4
spacy
torch
```

4. **Download SpaCy model:**  
```bash
python -m spacy download en_core_web_sm
```

5. **Set your NewsAPI key in the script:**  
```python
api_key = 'YOUR_NEWSAPI_KEY'
```

---

##  Usage

```bash
python main.py
```

**Sample Output:**  
```
Headline: U.S. Government Now ‘Controls’ 10% of Intel, Trump Says
Bias classification: right-wing
Average claim-source similarity (trust score): 0.29
Top supporting snippets:
- (0.90, NEUTRAL) U.S. Government Now ‘Controls’ 10% of Intel, Trump Says...
⚠️ Contradicting snippets detected:
- In foreign policy, Trump withdrew the U.S...
```

---

## 🧠 How it Works

1. **Fetch political headlines** from NewsAPI.  
2. **Classify bias** using Hugging Face zero-shot classifier.  
3. **Retrieve related sources** (Wikipedia, NewsAPI, optional fact-checkers).  
4. **Compute embeddings** to estimate similarity between claim and sources.  
5. **Run NLI model** to detect contradictions.  
6. **Output:** bias, trust score, top supporting snippets, and flagged contradictions.

---

## ⚡ Future Improvements

- Improve snippet retrieval to reduce noisy contradictions.  
- Add **source credibility weighting** for more accurate trust scores.  
- Implement **dashboard or CLI color coding** for easy visualization.  
- Expand fact-check sources to include **international news outlets**.

---

## 🔗 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [Sentence Transformers](https://www.sbert.net/)  
- [NewsAPI](https://newsapi.org/)  
- [Wikipedia API](https://wikipedia.readthedocs.io/)  
- [Natural Language Inference Models](https://huggingface.co/models?pipeline_tag=text-classification&search=mnli)  

---

## 💡 Notes

- This project is **a prototype**, not a definitive fact-checker.  
- Trust scores and contradiction detection are **rough indicators**, not verified verdicts.  
- Designed for educational purposes and **first GitHub AI project showcase**.