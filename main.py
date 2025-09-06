from transformers import pipeline
from newsapi import NewsApiClient
from sentence_transformers import SentenceTransformer, util
import wikipediaapi
import requests
from bs4 import BeautifulSoup
import spacy

# --- Setup ---
api_key = ''
api = NewsApiClient(api_key=api_key)

# Zero-shot / bias classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
candidate_labels = ["left-wing", "right-wing", "neutral"]

# Contradiction/entailment classifier (NLI)
nli_model = pipeline("text-classification", model="roberta-large-mnli")

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyPoliticalNewsBot/1.0 (contact: Enter your Email)'
)

nlp = spacy.load("en_core_web_sm")

# --- Helper functions ---
def extract_entities(text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents if ent.label_ in ["PERSON","ORG","GPE","NORP"]]))

def get_wikipedia_text(topic):
    page = wiki.page(topic)
    return page.summary.split('. ') if page.exists() else []

def get_snopes_text(query):
    try:
        url = f"https://www.snopes.com/?s={query.replace(' ','+')}"
        r = requests.get(url, headers={'User-Agent':'MyPoliticalNewsBot/1.0'})
        soup = BeautifulSoup(r.text, 'html.parser')
        return [h.text.strip() for h in soup.find_all('h2', class_='entry-title')[:3]]
    except:
        return []

def get_factcheck_text(query):
    try:
        url = f"https://www.factcheck.org/?s={query.replace(' ','+')}"
        r = requests.get(url, headers={'User-Agent':'MyPoliticalNewsBot/1.0'})
        soup = BeautifulSoup(r.text, 'html.parser')
        return [h.text.strip() for h in soup.find_all('h3', class_='entry-title')[:3]]
    except:
        return []

def get_related_news(headline):
    try:
        related = api.get_everything(q=headline, language="en", sort_by="relevancy", page_size=3)
        return [a['title'] + ". " + (a['description'] or "") for a in related['articles']]
    except:
        return []

def compute_similarity_and_contradictions(claim, sources):
    claim_emb = embedding_model.encode(claim, convert_to_tensor=True)
    sims = []
    source_snippets = []
    contradictions = []

    for src in sources:
        for s in src:
            if s:
                s_emb = embedding_model.encode(s, convert_to_tensor=True)
                sim = util.cos_sim(claim_emb, s_emb).item()
                sims.append(sim)
                # NLI prediction
                nli_result = nli_model(f"{claim} </s></s> {s}")[0]
                label = nli_result['label']
                if label == "CONTRADICTION":
                    contradictions.append(s[:200])
                source_snippets.append((s, sim, label))
    
    avg_score = sum(sims)/len(sims) if sims else 0.0
    source_snippets.sort(key=lambda x: x[1], reverse=True)
    return avg_score, source_snippets[:3], contradictions

# --- Main pipeline ---
articles = api.get_everything(
    q="politics OR government OR election OR congress OR president",
    language="en",
    sort_by="relevancy",
    page_size=5
)
headlines = [a['title'] for a in articles['articles']] if articles['articles'] else []

for headline in headlines:
    print(f"\nHeadline: {headline}")

    # Bias classification
    bias_result = classifier(headline, candidate_labels)
    print("Bias classification:", bias_result)

    # Gather sources
    sources = []
    entities = extract_entities(headline) or ["United States government"]
    for e in entities:
        wiki_text = get_wikipedia_text(e)
        if wiki_text: sources.append(wiki_text)
    snopes_text = get_snopes_text(headline)
    if snopes_text: sources.append(snopes_text)
    factcheck_text = get_factcheck_text(headline)
    if factcheck_text: sources.append(factcheck_text)
    news_sources = get_related_news(headline)
    if news_sources: sources.append(news_sources)

    # Compute trust + contradictions
    trust_score, top_snippets, contradictions = compute_similarity_and_contradictions(headline, sources)
    print(f"Average claim-source similarity (trust score): {trust_score:.2f}")

    print("Top supporting snippets:")
    for s, sim, label in top_snippets:
        print(f"- ({sim:.2f}, {label}) {s[:200]}...")

    if contradictions:
        print("\n⚠️ Contradicting snippets detected:")
        for c in contradictions:
            print("-", c)