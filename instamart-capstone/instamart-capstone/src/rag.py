"""Part D tasks 1-3: chunking, TF-IDF retrieval, grounded answering (via mock_llm)."""
import glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join("data", "policies")
SYSTEM_PROMPT = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
                 "If the answer is not in the context, say you could not find the information.")

def chunk_words(text, size=40, overlap=10):
    """First chunk = words 0..size-1, next starts size-overlap later; stop after the chunk whose end reaches the last word."""
    words, step, chunks, start = text.split(), size - overlap, [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks

def load_chunks(size=40, overlap=10, folder=POLICY_DIR):
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for ch in chunk_words(text, size, overlap):
            out.append({"source": os.path.basename(path), "text": ch})
    return out

class Retriever:
    def __init__(self, size=40, overlap=10):
        self.chunks = load_chunks(size, overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]} for i in order]

DEFAULT = Retriever(40, 10)

def retrieve(query, k=3):
    return DEFAULT.retrieve(query, k)

def answer(question, retriever=None):
    hits = (retriever or DEFAULT).retrieve(question, 3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    r = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
    return r.choices[0].message.content, [h["source"] for h in hits]
