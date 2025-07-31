from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import PyPDF2

def load_pdfs(folder):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith('.pdf'):
            with open(os.path.join(folder, filename), 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
                texts.append(text)
    return texts

def build_vector_store(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)
    return index, texts

def save_vector_store(index, texts, filename='vector_store.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((index, texts), f)

if __name__ == "__main__":
    docs = load_pdfs('docs')
    index, texts = build_vector_store(docs)
    save_vector_store(index, docs)
    print("Vector store saved.")