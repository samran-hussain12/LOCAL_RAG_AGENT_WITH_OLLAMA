import ollama
import pickle
from sentence_transformers import SentenceTransformer

def load_vector_store(filename='vector_store.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def find_similar(query, model, index, texts, top_k=2):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [texts[i] for i in I[0]]

def generate_answer(query, context, model='llama3'):
    prompt = f"""Use the context below to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

if __name__ == "__main__":
    query = input("Ask a question: ")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, texts = load_vector_store()
    context = " ".join(find_similar(query, model, index, texts))
    answer = generate_answer(query, context)
    print("Answer:", answer)