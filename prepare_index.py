from sentence_transformers import SentenceTransformer
import faiss
import os

# 1. Загрузка книги
with open("books/book1.txt", encoding="utf-8") as f:
    text = f.read()

# 2. Разбивка на чанки с overlap
def chunk_text(text, chunk_size=1000, overlap=200):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

chunks = chunk_text(text)

# 3. Вычисление эмбеддингов
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(chunks, convert_to_numpy=True)

# 4. Создание FAISS индекса
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. Сохранение индекса и чанков
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/books_index.faiss")
import pickle
with open("faiss_index/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS индекс готов!")
