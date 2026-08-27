import sqlite3
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from src.llm import create_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
DB_PATH = DATA_DIR / "rag.db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 1


def read_text_file(file_path):
    return file_path.read_text(encoding="utf-8")


def read_pdf_file(file_path):
    reader = PdfReader(str(file_path))
    metinler = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            metinler.append(text)
    return "\n\n".join(metinler)


def load_documents(documents_dir=DOCUMENTS_DIR):
    belgeler = []
    klasor = Path(documents_dir)
    if not klasor.exists():
        return belgeler

    for dosya in sorted(klasor.iterdir()):
        if dosya.is_dir():
            continue
        if dosya.suffix.lower() not in {".txt", ".pdf"}:
            continue

        if dosya.suffix.lower() == ".pdf":
            icerik = read_pdf_file(dosya)
        else:
            icerik = read_text_file(dosya)

        if icerik.strip():
            belgeler.append({"source": dosya.name, "content": icerik.strip()})

    return belgeler


def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    metin = text.strip()
    if not metin:
        return []

    kelimeler = metin.split()
    if len(kelimeler) <= chunk_size:
        return [metin]

    parcalar = []
    baslangic = 0
    adim = max(1, chunk_size - chunk_overlap)

    while baslangic < len(kelimeler):
        bitis = min(baslangic + chunk_size, len(kelimeler))
        parca = " ".join(kelimeler[baslangic:bitis]).strip()
        if parca:
            parcalar.append(parca)
        if bitis == len(kelimeler):
            break
        baslangic += adim

    return parcalar


def chunk_documents(documents):
    chunks = []
    for belge in documents:
        for parca in split_text(belge["content"]):
            chunks.append({"source": belge["source"], "text": parca})
    return chunks


def connect_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            text TEXT,
            embedding BLOB
        )
        """
    )
    return conn


def save_chunks(chunks, embeddings):
    conn = connect_db()
    try:
        conn.execute("DELETE FROM chunks")
        for chunk, embedding in zip(chunks, embeddings):
            vektor = np.asarray(embedding, dtype=np.float32)
            conn.execute(
                "INSERT INTO chunks (source, text, embedding) VALUES (?, ?, ?)",
                (chunk.get("source", ""), chunk.get("text", ""), vektor.tobytes()),
            )
        conn.commit()
    finally:
        conn.close()


def cosine_similarity(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def search_query(query_embedding, top_k=TOP_K_RESULTS):
    conn = connect_db()
    try:
        satirlar = conn.execute("SELECT source, text, embedding FROM chunks").fetchall()
    finally:
        conn.close()

    if not satirlar:
        return []

    skorlar = []
    for source, text, embedding_blob in satirlar:
        vektor = np.frombuffer(embedding_blob, dtype=np.float32)
        skor = cosine_similarity(query_embedding, vektor)
        skorlar.append({"source": source, "text": text, "score": round(skor, 4)})

    skorlar.sort(key=lambda item: item["score"], reverse=True)
    return skorlar[:top_k]


def build_index(documents_dir=DOCUMENTS_DIR, *, client):
    belgeler = load_documents(documents_dir)
    chunks = chunk_documents(belgeler)

    if not chunks:
        return 0

    metinler = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(client, metinler)
    save_chunks(chunks, embeddings)
    return len(chunks)


def format_context(results):
    if not results:
        return "İlgili belge bulunamadı."

    kisimlar = []
    for item in results:
        kisimlar.append(f"Kaynak: {item['source']}\n{item['text']}")
    return "\n\n".join(kisimlar)


def search_question(soru, *, client, top_k=TOP_K_RESULTS):
    query_embedding = create_embeddings(client, [soru])[0]
    return search_query(query_embedding, top_k=top_k)
