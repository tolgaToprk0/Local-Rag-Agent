from openai import OpenAI

BASE_URL = "http://127.0.0.1:64435/v1"
CHAT_MODEL_NAME = "qwen3-4b-cuda-gpu:2"
EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b-cuda-gpu:1"


def get_client():
    return OpenAI(base_url=BASE_URL, api_key="not-needed")


def create_embeddings(client, metinler):
    if not metinler:
        return []

    cevap = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=metinler,
    )
    return [item.embedding for item in cevap.data]


def ask_question(client, baglam, soru):
    prompt = (
        "Aşağıdaki bağlamı kullanarak soruya kısa ve net Türkçe cevap ver. "
        "Eğer bağlamda cevap yoksa bunu açıkça söyle.\n\n"
        f"Bağlam:\n{baglam}\n\n"
        f"Soru: {soru}"
    )

    cevap = client.chat.completions.create(
        model=CHAT_MODEL_NAME,
        messages=[
            {"role": "system", "content": "Sen kısa, net ve Türkçe konuşuyorsun."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    yanit = cevap.choices[0].message.content.strip()
    if "</think>" in yanit:
        yanit = yanit.split("</think>", 1)[1].strip()
    return yanit
