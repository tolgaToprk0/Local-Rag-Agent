import argparse

from src.llm import ask_question, get_client
from src.rag import format_context, search_question
from src.startup import prepare_runtime


def ask_flow(client, soru):
    bulunanlar = search_question(soru, client=client)
    baglam = format_context(bulunanlar)
    return ask_question(client, baglam, soru)


def main():
    parser = argparse.ArgumentParser(description="Local RAG uygulaması")
    parser.add_argument("command", nargs="?", choices=["index", "ask"], help="index = belge indisleme; ask = soru sor")
    parser.add_argument("value", nargs="?", help="Sorulacak soru")
    args = parser.parse_args()

    client = get_client()
    prepare_runtime(client)

    if args.command == "index":
        return

    soru = args.value or input("Sorunuz: ").strip()
    if not soru:
        print("Soru yazınız.")
        return

    cevap = ask_flow(client, soru)
    print(cevap)


if __name__ == "__main__":
    main()
