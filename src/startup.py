import json
import sqlite3
import subprocess

from src.rag import DB_PATH, build_index

EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen3-4b"


def decode_cli_output(result):
    stdout = (result.stdout or b"").decode("utf-8", errors="replace")
    stderr = (result.stderr or b"").decode("utf-8", errors="replace")
    return stdout.strip(), stderr.strip()


def run_foundry_command(*args):
    try:
        result = subprocess.run(["foundry", *args], capture_output=True, check=False)
        stdout, stderr = decode_cli_output(result)
        return result.returncode, stdout, stderr
    except FileNotFoundError:
        return 1, "", "Foundry CLI bulunamadı."


def parse_json_output(text):
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}


def get_loaded_models():
    returncode, stdout, stderr = run_foundry_command("model", "list", "--loaded", "--output", "json")
    if returncode != 0:
        raise RuntimeError(f"Loaded model list alınamadı: {stderr or stdout}")

    data = parse_json_output(stdout)
    models = data.get("models", [])
    return [{"id": item.get("id", ""), "alias": item.get("alias", "")} for item in models if isinstance(item, dict)]


def service_is_ready(status_data):
    service = status_data.get("service", {})
    return bool(service.get("ready") or service.get("state", "").lower() == "ready")


def ensure_foundry_service():
    print("Foundry Local kontrol ediliyor...")
    returncode, stdout, stderr = run_foundry_command("status", "--output", "json")
    if returncode != 0:
        print("Foundry Local çalışmıyor; başlatılıyor...")
        returncode, stdout, stderr = run_foundry_command("server", "start")
        if returncode != 0:
            raise RuntimeError(f"Foundry Local başlatılamadı: {stderr or stdout}")

    data = parse_json_output(stdout)
    if not service_is_ready(data):
        print("Foundry Local hazır değil; yeniden denemek için servis başlatılıyor...")
        returncode, stdout, stderr = run_foundry_command("server", "start")
        if returncode != 0:
            raise RuntimeError(f"Foundry Local başlatılamadı: {stderr or stdout}")
        data = parse_json_output(stdout)

    if not service_is_ready(data):
        raise RuntimeError("Foundry Local hazır değil. Lütfen sunucuyu kontrol edin.")

    print("Foundry Local hazır.")


def is_model_loaded(model_alias, loaded_models):
    aliases = {item.get("alias", "") for item in loaded_models}
    ids = {item.get("id", "") for item in loaded_models}
    return model_alias in aliases or model_alias in ids


def ensure_model_loaded(model_alias):
    print(f"{model_alias} modeli kontrol ediliyor...")
    loaded_models = get_loaded_models()
    if is_model_loaded(model_alias, loaded_models):
        print(f"{model_alias} modeli zaten hazır.")
        return

    print(f"{model_alias} modeli yükleniyor...")
    returncode, stdout, stderr = run_foundry_command("model", "load", model_alias)
    if returncode != 0:
        raise RuntimeError(f"{model_alias} modeli yüklenemedi: {stderr or stdout}")

    print(f"{model_alias} modeli hazır.")


def has_index():
    if not DB_PATH.exists():
        return False

    conn = sqlite3.connect(str(DB_PATH))
    try:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if table_exists is None:
            return False

        row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return row_count > 0
    finally:
        conn.close()


def ensure_index(client):
    print("Index kontrol ediliyor...")
    if has_index():
        print("Index mevcut; yeniden oluşturulmadı.")
        return

    print("Index oluşturuluyor...")
    sayac = build_index(client=client)
    print(f"İndeks oluşturuldu: {sayac} adet chunk kaydedildi.")


def prepare_runtime(client):
    print("LocalRAG başlatılıyor...")
    ensure_foundry_service()
    ensure_model_loaded(EMBEDDING_MODEL)
    ensure_model_loaded(CHAT_MODEL)
    ensure_index(client)
    print("Hazır.")
