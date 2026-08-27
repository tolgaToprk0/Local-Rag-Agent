#LocalRAG

Bu proje, Foundry Local üzerinde çalışan bir RAG örneğidir. Belgeler okunur, parçalara ayrılır, embedding'ler çıkarılır ve SQLite veritabanında saklanır. Kullanıcı soru sorduğunda sorunun embedding'i oluşturulur ve cosine similarity ile en ilgili chunk bulunur. Bulunan chunk, Qwen modeli tarafından cevap üretmek için kullanılır.


#Kurulum

Sanal ortamı oluşturup gerekli paketleri yükleyin:

pip install -r requirements.txt

Foundry Local'ın bilgisayarınızda kurulu olması gerekir. Uygulama başlatıldığında gerekli modeller ve index otomatik olarak kontrol edilir.

#Kullanım


Proje klasöründe terminali açın.

Programı başlatmak için:

python -m src.main

Program açıldığında soru terminalden girilebilir.

Soru sormak için:

python -m src.main ask "Algoritma nedir?"

Index'i kontrol etmek ve gerekirse oluşturmak için:

python -m src.main index

#RAG akışı

1. Belge okunur.
2. Belge chunk'lara ayrılır.
3. Chunk'lar için embedding oluşturulur.
4. Embedding'ler SQLite'ta saklanır.
5. Kullanıcının sorusu için embedding oluşturulur.
6. Cosine similarity ile en ilgili chunk bulunur.
7. Bulunan chunk Qwen 3 4B modeline gönderilir.
8. Model cevabı üretir.

Foundry Local'ın OpenAI API'sine bağlanır. Embedding ve chat model çağrılarını yapar. </think> etiketini temizler.

Not: OpenAI Python istemcisi bulut OpenAI API'si için kullanılmıyor. Foundry Local'ın yerel API'sine http://127.0.0.1:64435/v1 adresinden bağlanmak için kullanılıyor.

#Kullanılan Araçlar

* Python
* Foundry Local
* Qwen 3 4B chat modeli
* Qwen 3 embedding 0.6B modeli
* SQLite
* NumPy
* pypdf
* OpenAI Python client