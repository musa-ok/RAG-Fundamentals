# 🚀 RAG-Study-Suite: Yapay Zeka ve Ajan Mimarileri Koleksiyonu

Bu depo, **RAG (Retrieval-Augmented Generation)** dünyasına girişten ileri seviye ajan yapılarına kadar uzanan kapsamlı bir çalışma koleksiyonudur. Her klasör, belirli bir konsepti veya mimariyi temsil eden bağımsız bir alt projedir.

---

## 📂 Proje İçerikleri

Bu koleksiyon şu ana başlıkları kapsamaktadır:

* **`CorrectiveRAGProject`**: LangGraph ile inşa edilen, hata payını minimize eden ve gerekirse internet araması yapan "Düzeltici RAG" mimarisi.
* **`AgentsProject`**: Karar verme yeteneğine sahip otonom yapay zeka ajanları.
* **`VectorStoreProject`**: Vektör veritabanı (ChromaDB, FAISS) yönetimi ve semantik arama optimizasyonları.
* **`FirstProject` & `RAGIntro`**: RAG dünyasına ilk adımlar ve temel akışlar.
* **`MessagingHistory`**: Sohbet geçmişi yönetimi ve bellek (memory) yapıları.

---

## 🛠️ Kurulum ve Bağımlılıklar

Bu depo bir koleksiyon olduğu için **her projenin kendi bağımlılıkları ilgili klasör içinde yer almaktadır.**

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/musa-ok/RAG-Fundamentals.git](https://github.com/musa-ok/RAG-Fundamentals.git)
    cd RAG-Fundamentals
    ```

2.  **İlgili Projeye Gidin:**
    Çalışmak istediğiniz klasörün içine girin:
    ```bash
    cd CorrectiveRAGProject
    ```

3.  **Bağımlılıkları Yükleyin:**
    Her klasördeki `requirements.txt` dosyasını kullanarak kurulumu yapın:
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Geliştirme Ortamı

* **İşlemci:** Apple M4 Silicon (Miniforge/Conda ile optimize edilmiştir).
* **Modeller:** Google Gemini 1.5/2.0/2.5 Flash, OpenAI.
* **Kütüphaneler:** LangChain, LangGraph, Pydantic, Tavily (Web Search).

---

## 📝 Notlar
Bu çalışmalarda temel amaç, LLM'lerin halüsinasyonlarını azaltmak ve veriye dayalı, güvenilir cevaplar üreten sistemler kurmaktır. Her klasör, bu hedefe giden farklı bir tekniği (Self-RAG, CRAG, Adaptive RAG vb.) temsil eder.

---
👨‍💻 **Musa Ok** tarafından eğitim ve gelişim amacıyla hazırlanmıştır.