![Uploading image.png…]()

# 🧠 AI & LLM Learning Journey

Bu depo, Yapay Zeka ve **Büyük Dil Modelleri (LLM)** mimarilerini derinlemesine anlamak amacıyla geliştirdiğim projeleri ve teknik notları içerir.  
Hazır API'lerin ötesine geçerek, modellerin çalışma mantığını (**backend / matematik**) seviyesinde **sıfırdan inşa etmeyi** hedefler.

---

## 🚀 Proje 1: Baby GPT – Sıfırdan Transformer Eğitimi

Bu proje, modern dil modellerinin (**GPT-4, LLaMA, Mistral** vb.) temelini oluşturan **Transformer mimarisinin**,  
**PyTorch kullanılarak sıfırdan kodlanmış ve eğitilmiş** bir versiyonudur.

Hazır *Trainer* kütüphaneleri kullanılmadan;

- Self-Attention mekanizması  
- Multi-Head Attention yapısı  
- Tokenization süreci  

manuel olarak inşa edilmiştir.

Model, diyalog verisi üzerinde eğitilerek **basit bir chatbot** fonksiyonu kazanmıştır.

▶️ **Projeyi Google Colab'de İncele ve Çalıştır**  
*(link eklenebilir)*

---

## 🛠️ Kullanılan Teknolojiler

- **Core:** Python, PyTorch (CUDA)
- **Tokenizer:** Tiktoken (OpenAI BPE)
- **Data:** Hugging Face – `knkarthick/dialogsum`
- **Deployment:** Gradio

---

## 📊 Model Özeti (Safe Pro Config)

*T4 GPU sınırları içinde optimize edilmiş model yapılandırması*

| Parametre | Değer | Açıklama |
|---------|------|---------|
| **Model Tipi** | Decoder-only Transformer | GPT mimarisi |
| **Parametre Sayısı** | ~10 Milyon | Custom boyutta eğitildi |
| **Context Window** | 192 Token | Hafıza derinliği |
| **Embedding Size** | 384 | Katman genişliği |
| **Layers / Heads** | 6 Blok / 6 Kafa | Derinlik ve paralellik |

---

## 📉 Sonuç

Model **5000 adım** boyunca eğitilmiş ve  
**CrossEntropyLoss** değeri **~4.5 → ~0.5** seviyesine düşürülmüştür.

Bu sonuç, modelin:

- İngilizce gramer yapısını  
- Temel diyalog mantığını  

başarıyla öğrendiğini göstermektedir.

---

## 🗺️ Roadmap (Gelecek Hedefler)

- [x] Sıfırdan Transformer Mimarisi (Baby GPT)
- [ ] Büyük bir modelin (LLaMA-3 / Mistral) **Fine-Tuning** işlemi
- [ ] **RAG (Retrieval Augmented Generation)** ile döküman tabanlı sohbet
- [ ] **Vision Transformer (ViT)** entegrasyonu
