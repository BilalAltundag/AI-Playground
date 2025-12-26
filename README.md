🧠 AI & LLM Learning Journey

Bu depo (repository), Yapay Zeka ve Büyük Dil Modelleri (LLM) üzerine yaptığım çalışmaları, teorik analizleri ve sıfırdan geliştirdiğim modelleri içerir.

Amacım, sadece hazır API'leri kullanmak değil, "kaputun altındaki" matematiği ve mimariyi (Backend) derinlemesine anlayarak özelleştirilmiş AI çözümleri üretmektir.

🚀 Proje 1: Baby GPT - Sıfırdan Transformer Eğitimi

Bu projede, modern LLM'lerin (GPT-4, Gemini, Llama) temelini oluşturan Transformer mimarisini PyTorch kullanarak sıfırdan inşa ettim ve eğittim.

🎯 Projenin Amacı

Hazır kütüphaneler (HuggingFace Trainer vb.) kullanmadan, ham PyTorch ile Self-Attention mekanizmasını kodlamak.

Tokenization, Embedding ve Positional Encoding süreçlerini manuel yönetmek.

Modeli bir diyalog veri seti ile eğiterek basit bir Chatbot haline getirmek.

🛠️ Kullanılan Teknolojiler

Core: Python, PyTorch (CUDA desteği ile)

Tokenizer: Tiktoken (OpenAI GPT-2 BPE)

Data: Hugging Face Datasets (knkarthick/dialogsum)

Deployment: Gradio (Web Arayüzü)

Visualization: Torchinfo, Matplotlib

📚 Teorik Altyapı ve Notlar

Bu projeyi geliştirirken üzerine çalıştığım temel kavramlar:

1. Neden RNN değil de Transformer?

Eskiden kullanılan RNN ve LSTM modelleri veriyi sırayla (seri) işliyordu. Bu durum unutkanlığa (uzun cümlelerin başını unutma) ve yavaşlığa (paralel işlem yapamama) yol açıyordu. Transformerlar ise Dikkat (Attention) mekanizması sayesinde cümlenin tamamına aynı anda odaklanabilir.

2. Self-Attention Mekanizması (Modelin Beyni)

Modelin kelimeler arasındaki ilişkiyi anlamasını sağlayan algoritmadır. Bunu bir veritabanı sorgusuna benzetebiliriz:

Query (Q - Sorgu): Token ne arıyor? (Örn: "Kedi" kelimesi bir eylem arıyor)

Key (K - Anahtar): Diğer kelimeler ne sunuyor? (Örn: "Yemek", "Uyumak")

Value (V - Değer): Eğer eşleşme olursa ne kadar bilgi aktarılacak?

Örnek: "Kedi mama yer" cümlesinde; Kedi (Query) ile Yer (Key) arasındaki matematiksel uyum (Dot Product) yüksek çıkar. Böylece model, kedinin beslendiğini anlar.

3. Mimariden Kesitler

Projede kullandığım Multi-Head Attention yapısının basitleştirilmiş mantığı:

class Head(nn.Module):
    def forward(self, x):
        # Q, K, V vektörlerini oluştur
        k = self.key(x)
        q = self.query(x)
        
        # Dikkat skorlarını hesapla (Matris Çarpımı)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        
        # Maskeleme (Geleceği görmeyi engelle - Decoder Only)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Değerleri birleştir
        v = self.value(x)
        return wei @ v


📊 Model Konfigürasyonu

T4 GPU sınırları dahilinde optimize edilmiş "Safe Pro" ayarları kullanılmıştır:

Parametre

Değer

Açıklama

Model Tipi

Decoder-only Transformer

GPT mimarisi

Parametre Sayısı

~10 Milyon

Custom "Baby" boyutu

Context Window

192 Token

Modelin hafıza derinliği

Embedding Size

384

Nöron katman genişliği

Layers (Derinlik)

6 Blok

Soyutlama seviyesi

Heads

6 Kafa

Paralel dikkat mekanizması

📉 Eğitim Sonuçları

Model, DialogSum veri seti üzerinde 5000 adım boyunca eğitilmiştir.

Başlangıç Loss: ~4.5

Bitiş Loss: ~0.5 (Model dil yapısını ve cevap verme mantığını çözdü)

(Buraya notebook'tan aldığın Loss grafiğini ekleyebilirsin)

💻 Nasıl Çalıştırılır?

Bu repoyu klonlayın:

git clone [https://github.com/kullaniciadi/AI-Learning-Journey.git](https://github.com/kullaniciadi/AI-Learning-Journey.git)


Gerekli kütüphaneleri kurun:

pip install torch tiktoken datasets gradio torchinfo tqdm


BabyGPT_Egitim.ipynb dosyasını Jupyter Lab veya Google Colab ile açıp çalıştırın.

Roadmap (Gelecek Hedefler)

[x] Sıfırdan Transformer Mimarisi (Baby GPT)

[ ] Büyük bir modelin (Llama-3) Fine-Tuning işlemi

[ ] RAG (Retrieval Augmented Generation) ile döküman tabanlı sohbet

[ ] Vision Transformer (ViT) ile görüntü işleme

Bu çalışma, AI mimarisini derinlemesine öğrenmek amacıyla oluşturulmuştur.
