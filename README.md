# 🧭 Datagram MindCoach

**YKS öğrencilerine yönelik kişiselleştirilmiş yapay zeka çalışma koçu.**
Enneagram kişilik tipi + günlük ruh hali + zorlanılan konuyu birleştirip
kişiye özel **tavsiye**, **çalışma planı**, **örnek soru** ve **başarı analizi** üretir.

> Datagram — geliştirici TeknoLab grubunun adıdır. Uygulamanın adı **Datagram MindCoach**'tır.
> TeknoGenç / TeknoLAB dönem sonu projesi.

---

## ✨ Özellikler (Sekmeler)

| Sekme | Ne yapar |
|-------|----------|
| 🧭 **Kişilik Testi** | 27 soruluk (9 tip × 3) test, sorular **tek tek** gelir (wizard); İleri/Geri ile gezilir → radar grafik, en yüksek Enneagram tipi otomatik seçilir. |
| 💡 **Tavsiye** | Kişilik + ruh hali + konu → **hibrit** tavsiye (duyguya göre açılış + AI ile kişiselleştirilmiş tavsiye kısmı + sıcak kapanış) ve çalışma planı. |
| 📚 **Konu Çalış** | Konu özeti, zorluk seçilebilir örnek soru üretimi, detaylı çözüm, cevap değerlendirme. |
| 📈 **Ruh Hali Takibi** | Günlük ruh hali girişi (6'lı skala) → zaman içinde değişim grafiği. |
| 📊 **Ders Başarı Analizi** | Ders bazlı D/Y/B + **elle tarih girişi** → yüzde, başarı grubu, YKS neti, AI tavsiyesi + ruh hali ile **korelasyon** grafiği. Kayıtlar tarihe göre sıralı çizilir. |
| 💬 **Koçla Sohbet** | Açık uçlu sohbet (base model). |

---

## 🧠 Mimari

Çekirdek ilke: **içerik koddan (garantili) + Qwen3-8B model akıcılığı.**

- **Model:** `Qwen3-8B`, LoRA/Unsloth ile fine-tune edildi (A100, 4-bit, ChatML formatı, 489 temiz örnek).
- **Hibrit tavsiye:** Fine-tune model tek başına duyguyu kaybedip yalnızca teknik üretiyordu. Çözüm:
  **duygu açılış (koddan)** + **teknik orta (fine-tune model)** + **sıcak kapanış (koddan).**
- **Sekmeye göre adaptör:** `model_uret(messages, ..., adapter_kullan=True/False)`.
  Tavsiye → adaptör **açık** (fine-tune). 
  sohbet / değerlendirme → adaptör **kapalı** (base davranış).
  -Sohbet ve değerledirme kısmının base modelle çalışması programın ana amacının eğitilmiş modelle çalıştırıldığı gerçeğini değiştirmez fakat ilerleyen aşamalarda projenin farklı kısımlarında farklı yönlerden eğitilmiş modeller kullanmayı amaçlamaktayız.
- **Çeşitlilik:** Açılış/çalışma-tarzı/kapanış cümleleri havuzlardan `random.choice` ile seçilir (her kombinasyona birden çok varyant) → tekrar önlenir.
- **Çıktı güvenliği:** `latex_guvenli_sar()` markdown başlıklarını temizler, çıplak LaTeX'i `$...$` ile sarar, yarım `$` düzeltir, okunabilirlik için adımları ayrı satıra alır.
- **Sözel/sayısal ayrımı:** Sözel derslerde kavramsal soru (sayısal yasak), sayısal derslerde matematik mantığı.

---

## 🚀 Çalıştırma (Google Colab)

> GPU zorunlu. Sıra önemlidir.

1. **Bağımlılıkları kur** (`requirements.txt`). Unsloth için Colab'da:
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```
2. **Model yükleme hücresini çalıştır** — `model` ve `tokenizer` global olarak yüklenir.
   LoRA adaptör yolu: `/content/drive/MyDrive/MindCoach_Ozetler/mindcoach_qwen3_lora`
   (yol yoksa otomatik base modele düşer).
   ```python
   from unsloth import FastLanguageModel
   import torch, os
   max_seq_length = 1024
   LORA_YOLU = "/content/drive/MyDrive/MindCoach_Ozetler/mindcoach_qwen3_lora"
   if os.path.isdir(LORA_YOLU):
       model, tokenizer = FastLanguageModel.from_pretrained(
           model_name=LORA_YOLU, max_seq_length=max_seq_length, dtype=None, load_in_4bit=True)
   else:
       model, tokenizer = FastLanguageModel.from_pretrained(
           model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit", max_seq_length=max_seq_length, dtype=None, load_in_4bit=True)
   FastLanguageModel.for_inference(model)
   ```
3. **Konu özetleri (Konu Çalış sekmesi için):** Ders müfredat notu PDF'lerini
   `/content/drive/MyDrive/MindCoach_Ozetler` klasörüne koy. PDF başlıkları
   `### KONU: <ad>` formatında ve adlar `DERS_KONULARI`'ndaki adlarla birebir aynı olmalı.
   Uygulamada "özetleri yükle" ile parse edilir (pdfplumber).
4. **TÜM `datagram_mindcoach.py`'yi tek hücrede çalıştır.** Sonda `demo.launch(share=True)` ile public link açılır.

> ⚠️ Colab uyarısı: kodu parça parça farklı hücrelere yapıştırmak eski tanımları bellekte bırakır.
> Tekrar çalıştırırken `demo.launch()`'u durdurup tüm dosyayı tek hücrede çalıştır.

`model`/`tokenizer` yüklenmemişse uygulama çökmek yerine şablon yedeğe düşer ve uyarı gösterir.

### 🌐 Canlı Demo & Erişim

Model ağırlıkları (LoRA adaptörü) ve konu özeti PDF'leri geliştiricinin Google Drive'ında tutulur.
Uygulama, geliştiricinin kendi Colab ortamında Drive bağlıyken çalıştırılır ve `demo.launch(share=True)`
ile üretilen geçici public link üzerinden test edilebilir. Bu yapı, yönergedeki **"model ağırlıkları
sizin ortamınızda (lokal / Colab / Spaces) çalışıyor olmalıdır"** kuralına uygundur — çıkarım (inference)
tamamen kendi ortamımızda yapılır, hiçbir ticari API'ye devredilmez.

---

## 📚 Eğitim & Veri

- **Veri:** 489 ChatML örneği (`mindcoach_chatml.jsonl`), ders ders Gemini ile üretilip tarafımızca temizlendi.
- **Eğitim:** `Training.ipynb` (A100, LoRA/Unsloth).
- **Karşılaştırma:** Fine-tune model, base modeli açık ara geçti (base model, teknik cevaplardan çok uzak, duygusal ağırlıklı cevaplar veriyordu — kötü anlamda).

---

## ⚠️ Sınırlamalar & Şeffaflık

- 8B model, üretilen örnek soruların cevabını **her zaman doğru çözemez**. Soru kartlarında bu açıkça belirtilir
  ("yapay zeka üretimi, cevabı kendin de kontrol et").
- `sympy` ile otomatik doğrulama **denendi ama bırakıldı**: modelin çıktı formatları çok değişken olduğu için
  güvenilir doğrulama yapılamadı. Bu yüzden `requirements.txt`'te `sympy` yoktur.

---

## 🗂️ Dosya Yapısı

```
datagram_mindcoach.py    # Ana uygulama (Gradio, tüm sekmeler)
Datagram-MindCoach.ipynb # Uygulamanın Colab notebook hali
mindcoach_chatml.jsonl   # 489 eğitim örneği (ChatML)
Training.ipynb           # A100 fine-tune notebook'u (eğitim adımları)
training_loss.png        # Eğitim loss grafiği (kanıt)
requirements.txt         # Bağımlılıklar
README.md                # Bu dosya
ozetler/                 # Konu özeti PDF'leri (Konu Çalış sekmesini besler)
```
