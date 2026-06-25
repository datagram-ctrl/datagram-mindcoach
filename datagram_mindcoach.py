# -*- coding: utf-8 -*-
# =====================================================================
#  DATAGRAM MINDCOACH - Kapsamli YKS Calisma Kocu
#          + otomatik temizleme + sablon yedek
# =====================================================================
import re
import random
import os
import gradio as gr
import plotly.graph_objects as go
from datetime import datetime


# =====================================================================
#  ORTAK MODEL CAGRISI (Qwen3-8B)
#  cagiran fonksiyonlar except'e dusup sablon yedege gecer.
# =====================================================================
def model_uret(messages, max_new_tokens=320, temperature=0.6,
               repetition_penalty=1.15, adapter_kullan=True):
    """messages (system+user) alir, modelin urettigi ham metni dondurur.
    adapter_kullan=False ise LoRA adaptoru gecici devre disi birakilir (base davranis).
    Hata olursa Exception firlatir (cagiran taraf yakalar)."""
    import contextlib
    if (not adapter_kullan) and hasattr(model, "disable_adapter"):
        try:
            ctx = model.disable_adapter()
        except Exception:
            ctx = contextlib.nullcontext()
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        try:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False,
            ).to("cuda")
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        attention_mask = (inputs != pad_id).long()
        outputs = model.generate(
            input_ids=inputs, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, temperature=temperature, min_p=0.1,
            repetition_penalty=repetition_penalty, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(outputs)[0]
    if "<|im_start|>assistant" in decoded:
        decoded = decoded.split("<|im_start|>assistant")[-1]
    for bitis in ["<|im_end|>", "<|endoftext|>"]:
        if bitis in decoded:
            decoded = decoded.split(bitis)[0]
    decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL)
    return decoded.strip()





# =================================================================
# KONU TEKNIGI HAVUZU
# =================================================================
# -*- coding: utf-8 -*-
"""
KONU CALISMA TEKNIGI HAVUZU
Her ders-konu icin somut, konuya-ozel calisma ipucu.
Model bunu alip kisilik tonuyla yeniden yazacak.
Boylece KONU her zaman dogru olur (koddan geliyor).
"""

KONU_HAVUZU = {
    "Temel Kavramlar": "tek-cift sayi kurallarini ve ardisik sayilari tekrar et, birkac temel soru coz",
    "Sayı Basamakları": "basamak degeri ve cozumleme yontemini tekrar et, basamaklar arasi iliski sorulari coz",
    "Bölme ve Bölünebilme": "3, 9 ve 11 ile bolunebilme kurallarini tekrar et, kalan bulma sorulari coz",
    "EBOB-EKOK": "fayans dosseme ve periyodik tekrar problemlerini cozerek EBOB-EKOK ayrimini pekistir",
    "Rasyonel Sayılar": "payda esitleme ve kesirlerde dort islem sorulariyla pratik yap",
    "Mutlak Değer": "sayi dogrusunda sifira uzaklik mantigiyla mutlak deger denklemlerini coz",
    "Üslü Sayılar": "tabanlari esitleyerek uslu sayi denklemlerini coz, kurallari tekrar et",
    "Köklü Sayılar": "eslenik carpimi kuralini tekrar et, koklu ifadeleri sadelestir",
    "Çarpanlara Ayırma": "iki kare farki ve ortak parantez yontemleriyle ifadeleri carpanlarina ayir",
    "Oran-Orantı": "dogru ve ters oranti problemlerini cozerek mantigini pekistir",
    "Denklem Çözme": "bilinmeyeni yalniz birakma adimlarini tekrar et, birinci dereceden denklem sorulari coz",
    "Problemler": "sayi, yas ve isci problemlerinden secerek denklem kurma pratigi yap",
    "Kümeler": "venn semasi cizerek kesisim-birlesim sorularini coz",
    "Mantık": "onerme ve baglac (ve/veya) dogruluk tablolarini cikar",
    "Fonksiyonlar": "f(x) tanim-goruntu kumesi ve bileske fonksiyon sorulariyla calis",
    "Logaritma": "taban degistirme kuralini tekrar et, logaritmik denklemler coz",
    "Limit": "0/0 belirsizligi sorularini carpanlara ayirarak coz",
    "Türev": "us alma kuralini (turev formulunu) tekrar et, teget egimi sorulari coz",
    "İntegral": "alan hesabi sorulariyla integral alma kurallarini pekistir",
    "Permütasyon": "siralama (permutasyon) ve secme (kombinasyon) ayrimini ornek sorularla calis",
    "İstatistik": "ortalama, ortanca ve tepe deger hesaplarini tablo uzerinde tekrar et",
    "Olasılık": "zar, para ve torba problemleriyle olasilik mantigini pekistir",
    "Diziler": "aritmetik ve geometrik dizilerde genel terim bulma sorulari coz",

    "Açılar": "yondes, ters ve ic ters aci kurallarini (Z ve U kurali) tekrar et",
    "Üçgenler": "Pisagor ve benzerlik teoremleriyle ucgen sorulari coz",
    "Dörtgenler": "ozel dortgenlerin (paralelkenar, yamuk) alan formulleriyle calis",
    "Çember": "teget-kiris ve cevre aci kurallarini tekrar et, cember sorulari coz",
    "Analitik": "iki nokta arasi uzaklik ve egim formulleriyle analitik sorular coz",
    "Katı Cisimler": "silindir, koni ve kurenin hacim-alan formulleriyle calis",

    "Madde ve Özellikleri": "ozkutle (d=m/V) formuluyle birim cevirme sorulari coz",
    "Hareket ve Kuvvet": "ivme-zaman grafikleriyle hareket sorularini coz",
    "Enerji": "potansiyel ve kinetik enerji donusumu sorulariyla calis",
    "Isı ve Sıcaklık": "Q=m.c.deltaT formuluyle isi alisverisi sorulari coz",
    "Elektrostatik": "Coulomb kuvveti formuluyle yuk-cekim sorulari coz",
    "Optik": "cukur-tumsek ayna ve mercek goruntu ozelliklerini tekrar et",
    "Dalgalar": "dalga boyu-frekans (v=lambda.f) iliskisiyle sorular coz",
    "Basınç": "sivilarda ve katilarda basinc formulleriyle sorular coz",
    "Elektrik": "akim-gerilim-direnc (V=I.R) iliskisiyle devre sorulari coz",
    "Manyetizma": "sag el kuralini tekrar et, manyetik alan yonu sorulari coz",

    "Atom ve Periyodik Sistem": "proton-notron-elektron sayilarini ve periyodik ozellikleri tekrar et",
    "Periyodik Sistem": "atom capi ve iyonlasma enerjisi degisimlerini tekrar et",
    "Kimyasal Türler Arası Etkileşimler": "iyonik, kovalent ve hidrojen baglarini ayirt et",
    "Maddenin Halleri": "hal degisimi ve buhar basinci grafiklerini incele",
    "Karışımlar": "kutlece yuzde derisim formuluyle karisim sorulari coz",
    "Asit-Baz": "pH hesabi ve notrlesme tepkimeleriyle calis",
    "Mol Kavramı": "Avogadro sayisi ve mol-kutle iliskisiyle hesap sorulari coz",
    "Kimyasal Hesaplamalar": "mol-kutle-hacim oran orantisiyla tepkime hesaplari yap",

    "Hücre": "hucre organellerini (zarli-zarsiz) ve gorevlerini tablo halinde cikar",
    "Canlıların Sınıflandırılması": "alem-sube siralamasini ve ornekleri tekrar et",
    "Hücre Bölünmeleri": "mitoz ve mayoz evrelerini ve farklarini tekrar et",
    "Kalıtım": "Mendel caprazlamalarini (baskin-cekinik) Punnett karesiyle coz",
    "Ekosistem": "besin zinciri ve madde donguleriyle calis",
    "Sindirim": "sindirim enzimlerini ve gorev yerlerini tekrar et",
    "Dolaşım": "buyuk ve kucuk dolasim yolunu sema cizerek calis",
    "Solunum": "glikoliz ve Krebs dongusu evrelerini tekrar et",

    "Sözcükte Anlam": "gercek, yan ve mecaz anlam sorulariyla pratik yap",
    "Cümlede Anlam": "neden-sonuc ve amac-sonuc cumlelerini ayirt et",
    "Paragraf": "ana dusunce ve yardimci dusunce bulma sorulari coz",
    "Dil Bilgisi": "fiil cati (gecisli-gecissiz) ve sozcuk turlerini tekrar et",
    "Anlatım Bozukluğu": "ozne-yuklem uyumsuzlugu ve gereksiz sozcuk sorulari coz",
    "Noktalama-Yazım": "virgul, nokta kullanimi ve 'de/ki' yazimini tekrar et",
    "Ses Bilgisi": "unlu dusmesi ve unsuz yumusamasi kurallarini tekrar et",
    "Yazım Kuralları": "buyuk harf ve birlesik kelime yazim kurallarini tekrar et",
    "Cümle Ögeleri": "yuklem, ozne ve tumlecleri bulma sorulari coz",
    "Fiilimsiler": "isim-fiil, sifat-fiil ve zarf-fiil eklerini tekrar et",

    "İlk Türk Devletleri": "Gokturkler ve Hunlarin kurulus ve ozelliklerini tekrar et",
    "İslam Tarihi": "dort halife donemi ve Emeviler-Abbasiler ayrimini tekrar et",
    "Osmanlı Kuruluş-Yükseliş": "kurulus ve fetih donemlerini kronolojik tekrar et",
    "Kurtuluş Savaşı": "cepheler ve onemli muharebeleri kronolojik tekrar et",
    "İnkılaplar": "Ataturk ilke ve inkilaplarini konularina gore grupla",
    "İnkılap Tarihi": "Ataturk ilkelerini ve inkilaplari kronolojik tekrar et",
    "Osmanlı Yükselme Dönemi": "Fatih, Yavuz ve Kanuni donemi fetihlerini tekrar et",
    "Osmanlı Duraklama Dönemi": "duraklama nedenlerini ve onemli antlasmalari tekrar et",
    "İslam Öncesi Türk Tarihi": "Orta Asya Turk devletlerini ve Kavimler Gocu'nu tekrar et",
    "Türk İslam Devletleri": "Karahanlilar, Gazneliler ve Selcuklulari tekrar et",
    "Milli Mücadele Dönemi": "kongreler ve Milli Mucadele kronolojisini tekrar et",
    "Haçlı Seferleri": "Hacli Seferleri'nin nedenlerini ve sonuclarini tekrar et",
    "Çağdaş Türk ve Dünya Tarihi": "Soguk Savas donemi ve bloklari tekrar et",

    "Harita Bilgisi": "izohips (esyukselti) kurallariyla harita okuma sorulari coz",
    "İklim": "iklim grafiklerinde sicaklik-yagis egrilerini yorumla",
    "İklim Bilgisi": "iklim tiplerini ve ozelliklerini tablo halinde cikar",
    "İklim Tipleri": "Akdeniz, Karasal ve Okyanusal iklim ozelliklerini ayirt et",
    "Yer Şekilleri": "ic ve dis kuvvetlerin olusturdugu yer sekillerini tekrar et",
    "Nüfus": "nufus piramitlerini yorumlama sorulari coz",
    "Yerleşme": "kirsal ve kentsel yerlesme tiplerini tekrar et",
    "Türkiye Ekonomisi": "tarim, sanayi ve madencilik dagilimini harita uzerinde calis",
    "Nüfus ve Yerleşme": "nufus dagilimi ve yerlesme tiplerini harita uzerinde calis",
    "Dış Kuvvetler": "akarsu, ruzgar ve buzul asindirma sekillerini tekrar et",
    "Türkiye'nin Yeryüzü Şekilleri": "Turkiye'nin daglari, ovalari ve platolarini harita uzerinde calis",
    "Doğal Afetler": "deprem kusaklari ve fay hatlarini harita uzerinde isaretle",

    "Bilgi Felsefesi": "rasyonalizm ve empirizm ayrimini ornek filozoflarla tekrar et",
    "Varlık Felsefesi": "idealizm ve materyalizm goruslerini ayirt et",
    "Ahlak Felsefesi": "Kant'in odev ahlaki ve faydaci ahlak goruslerini tekrar et",
    "Sanat Felsefesi": "guzellik ve estetik kavramlarini farkli goruslerle calis",
    "Siyaset Felsefesi": "devletin kokeni ve iktidar goruslerini tekrar et",
    "Din Felsefesi": "teizm, deizm ve ateizm kavramlarini ayirt et",

    "AYT Matematik - Fonksiyonlar": "bileske ve ters fonksiyon sorulariyla calis",
    "AYT Matematik - Polinomlar": "polinomlarda bolme ve kalan bulma sorulari coz",
    "AYT Matematik - Trigonometri": "birim cember ve trigonometrik ozdeslikleri tekrar et",
    "AYT Matematik - Logaritma": "logaritmik denklem ve esitsizlik sorulari coz",
    "AYT Matematik - Diziler": "aritmetik-geometrik dizi ve seri toplam sorulari coz",
    "AYT Matematik - Limit-Süreklilik": "sagdan-soldan limit ve sureklilik sorulari coz",
    "AYT Matematik - Türev": "us alma kuralini tekrar et, maksimum-minimum sorulari coz",
    "AYT Matematik - İntegral": "belirli integral ile iki egri arasi alan sorulari coz",
    "AYT Matematik - Permütasyon-Kombinasyon": "siralama ve secme sorularini ayirt ederek coz",
    "AYT Matematik - Olasılık": "kosullu olasilik ve bagimsiz olay sorulari coz",
    "AYT Matematik - Mantık": "onerme ve dogruluk tablosu sorulari coz",

    "AYT Geometri - Çemberin Analitiği": "merkez-yaricap denklemiyle cember sorulari coz",
    "AYT Geometri - Analitik Geometri": "nokta-dogru uzakligi ve egim sorularini coz",
    "AYT Geometri - Vektörler": "vektorel toplama ve carpim sorulariyla calis",
    "AYT Geometri - Dönüşüm Geometrisi": "oteleme, yansima ve donme sorulari coz",

    "AYT Fizik - Vektörler": "vektorel toplama (uca ekleme) yontemiyle sorular coz",
    "AYT Fizik - Kuvvet-Tork-Denge": "tork ve denge kosulu sorulariyla calis",
    "AYT Fizik - Çembersel Hareket": "merkezcil kuvvet ve acisal hiz sorulari coz",
    "AYT Fizik - Basit Harmonik Hareket": "periyot formulu ve yay sarkaci sorulari coz",
    "AYT Fizik - Elektrik-Manyetizma": "sag el kurali ve manyetik kuvvet sorulari coz",
    "AYT Fizik - Modern Fizik": "fotoelektrik olay ve gorelilik konularini tekrar et",
    "AYT Fizik - Dalga Mekaniği": "girisim ve kirinim sorularini coz",

    "AYT Kimya - Modern Atom Teorisi": "kuantum sayilarini (n, l, ml, ms) tekrar et",
    "AYT Kimya - Gazlar": "ideal gaz denklemi (PV=nRT) sorulari coz",
    "AYT Kimya - Sıvı Çözeltiler": "koligatif ozellikler ve derisim sorulari coz",
    "AYT Kimya - Kimyasal Tepkimeler": "tepkime denklestirme ve mol hesabi yap",
    "AYT Kimya - Kimya ve Enerji": "Hess yasasi ve entalpi hesabi sorulari coz",
    "AYT Kimya - Tepkime Hızı": "tepkime hizi ve katalizor grafiklerini yorumla",
    "AYT Kimya - Kimyasal Denge": "Le Chatelier ilkesi ve Kc hesabi sorulari coz",
    "AYT Kimya - Asit-Baz Dengesi": "pH-pOH hesabi ve titrasyon sorulari coz",
    "AYT Kimya - Organik Kimya": "fonksiyonel gruplari ve IUPAC adlandirmayi tekrar et",

    "AYT Biyoloji - Sinir Sistemi": "impuls iletimi ve sinaps yapisini tekrar et",
    "AYT Biyoloji - Endokrin Sistem": "hormonlari ve salgilandiklari bezleri tekrar et",
    "AYT Biyoloji - Destek-Hareket": "kas kasilma (kayan iplikler) modelini tekrar et",
    "AYT Biyoloji - Genden Proteine": "transkripsiyon ve translasyon asamalarini tekrar et",
    "AYT Biyoloji - Fotosentez": "isiga bagimli ve Calvin dongusu evrelerini tekrar et",
    "AYT Biyoloji - Solunum": "glikoliz, Krebs ve ETS evrelerini tekrar et",
    "AYT Biyoloji - Komünite Ekolojisi": "tur iliskileri (rekabet, parazitlik) tekrar et",
    "AYT Biyoloji - Popülasyon Ekolojisi": "populasyon buyume egrilerini (S ve J tipi) yorumla",

    "AYT Edebiyat - Divan Edebiyatı": "gazel, kaside nazim bicimlerini ve aruz olcusunu tekrar et",
    "AYT Edebiyat - Halk Edebiyatı": "kosma, semai ve hece olcusunu tekrar et",
    "AYT Edebiyat - Tanzimat Edebiyatı": "Sinasi, Namik Kemal ve donem ozelliklerini tekrar et",
    "AYT Edebiyat - Servet-i Fünun": "Tevfik Fikret, Cenap Sahabettin ve donem ozelliklerini tekrar et",
    "AYT Edebiyat - Milli Edebiyat": "Yakup Kadri, Halide Edip ve donem ozelliklerini tekrar et",
    "AYT Edebiyat - Cumhuriyet Edebiyatı": "Garip ve Ikinci Yeni akimlarini tekrar et",
    "AYT Edebiyat - Şiir Bilgisi": "uyak, redif ve olcu turlerini tekrar et",
    "AYT Edebiyat - Edebi Sanatlar": "tesbih, istiare ve mecaz sanatlarini ornekle tekrar et",
    "AYT Edebiyat - Roman ve Hikaye": "donemlere gore roman ozelliklerini ve yazarlari tekrar et",

    "AYT Tarih-2 - Osmanlı Dağılma Dönemi": "dagilma donemi antlasmalarini ve kayiplari tekrar et",
    "AYT Tarih-2 - 20. Yüzyıl Başlarında Osmanlı": "Trablusgarp ve Balkan Savaslarini tekrar et",
    "AYT Tarih-2 - Atatürk İnkılapları": "inkilaplari konularina gore (siyasi, hukuki) grupla",
    "AYT Tarih-2 - İnkılaplar": "Ataturk ilke ve inkilaplarini kronolojik tekrar et",
    "AYT Tarih-2 - Çağdaş Türk Dünyası": "Soguk Savas ve bloklasmayi tekrar et",

    "AYT Coğrafya-2 - Çevre ve Toplum": "cevre sorunlari ve surdurulebilirlik konularini tekrar et",
    "AYT Coğrafya-2 - Küresel Ortam": "uluslararasi orgutleri ve bolgeleri tekrar et",
    "AYT Coğrafya-2 - Doğal Sistemler": "iklim tipleri ve biyomlari tekrar et",
    "AYT Coğrafya-2 - Beşeri Sistemler": "nufus, goc ve yerlesme konularini tekrar et",

    "Felsefe Grubu - Mantık": "klasik mantik (kiyas) ve sembolik mantik sorulari coz",
    "Felsefe Grubu - Psikoloji": "ogrenme, bellek ve gelisim konularini tekrar et",
    "Felsefe Grubu - Sosyoloji": "toplumsal kurumlar ve toplumsal degisme konularini tekrar et",
    "Felsefe Grubu - Felsefe": "bilgi, varlik ve ahlak felsefesi konularini tekrar et",

    "Din Kültürü": "inanc esaslari ve ibadet konularini tekrar et",
    "AYT Din Kültürü": "Islam dusuncesi ve yorumlari konularini tekrar et",
}


def _tr_sadelestir(s):
    """Turkce karakterleri ASCII'ye cevirir ve kucuk harf yapar (eslestirme icin)."""
    buyuk = {"İ":"i","I":"i","Ç":"c","Ş":"s","Ü":"u","Ö":"o","Ğ":"g","Â":"a","Î":"i","Û":"u"}
    for tr, asc in buyuk.items():
        s = s.replace(tr, asc)
    s = s.lower()
    kucuk = {"ç":"c","ş":"s","ı":"i","ü":"u","ö":"o","ğ":"g","â":"a","î":"i","û":"u","i̇":"i"}
    for tr, asc in kucuk.items():
        s = s.replace(tr, asc)
    return s.strip()


_SADE_HARITA = {_tr_sadelestir(k): v for k, v in KONU_HAVUZU.items()}
_SADE_SON_HARITA = {_tr_sadelestir(k.split(" - ")[-1]): v for k, v in KONU_HAVUZU.items()}


def konu_teknigi_bul(zayif_konu):
    """
    Kullanicinin girdigi konuya en uygun calisma teknigini bulur.
    Turkce karakter farkliliklarina dayaniklidir (Turev = Türev).
    """
    zayif_konu = zayif_konu.strip()
    sade = _tr_sadelestir(zayif_konu)

    if sade in _SADE_HARITA:
        return _SADE_HARITA[sade]

    if " - " in zayif_konu:
        konu_kismi_sade = _tr_sadelestir(zayif_konu.split(" - ")[-1])
        if konu_kismi_sade in _SADE_HARITA:
            return _SADE_HARITA[konu_kismi_sade]
        if konu_kismi_sade in _SADE_SON_HARITA:
            return _SADE_SON_HARITA[konu_kismi_sade]

    if sade in _SADE_SON_HARITA:
        return _SADE_SON_HARITA[sade]

    for anahtar_sade, teknik in _SADE_SON_HARITA.items():
        if anahtar_sade and (anahtar_sade in sade or sade in anahtar_sade):
            return teknik

    return f"{zayif_konu} konusunun temel kurallarini tekrar et ve bu konudan secme sorular coz"


# =================================================================
# ORNEK SORU HAVUZU
# =================================================================
# -*- coding: utf-8 -*-
"""
ORNEK SORU HAVUZU
En kritik TYT/AYT konularina dogrulanmis ornek sorular.
Her soru: {"soru":..., "cevap":..., "ipucu":...}
Model degil, kod bunlari ceker -> her zaman dogru.
"""

SORU_HAVUZU = {
    "Türev": [
        {"soru": "f(x) = x³ - 3x² + 2 fonksiyonunun türevi f'(x) nedir?",
         "cevap": "f'(x) = 3x² - 6x",
         "ipucu": "Her terimin üssünü öne al, üssü 1 azalt: (xⁿ)' = n·xⁿ⁻¹"},
        {"soru": "f(x) = 5x² + 4x - 7 fonksiyonunun x=1 noktasındaki türev değeri kaçtır?",
         "cevap": "f'(x) = 10x + 4 → f'(1) = 14",
         "ipucu": "Önce türevi al, sonra x=1 yerine koy."},
    ],
    "Limit": [
        {"soru": "lim(x→2) (x² - 4)/(x - 2) limitinin değeri nedir?",
         "cevap": "4",
         "ipucu": "0/0 belirsizliği var. Payı çarpanlara ayır: (x-2)(x+2)/(x-2) = x+2"},
    ],
    "Logaritma": [
        {"soru": "log₂8 değeri kaçtır?",
         "cevap": "3 (çünkü 2³ = 8)",
         "ipucu": "log₂8 = x ise 2ˣ = 8 olur. 8'i 2'nin üssü olarak yaz."},
        {"soru": "log₃27 + log₅25 işleminin sonucu kaçtır?",
         "cevap": "3 + 2 = 5",
         "ipucu": "27 = 3³ ve 25 = 5² olduğunu kullan."},
    ],
    "Üslü Sayılar": [
        {"soru": "2³ · 2⁴ işleminin sonucu kaçtır?",
         "cevap": "2⁷ = 128",
         "ipucu": "Tabanlar aynıysa üsler toplanır: aᵐ · aⁿ = aᵐ⁺ⁿ"},
    ],
    "Köklü Sayılar": [
        {"soru": "√12 + √27 işleminin en sade hali nedir?",
         "cevap": "2√3 + 3√3 = 5√3",
         "ipucu": "Kök içlerini sadeleştir: √12 = 2√3, √27 = 3√3"},
    ],
    "Çarpanlara Ayırma": [
        {"soru": "x² - 9 ifadesini çarpanlarına ayırın.",
         "cevap": "(x-3)(x+3)",
         "ipucu": "İki kare farkı: a² - b² = (a-b)(a+b)"},
    ],
    "Olasılık": [
        {"soru": "Bir zar atıldığında çift sayı gelme olasılığı kaçtır?",
         "cevap": "3/6 = 1/2",
         "ipucu": "Çift sayılar: 2, 4, 6 (3 tane). Toplam 6 durum var."},
    ],
    "Permütasyon": [
        {"soru": "A, B, C harfleri kaç farklı şekilde sıralanabilir?",
         "cevap": "3! = 6",
         "ipucu": "n farklı nesnenin sıralanışı n! ile bulunur."},
    ],

    "Üçgenler": [
        {"soru": "Dik kenarları 3 ve 4 olan dik üçgenin hipotenüsü kaçtır?",
         "cevap": "5",
         "ipucu": "Pisagor: a² + b² = c² → 9 + 16 = 25 → c = 5"},
    ],
    "Açılar": [
        {"soru": "Bir üçgenin iki açısı 50° ve 60° ise üçüncü açı kaç derecedir?",
         "cevap": "70°",
         "ipucu": "Üçgenin iç açıları toplamı 180°'dir."},
    ],

    "Hareket ve Kuvvet": [
        {"soru": "10 m/s hızla giden bir araç 5 saniyede kaç metre yol alır? (sabit hız)",
         "cevap": "50 m",
         "ipucu": "Yol = hız × zaman → 10 × 5"},
    ],
    "Optik": [
        {"soru": "Düz aynada 2 m uzaklıktaki bir cismin görüntüsü aynadan ne kadar uzaktadır?",
         "cevap": "2 m (ayna arkasında)",
         "ipucu": "Düz aynada görüntü, cisim ile ayna arasındaki uzaklık kadar arkadadır."},
    ],
    "Elektrik": [
        {"soru": "Üzerinden 2 A akım geçen ve 5 Ω direnci olan iletkenin uçları arasındaki gerilim kaç V'tur?",
         "cevap": "V = I·R = 2·5 = 10 V",
         "ipucu": "Ohm yasası: V = I × R"},
    ],

    "Mol Kavramı": [
        {"soru": "12 gram karbon (C) kaç moldür? (C'nin atom kütlesi = 12)",
         "cevap": "1 mol",
         "ipucu": "Mol = kütle / atom kütlesi → 12/12"},
    ],
    "Asit-Baz": [
        {"soru": "pH = 3 olan bir çözelti asidik mi bazik mi?",
         "cevap": "Asidik (pH < 7)",
         "ipucu": "pH < 7 asidik, pH = 7 nötr, pH > 7 baziktir."},
    ],

    "Kalıtım": [
        {"soru": "Aa × Aa çaprazlamasında oğul döllerin genotip oranı nedir?",
         "cevap": "1 AA : 2 Aa : 1 aa",
         "ipucu": "Punnett karesi çiz: A ve a'yı çaprazla."},
        {"soru": "Saf uzun (AA) bezelye ile saf kısa (aa) bezelye çaprazlanırsa F1 dölü nasıl olur?",
         "cevap": "Hepsi Aa (melez uzun)",
         "ipucu": "Her döl bir ebeveynden A, diğerinden a alır."},
    ],
    "Hücre": [
        {"soru": "Bitki hücresinde olup hayvan hücresinde olmayan iki yapı nedir?",
         "cevap": "Hücre duvarı ve kloroplast",
         "ipucu": "Fotosentez ve destek yapılarını düşün."},
    ],

    "Paragraf": [
        {"soru": "Bir paragrafın ana düşüncesini bulurken hangi soruyu sormalısın?",
         "cevap": "'Yazar bu metinde okuyucuya ne vermek/anlatmak istiyor?'",
         "ipucu": "Ana düşünce metnin tamamını kapsar, bir ayrıntı değildir."},
    ],
    "Sözcükte Anlam": [
        {"soru": "'Bu işin altından kalkamadı.' cümlesinde 'altından kalkmak' deyimi ne anlama gelir?",
         "cevap": "Bir işi başarmak/üstesinden gelmek",
         "ipucu": "Deyimler gerçek değil mecaz anlam taşır."},
    ],
    "Fiilimsiler": [
        {"soru": "'Koşarak geldi.' cümlesinde fiilimsi hangisidir ve türü nedir?",
         "cevap": "'Koşarak' - zarf-fiil (bağ-fiil)",
         "ipucu": "-arak/-erek eki zarf-fiil ekidir."},
    ],

    "Osmanlı Kuruluş-Yükseliş": [
        {"soru": "Osmanlı Devleti hangi yılda ve kim tarafından kurulmuştur?",
         "cevap": "1299, Osman Bey tarafından",
         "ipucu": "Söğüt-Domaniç bölgesini düşün."},
    ],
    "Kurtuluş Savaşı": [
        {"soru": "TBMM hangi tarihte açılmıştır?",
         "cevap": "23 Nisan 1920",
         "ipucu": "Ulusal Egemenlik ve Çocuk Bayramı ile aynı tarih."},
    ],

    "İklim": [
        {"soru": "Akdeniz ikliminde yazlar nasıldır?",
         "cevap": "Sıcak ve kurak",
         "ipucu": "Kışları ılık ve yağışlı, yazları ne olur?"},
    ],
    "Harita Bilgisi": [
        {"soru": "İzohips (eş yükselti) eğrileri sık geçiyorsa arazi nasıldır?",
         "cevap": "Eğimli/dik (yamaç dik)",
         "ipucu": "Eğriler sıksa yükseklik hızlı değişir."},
    ],

    "AYT Matematik - Türev": [
        {"soru": "f(x) = x² fonksiyonunun maksimum/minimum noktası nerede? (f'(x)=0)",
         "cevap": "x = 0 (minimum nokta)",
         "ipucu": "Türevi sıfıra eşitle: 2x = 0 → x = 0"},
    ],
    "AYT Matematik - Trigonometri": [
        {"soru": "sin30° + cos60° işleminin sonucu kaçtır?",
         "cevap": "1/2 + 1/2 = 1",
         "ipucu": "sin30° = 1/2, cos60° = 1/2"},
    ],

    "AYT Fizik - Modern Fizik": [
        {"soru": "Fotoelektrik olayda ışık hangi özelliğiyle elektron koparır?",
         "cevap": "Frekansı (enerjisi) ile - E = h·f",
         "ipucu": "Işığın tanecik (foton) modelini düşün."},
    ],

    "AYT Kimya - Organik Kimya": [
        {"soru": "CH₄ (metan) molekülündeki bağ türü nedir?",
         "cevap": "Kovalent bağ (apolar)",
         "ipucu": "Ametal-ametal bağı kovalenttir."},
    ],

    "AYT Biyoloji - Fotosentez": [
        {"soru": "Fotosentezin ışığa bağlı reaksiyonları kloroplastın neresinde gerçekleşir?",
         "cevap": "Tilakoid (granum) zarlarında",
         "ipucu": "Calvin döngüsü stromada, ışık reaksiyonları nerede?"},
    ],

    "AYT Edebiyat - Divan Edebiyatı": [
        {"soru": "Divan edebiyatında iki dizelik en küçük nazım birimi nedir?",
         "cevap": "Beyit",
         "ipucu": "Gazel ve kasidenin yapı taşıdır."},
    ],
}


def _tr_sade(s):
    buyuk = {"İ":"i","I":"i","Ç":"c","Ş":"s","Ü":"u","Ö":"o","Ğ":"g","Â":"a","Î":"i","Û":"u"}
    for tr, asc in buyuk.items():
        s = s.replace(tr, asc)
    s = s.lower()
    kucuk = {"ç":"c","ş":"s","ı":"i","ü":"u","ö":"o","ğ":"g","â":"a","î":"i","û":"u","i̇":"i"}
    for tr, asc in kucuk.items():
        s = s.replace(tr, asc)
    return s.strip()


_SORU_SADE = {_tr_sade(k): v for k, v in SORU_HAVUZU.items()}
_SORU_SADE_SON = {_tr_sade(k.split(" - ")[-1]): v for k, v in SORU_HAVUZU.items()}


def ornek_soru_bul(zayif_konu):
    """Konuya ait ornek sorulari dondurur. Yoksa bos liste."""
    sade = _tr_sade(zayif_konu.strip())
    if sade in _SORU_SADE:
        return _SORU_SADE[sade]
    if " - " in zayif_konu:
        son = _tr_sade(zayif_konu.split(" - ")[-1])
        if son in _SORU_SADE:
            return _SORU_SADE[son]
        if son in _SORU_SADE_SON:
            return _SORU_SADE_SON[son]
    if sade in _SORU_SADE_SON:
        return _SORU_SADE_SON[sade]
    return []


def _konunun_dersi(zayif_konu):
    """Konu adindan ait oldugu dersi bulur (sozel/sayisal ayrimi icin).
    Once tam eslesme aranir (ayni konu adi birden fazla derste olabilir,
    or. 'Mantık' hem Matematik hem Felsefe), sonra son-parca eslesmesine dusulur."""
    z = (zayif_konu or "").strip()
    for ders, konular in DERS_KONULARI.items():
        if z in konular:
            return ders
    z_son = z.split(" - ")[-1].strip()
    for ders, konular in DERS_KONULARI.items():
        for k in konular:
            if k.split(" - ")[-1].strip() == z_son:
                return ders
    return z


def _model_soru_uret(zayif_konu, adet=3, zorluk="Orta", kacinilacak=None):
    """Modelden konuya ozel ornek sorular uretir. Her soru icin soru/ipucu/cevap.
    kacinilacak: daha once uretilmis sorularin metinleri (bunlardan farkli uretsin).
    Basarisiz olursa None doner (cagiran SORU_HAVUZU yedegine duser)."""
    zorluk_tarif = {
        "Kolay": "kolay seviye, temel kavramları ölçen, tek adımlı",
        "Orta": "orta seviye, YKS standart zorluğunda",
        "Zor": "zor seviye, çok adımlı, çeldiricili, ileri düzey",
    }.get(zorluk, "orta seviye")

    kacinma_notu = ""
    if kacinilacak:
        ornekler = "; ".join(s[:50] for s in kacinilacak[-6:])
        kacinma_notu = (
            f"\n\nÇOK ÖNEMLİ: Aşağıdaki sorular DAHA ÖNCE soruldu, bunlardan "
            f"TAMAMEN FARKLI, yeni sayılar ve farklı kurgular kullan:\n{ornekler}\n"
            f"Aynı sayıları veya aynı kalıbı TEKRARLAMA."
        )

    sozel_dersler = ["Biyoloji", "Tarih", "Coğrafya", "Felsefe", "Türkçe", "Edebiyat",
                     "Din", "Psikoloji", "Sosyoloji", "Mantık", "Sosyal"]
    sayisal_dersler = ["Matematik", "Fizik", "Kimya", "Geometri"]
    ders_adi = _konunun_dersi(zayif_konu)
    is_sozel = any(s in ders_adi for s in sozel_dersler) and not any(s in ders_adi for s in sayisal_dersler)

    if is_sozel:
        prompt = (
            f"Sen deneyimli bir YKS öğretmenisin. '{zayif_konu}' konusunda "
            f"TAM {adet} adet KAVRAMSAL örnek soru hazırla. Sorular {zorluk_tarif} olsun.\n\n"
            f"ÇOK ÖNEMLİ KURALLAR:\n"
            f"- Bu bir SÖZEL/KAVRAMSAL derstir. ASLA matematiksel/sayısal problem sorma! "
            f"'Yüzde kaçı', 'kaç tane', 'toplam sayı' gibi hesap soruları YASAK.\n"
            f"- Gerçek YKS tarzında, KAVRAM, SÜREÇ, İLİŞKİ, SEBEP-SONUÇ ölçen sorular sor. "
            f"Örneğin bir olayın nedenini, bir kavramın tanımını, bir sürecin nasıl işlediğini sor.\n"
            f"- Şekil, grafik, tablo gerektiren soru SORMA (görsel yok).\n"
            f"- Her soru farklı bir alt konuyu/kavramı ölçsün, çeşitlilik önemli.{kacinma_notu}\n\n"
            f"Cevabını SADECE şu formatta ver, başka hiçbir şey yazma:\n"
            f"SORU: <kavramsal soru metni>\n"
            f"IPUCU: <kısa ipucu>\n"
            f"CEVAP: <net cevap>\n"
            f"===\n"
            f"(her soru için bu üç satırı tekrarla, her soruyu === ile ayır)"
        )
        sistem = "Sen YKS'nin sözel/kavramsal derslerinde uzman bir öğretmensin. Kavram, süreç ve ilişki ölçen sorular sorarsın; asla sayısal hesap problemi sormazsın. Her seferinde farklı ve özgün sorular üretirsin."
    else:
        prompt = (
            f"Sen bir YKS/LGS matematik-fen ogretmenisin. '{zayif_konu}' konusunda "
            f"TAM {adet} adet ornek soru hazirla. Sorular {zorluk_tarif} olsun.\n\n"
            f"ONEMLI KURALLAR:\n"
            f"- Sorular SADECE metin ve sayilarla cozulebilsin. SEKIL, GRAFIK, DEVRE, "
            f"TABLO veya GORSEL gerektiren soru SORMA. 'Yukaridaki sekilde', 'verilen "
            f"grafikte' gibi ifadeler KULLANMA cunku gorsel yok.\n"
            f"- Her soruda FARKLI sayilar kullan, cesitlilik onemli.\n"
            f"- Matematiksel ifadeleri LaTeX ile yaz: us icin $x^2$, "
            f"kesir icin $\\frac{{a}}{{b}}$, kok icin $\\sqrt{{x}}$ kullan. "
            f"Ornek: '$x^2 - 9$ ifadesini carpanlarina ayir'.{kacinma_notu}\n\n"
            f"Cevabini SADECE su formatta ver, baska hicbir sey yazma:\n"
            f"SORU: <soru metni>\n"
            f"IPUCU: <kisa ipucu>\n"
            f"CEVAP: <net cevap>\n"
            f"===\n"
            f"(her soru icin bu uc satiri tekrarla, her soruyu === ile ayir)"
        )
        sistem = "Sen dogru ve net YKS sorulari hazirlayan bir ogretmensin. Cevaplarin matematiksel olarak dogru olmali. Asla gorsel/sekil gerektiren soru sormezsin. Her seferinde farkli ve ozgun sorular uretirsin."

    messages = [
        {"role": "system", "content": sistem},
        {"role": "user", "content": prompt},
    ]
    try:
        token_limit = min(1800, 350 + adet * 230)
        ham = model_uret(messages, max_new_tokens=token_limit, temperature=0.9,
                         repetition_penalty=1.15, adapter_kullan=False)
    except Exception:
        return None

    kacinilacak_set = set((kacinilacak or []))
    sorular = []
    bloklar = re.split(r"\n={2,}\n|\n-{3,}\n|\n={2,}|={2,}\n", ham)
    for blok in bloklar:
        soru_m = re.search(r"SORU:\s*(.+?)(?=\n*(?:IPUCU|CEVAP):|$)", blok, re.DOTALL | re.IGNORECASE)
        ipucu_m = re.search(r"IPUCU:\s*(.+?)(?=\n*(?:SORU|CEVAP):|$)", blok, re.DOTALL | re.IGNORECASE)
        cevap_m = re.search(r"CEVAP:\s*(.+?)(?=\n*(?:SORU|IPUCU):|$)", blok, re.DOTALL | re.IGNORECASE)
        if soru_m and cevap_m:
            soru_metni = re.sub(r"\s+", " ", soru_m.group(1)).strip()
            if re.search(r"(yukar[ıi]daki|a[şs]a[ğg][ıi]daki|verilen|[şs]ekilde|grafik|devre|tablo)", soru_metni, re.IGNORECASE):
                continue
            if soru_metni in kacinilacak_set:
                continue
            sorular.append({
                "soru": soru_metni,
                "ipucu": re.sub(r"\s+", " ", ipucu_m.group(1)).strip() if ipucu_m else "Adım adım düşün.",
                "cevap": re.sub(r"\s+", " ", cevap_m.group(1)).strip(),
            })
    return sorular if sorular else None


def _sorulari_karta_cevir(zayif_konu, sorular, kaynak_notu):
    if not sorular:
        return f"### 📝 {zayif_konu}\n\n_Bu konu için şu an soru üretemedim, lütfen tekrar dene veya başka bir konu seç._"
    kart = f"### 📝 {zayif_konu} — Örnek Sorular ({len(sorular)} soru)\n{kaynak_notu}\n\n"
    if "Yapay zeka" in kaynak_notu:
        kart += (
            "> ℹ️ *Bu sorular yapay zeka tarafından üretildi. Cevaplar çoğunlukla doğrudur "
            "ama nadiren hata olabilir; cevabı kendin de kontrol etmeni öneririm.*\n\n"
        )
    for i, s in enumerate(sorular, 1):
        soru_m = latex_guvenli_sar(s['soru'])
        ipucu_m = latex_guvenli_sar(s['ipucu'])
        cevap_m = latex_guvenli_sar(s['cevap'])
        kart += f"**Soru {i}.** {soru_m}\n\n"
        kart += f"<details><summary>💡 İpucu</summary>\n\n{ipucu_m}\n\n</details>\n\n"
        kart += f"<details><summary>✅ Cevap</summary>\n\n**{cevap_m}**\n\n</details>\n\n---\n\n"
    return kart


def ornek_soru_uret_buton(zayif_konu, adet=3, zorluk="Orta"):
    """'Soru Üret' butonu: modelden taze soru dener, olmazsa SORU_HAVUZU yedek.
    Donus: (kart_markdown, soru_listesi_state, daha_fazla_butonu_gorunurlugu)"""
    if not zayif_konu or not zayif_konu.strip():
        return "Önce bir konu seç.", [], gr.update(visible=False)

    try:
        adet = int(adet)
    except (ValueError, TypeError):
        adet = 3
    adet = max(1, min(10, adet))

    model_sorulari = _model_soru_uret(zayif_konu, adet=adet, zorluk=zorluk)
    if model_sorulari and len(model_sorulari) < adet:
        eksik = adet - len(model_sorulari)
        mevcut_metinler = [s["soru"] for s in model_sorulari]
        ek = _model_soru_uret(zayif_konu, adet=eksik, zorluk=zorluk,
                              kacinilacak=mevcut_metinler)
        if ek:
            for s in ek:
                if s["soru"] not in mevcut_metinler:
                    model_sorulari.append(s)
                    mevcut_metinler.append(s["soru"])
                if len(model_sorulari) >= adet:
                    break

    if model_sorulari:
        kaynak_notu = f"_(Yapay zeka tarafından üretildi · {zorluk} zorluk)_"
        sorular = model_sorulari
    else:
        sorular = ornek_soru_bul(zayif_konu)
        kaynak_notu = "_(Doğrulanmış soru havuzundan — model şu an üretemedi)_"

    gorsel_uyari = ""
    if _konu_gorsel_yogun_mu(zayif_konu):
        gorsel_uyari = (
            "> ⚠️ **Not:** Bu konu genelde şekil/çizim gerektirir. Yapay zeka görsel "
            "üretemediği için yalnızca **şekilsiz çözülebilen** (açı/alan hesabı, formül "
            "uygulama gibi) sorular gösterilir. Şekilli sorular için kitabını kullan.\n\n"
        )

    kart = gorsel_uyari + _sorulari_karta_cevir(zayif_konu, sorular, kaynak_notu)
    gorunur = gr.update(visible=bool(sorular))
    return kart, sorular, gorunur


def detayli_cozum_uret(secili_soru_label, sorular):
    """Secili sorunun adim adim cozumunu modelden uretir."""
    sorular = sorular or []
    if not sorular:
        return "Önce soru üret, sonra detaylı çözüm iste."
    if not secili_soru_label:
        return "Lütfen yukarıdan bir soru seç."

    m = re.search(r"(\d+)", secili_soru_label)
    if not m:
        return "Soru seçimi anlaşılamadı."
    idx = int(m.group(1)) - 1
    if idx < 0 or idx >= len(sorular):
        return "Seçilen soru bulunamadı."

    s = sorular[idx]
    try:
        prompt = (
            f"Aşağıdaki YKS sorusunu bir öğrenciye ADIM ADIM, çok anlaşılır şekilde çöz.\n\n"
            f"SORU: {s['soru']}\n"
            f"(Doğru cevap: {s['cevap']})\n\n"
            f"ÇÖZÜMÜ ŞÖYLE YAZ (her bölümü AYRI SATIRDA, aralarına boş satır koy):\n"
            f"1. Önce sorunun ne istediğini bir cümleyle açıkla\n"
            f"2. Hangi kural/formülü kullanacağını söyle\n"
            f"3. Adım adım işlemi göster (her adımı 'Adım 1:', 'Adım 2:' diye numarala ve ayrı satıra yaz)\n"
            f"4. Sonucu 'Sonuç:' diye net belirt\n\n"
            f"Matematiksel ifadeleri $ işaretleri arasında LaTeX ile yaz (örn: $x^2$, $\\frac{{1}}{{2}}$). "
            f"Başlık işareti (###) veya markdown başlığı KULLANMA. "
            f"Her adımı kısa tut ve ayrı satıra yaz; tek bir uzun paragraf YAZMA. "
            f"Öğretici ve sıcak bir dille, lise öğrencisinin anlayacağı şekilde anlat."
        )
        messages = [
            {"role": "system", "content": "Sen sabırlı, açıklayıcı bir YKS öğretmenisin. Soruları adım adım, öğrencinin anlayacağı şekilde çözersin."},
            {"role": "user", "content": prompt},
        ]
        ham = model_uret(messages, max_new_tokens=850, temperature=0.5,
                         repetition_penalty=1.1, adapter_kullan=False)
        ham = ham.strip()
        if len(ham) >= 30:
            ham = _yarim_cumleyi_at(ham)
            ham = latex_guvenli_sar(ham)
            return f"### 🔍 Soru {idx+1} — Detaylı Çözüm\n\n**Soru:** {s['soru']}\n\n---\n\n{ham}"
    except Exception:
        pass
    return (
        f"### 🔍 Soru {idx+1} — Çözüm\n\n**Soru:** {s['soru']}\n\n"
        f"**İpucu:** {s['ipucu']}\n\n**Cevap:** {s['cevap']}\n\n"
        f"_(Detaylı adım adım çözüm şu an üretilemedi, tekrar dene.)_"
    )


def soru_secenekleri_guncelle(sorular):
    """Soru uretildikten sonra 'detaylandir' dropdown'unu gunceller."""
    sorular = sorular or []
    secenekler = [f"Soru {i+1}" for i in range(len(sorular))]
    return gr.update(choices=secenekler, value=secenekler[0] if secenekler else None)


def daha_fazla_soru_uret(zayif_konu, adet, zorluk, mevcut_sorular):
    """'Daha Fazla Soru Üret': yeni sorulari mevcutlara EKLER (uzerine yazmaz)."""
    mevcut_sorular = mevcut_sorular or []
    if not zayif_konu or not zayif_konu.strip():
        return _sorulari_karta_cevir(zayif_konu or "", mevcut_sorular, ""), mevcut_sorular, gr.update(visible=bool(mevcut_sorular))

    if len(mevcut_sorular) >= 20:
        kart = _sorulari_karta_cevir(zayif_konu, mevcut_sorular,
            "_(En fazla 20 soru gösterilebilir. Yeni konu için yukarıdan tekrar üret.)_")
        return kart, mevcut_sorular, gr.update(visible=True)

    try:
        adet = int(adet)
    except (ValueError, TypeError):
        adet = 3
    adet = max(1, min(10, adet))

    yeni = _model_soru_uret(zayif_konu, adet=adet, zorluk=zorluk,
                            kacinilacak=[s["soru"] for s in mevcut_sorular])
    if yeni:
        kaynak_notu = f"_(Yapay zeka tarafından üretildi · {zorluk} zorluk)_"
        birlesik = mevcut_sorular + yeni
    else:
        kaynak_notu = "_(Yeni soru üretilemedi, mevcut sorular korundu — tekrar dene)_"
        birlesik = mevcut_sorular

    kart = _sorulari_karta_cevir(zayif_konu, birlesik, kaynak_notu)
    return kart, birlesik, gr.update(visible=True)



# =====================================================================
#  DERS -> KONU HARITASI
# =====================================================================
DERS_KONULARI = {
    "TYT Matematik": ["Temel Kavramlar","Sayı Basamakları","Bölme ve Bölünebilme","EBOB-EKOK","Rasyonel Sayılar","Mutlak Değer","Üslü Sayılar","Köklü Sayılar","Çarpanlara Ayırma","Oran-Orantı","Denklem Çözme","Problemler","Kümeler","Mantık","Fonksiyonlar","Permütasyon","Olasılık","İstatistik","Diziler"],
    "TYT Geometri": ["Açılar","Üçgenler","Dörtgenler","Çember","Analitik","Katı Cisimler"],
    "TYT Fizik": ["Madde ve Özellikleri","Hareket ve Kuvvet","Enerji","Isı ve Sıcaklık","Elektrostatik","Optik","Dalgalar","Basınç","Elektrik","Manyetizma"],
    "TYT Kimya": ["Atom ve Periyodik Sistem","Periyodik Sistem","Kimyasal Türler Arası Etkileşimler","Maddenin Halleri","Karışımlar","Asit-Baz","Mol Kavramı","Kimyasal Hesaplamalar"],
    "TYT Biyoloji": ["Hücre","Canlıların Sınıflandırılması","Hücre Bölünmeleri","Kalıtım","Ekosistem","Sindirim","Dolaşım","Solunum"],
    "TYT Türkçe": ["Sözcükte Anlam","Cümlede Anlam","Paragraf","Dil Bilgisi","Anlatım Bozukluğu","Noktalama-Yazım","Ses Bilgisi","Yazım Kuralları","Cümle Ögeleri","Fiilimsiler"],
    "TYT Tarih": ["İlk Türk Devletleri","İslam Tarihi","Osmanlı Kuruluş-Yükseliş","Kurtuluş Savaşı","İnkılaplar","Osmanlı Yükselme Dönemi","Osmanlı Duraklama Dönemi","İslam Öncesi Türk Tarihi","Türk İslam Devletleri","Haçlı Seferleri"],
    "TYT Coğrafya": ["Harita Bilgisi","İklim","Yer Şekilleri","Nüfus","Yerleşme","Türkiye Ekonomisi","Dış Kuvvetler","Doğal Afetler"],
    "TYT Felsefe": ["Bilgi Felsefesi","Varlık Felsefesi","Ahlak Felsefesi","Sanat Felsefesi","Siyaset Felsefesi","Din Felsefesi"],
    "AYT Matematik": ["AYT Matematik - Fonksiyonlar","AYT Matematik - Polinomlar","AYT Matematik - Trigonometri","AYT Matematik - Logaritma","AYT Matematik - Diziler","AYT Matematik - Limit-Süreklilik","AYT Matematik - Türev","AYT Matematik - İntegral","AYT Matematik - Permütasyon-Kombinasyon","AYT Matematik - Olasılık"],
    "AYT Geometri": ["AYT Geometri - Çemberin Analitiği","AYT Geometri - Analitik Geometri","AYT Geometri - Vektörler","AYT Geometri - Dönüşüm Geometrisi"],
    "AYT Fizik": ["AYT Fizik - Vektörler","AYT Fizik - Kuvvet-Tork-Denge","AYT Fizik - Çembersel Hareket","AYT Fizik - Basit Harmonik Hareket","AYT Fizik - Elektrik-Manyetizma","AYT Fizik - Modern Fizik","AYT Fizik - Dalga Mekaniği"],
    "AYT Kimya": ["AYT Kimya - Modern Atom Teorisi","AYT Kimya - Gazlar","AYT Kimya - Sıvı Çözeltiler","AYT Kimya - Kimyasal Tepkimeler","AYT Kimya - Kimya ve Enerji","AYT Kimya - Tepkime Hızı","AYT Kimya - Kimyasal Denge","AYT Kimya - Asit-Baz Dengesi","AYT Kimya - Organik Kimya"],
    "AYT Biyoloji": ["AYT Biyoloji - Sinir Sistemi","AYT Biyoloji - Endokrin Sistem","AYT Biyoloji - Destek-Hareket","AYT Biyoloji - Genden Proteine","AYT Biyoloji - Fotosentez","AYT Biyoloji - Solunum","AYT Biyoloji - Komünite Ekolojisi","AYT Biyoloji - Popülasyon Ekolojisi"],
    "AYT Edebiyat": ["AYT Edebiyat - Divan Edebiyatı","AYT Edebiyat - Halk Edebiyatı","AYT Edebiyat - Tanzimat Edebiyatı","AYT Edebiyat - Servet-i Fünun","AYT Edebiyat - Milli Edebiyat","AYT Edebiyat - Cumhuriyet Edebiyatı","AYT Edebiyat - Şiir Bilgisi","AYT Edebiyat - Edebi Sanatlar","AYT Edebiyat - Roman ve Hikaye"],
    "AYT Tarih": ["AYT Tarih-2 - Osmanlı Dağılma Dönemi","AYT Tarih-2 - 20. Yüzyıl Başlarında Osmanlı","AYT Tarih-2 - Atatürk İnkılapları","AYT Tarih-2 - İnkılaplar","AYT Tarih-2 - Çağdaş Türk Dünyası"],
    "AYT Coğrafya": ["AYT Coğrafya-2 - Çevre ve Toplum","AYT Coğrafya-2 - Küresel Ortam","AYT Coğrafya-2 - Doğal Sistemler","AYT Coğrafya-2 - Beşeri Sistemler"],
    "Felsefe Grubu": ["Felsefe Grubu - Mantık","Felsefe Grubu - Psikoloji","Felsefe Grubu - Sosyoloji","Felsefe Grubu - Felsefe"],
    "Din Kültürü": ["Din Kültürü","AYT Din Kültürü"],
}

GORSEL_YOGUN_ANAHTARLAR = {
    "analitik", "çember", "vektör", "dönüşüm geomet", "katı cisim",
    "üçgen", "dörtgen", "optik", "dalga", "harita", "yer şekil",
    "manyetizma", "elektrik-manyet",
}

def _konu_gorsel_yogun_mu(konu):
    """Konu adi gorsel-yogun anahtarlardan birini iceriyorsa True."""
    k = (konu or "").lower()
    return any(anahtar in k for anahtar in GORSEL_YOGUN_ANAHTARLAR)


# =====================================================================
#  KONU OZETLERI (PDF'ten okunur)
#  Eslestirme sayesinde satir sonu kaybolsa bile dogru calisir.
# =====================================================================

OZET_PDF_KLASORU = "/content/drive/MyDrive/MindCoach_Ozetler"

def _tum_konu_adlari():
    adlar = []
    for konular in DERS_KONULARI.values():
        adlar.extend(konular)
    return adlar

_KONU_OZETLERI_CACHE = {}

def _ozet_metni_temizle(ozet):
    """PDF'ten gelen duz metni okunabilir markdown'a cevirir:
    yapay satir kirilmalarini siler, '•' maddelerini gercek listeye cevirir,
    'Sık Yapılan Hatalar' ve alt basliklari ayirir."""
    if not ozet:
        return ozet
    ozet = re.sub(r"\s*(Sık Yapılan Hatalar|SIK YAPILAN HATALAR)\s*:", "@@HATA@@", ozet)
    ozet = re.sub(r"\n\s*[•·∙]\s*", "@@B@@", ozet)
    ozet = re.sub(r"\n\s*-\s+", "@@B@@", ozet)
    ozet = ozet.replace("\n", " ")
    ozet = ozet.replace("@@HATA@@", "\n\n**⚠️ Sık Yapılan Hatalar:**\n")
    ozet = ozet.replace("@@B@@", "\n- ")
    ozet = re.sub(
        r"(?<=[.\)])\s+([A-ZÇĞİÖŞÜ][A-Za-zçğıöşüÇĞİÖŞÜ ,\-/]{4,55}?):\s",
        r"\n\n**\1:** ", ozet)
    ozet = re.sub(r"[ \t]{2,}", " ", ozet)
    ozet = re.sub(r"\n{3,}", "\n\n", ozet)
    return ozet.strip()

def pdften_konulari_oku(pdf_yolu, bilinen_konular):
    """Tek bir PDF'ten konulari okur. Donus: {konu_adi: ozet} sozlugu."""
    try:
        import pdfplumber
    except ImportError:
        return {}
    try:
        tam_metin = ""
        with pdfplumber.open(pdf_yolu) as pdf:
            for sayfa in pdf.pages:
                tam_metin += (sayfa.extract_text(x_tolerance=1) or "") + "\n"
    except Exception:
        return {}

    konular = {}
    bloklar = re.split(r"(?:###\s*)?KONU\s*:\s*", tam_metin)
    for blok in bloklar[1:]:
        blok = blok.strip()
        eslesen = None
        for konu in sorted(bilinen_konular, key=len, reverse=True):
            if blok.startswith(konu):
                eslesen = konu
                break
        if eslesen:
            ozet = blok[len(eslesen):].strip()
            konular[eslesen] = _ozet_metni_temizle(ozet)
    return konular

def tum_ozetleri_yukle(klasor=None):
    """Klasordeki tum PDF'leri okuyup _KONU_OZETLERI_CACHE'i doldurur.
    Donus: (yuklenen_konu_sayisi, mesaj)."""
    global _KONU_OZETLERI_CACHE
    klasor = klasor or OZET_PDF_KLASORU
    bilinen = _tum_konu_adlari()

    if not os.path.isdir(klasor):
        return 0, f"⚠️ Klasör bulunamadı: {klasor}\nGoogle Drive'ı bağladın mı? PDF'leri bu klasöre koy."

    _KONU_OZETLERI_CACHE = {}
    pdf_sayisi = 0
    for dosya in os.listdir(klasor):
        if dosya.lower().endswith(".pdf"):
            pdf_sayisi += 1
            yol = os.path.join(klasor, dosya)
            konular = pdften_konulari_oku(yol, bilinen)
            _KONU_OZETLERI_CACHE.update(konular)

    n = len(_KONU_OZETLERI_CACHE)
    if n == 0:
        return 0, f"⚠️ {pdf_sayisi} PDF bulundu ama hiç konu okunamadı. Format/konu adlarını kontrol et."
    return n, f"✅ {pdf_sayisi} PDF'ten {n} konu özeti yüklendi."

def konu_ozeti_getir(konu_adi):
    """Bir konunun ozetini cache'ten dondurur. Yoksa None."""
    if not konu_adi:
        return None
    if konu_adi in _KONU_OZETLERI_CACHE:
        return _KONU_OZETLERI_CACHE[konu_adi]
    if " - " in konu_adi:
        son = konu_adi.split(" - ")[-1].strip()
        if son in _KONU_OZETLERI_CACHE:
            return _KONU_OZETLERI_CACHE[son]
    return None


ENNEAGRAM = {
    "Tip 1 - Mükemmeliyetçi": {"no":"1","yaklasim":"düzenli ve kuralcı, sistemli yaklaş"},
    "Tip 2 - Yardımsever": {"no":"2","yaklasim":"duygusal ve sıcak, şefkatli yaklaş"},
    "Tip 3 - Başarı Odaklı": {"no":"3","yaklasim":"hedef odaklı, başarı ve zafer dili kullan"},
    "Tip 4 - Bireyci": {"no":"4","yaklasim":"özgün ve duygusal, estetik bir dil kullan"},
    "Tip 5 - Araştırmacı": {"no":"5","yaklasim":"analitik, mantık ve nedensellik vurgula"},
    "Tip 6 - Sadakatçi": {"no":"6","yaklasim":"güven arayan, güven verici ol"},
    "Tip 7 - Maceracı": {"no":"7","yaklasim":"eğlenceli, oyunlaştır ve heyecanlandır"},
    "Tip 8 - Meydan Okuyan": {"no":"8","yaklasim":"güçlü ve kontrol seven, güç dili kullan"},
    "Tip 9 - Barışçıl": {"no":"9","yaklasim":"sakin ve huzurlu yaklaş"},
}

RUH_HALI_TON = {
    "😣 Tükenmiş": {"key":"tukenmis","puan":1,"yaklasim":"öğrenci tükenmiş ve bitkin; baskı yapma, çok küçük bir adım öner","ton":"çok yumuşak ve şefkatli","hedef":"küçük bir adım: 1-2 soru çöz veya konuyu oku"},
    "😟 İsteksiz": {"key":"isteksiz","puan":2,"yaklasim":"morali düşük, çalışası yok; nazikçe cesaret ver","ton":"destekleyici ve sıcak","hedef":"5 soru civarı küçük bir hedef"},
    "😰 Kaygılı": {"key":"kaygili","puan":3,"yaklasim":"stresli ve sınav kaygısı var; sakinleştir, güven ver","ton":"sakinleştirici ve güven verici","hedef":"bildiğin konulardan 5-10 soru, kendini kötü hissetme"},
    "😐 Normal": {"key":"normal","puan":4,"yaklasim":"ne iyi ne kötü, dengeli; net bir plan sun","ton":"dengeli ve yapıcı","hedef":"15-20 soru"},
    "🙂 Odaklı": {"key":"odakli","puan":5,"yaklasim":"sakin ve çalışmaya hazır; odağı destekle, yönlendir","ton":"net ve yönlendirici","hedef":"20-30 soru, zorlandığın konuya odaklan"},
    "🔥 Motive": {"key":"motive","puan":6,"yaklasim":"enerjik ve hırslı; bu enerjiyi kullan ve onu zorla","ton":"coşkulu ve hırslandırıcı","hedef":"30-40 soru, en zor test"},
}


# =====================================================================
#  SABLON YEDEK (model bozulursa)
# =====================================================================
KISILIK_ACILIS_HAVUZ = {
    "1": {
        "tukenmis": ["Bugün sadece listenden en küçük maddeyi tamamla, o yeter.", "Düzenini bozmadan, sadece tek bir konuya kısaca göz atalım.", "Planı yarına esnetebiliriz, bugün ufak bir tekrar işini görür.", "Sistemin sağlam, bugün sadece hafif bir okumayla günü bitir."],
        "isteksiz": ["Adım adım ilerlemek her zaman işe yarar, ilk göreve bak.", "Sadece ilk on dakikalık rutini tamamla, devamı kendiliğinden gelir.", "Hadi, planına sadık kalmanın verdiği o büyük rahatlığı hatırla.", "Ufak bir başlangıç bile dünün programını harika şekilde kurtarır."],
        "kaygili": ["Planın tıkır tıkır işliyor, sadece şu anki görevine odaklan.", "Eksiklerini çok iyi biliyorsun ve sistemli bir şekilde kapatıyoruz.", "Her şeyi bir anda çözemeyiz, sırayla ve kurallara uyarak gidelim.", "Derin bir nefes al, yaptığımız bu program seni hedefe ulaştıracak."],
        "normal": ["Programımız hazır, bugünkü görevleri eritmeye hızlıca başlayalım.", "Hedeflerimiz son derece net, bugünün maddelerini sırayla çizelim.", "Sistemli şekilde ilerliyoruz, sıradaki konuya hemen geçiş yapabilirsin.", "Düzenli ve kurallı çalışmanın tam vakti, hemen masaya geç."],
        "odakli": ["Dikkatini tam toplamışken o en zor testleri aradan çıkaralım.", "Plan tıkır tıkır işliyor, bu güzel temponu asla bozma.", "Mükemmel bir odaklanma, hemen sıradaki en önemli konuya geç.", "Zaman yönetimin harika ilerliyor, doğrudan sıradaki hedefe kilitlen."],
        "motive": ["Bugün tüm o çalışma listesini kusursuz bir şekilde temizleyelim!", "Hadi, bu harika enerjiyle günlük planının da ilerisine geçebilirsin!", "Mükemmel bir gün, tüm belirlediğin hedefleri tek tek devir!", "Çalışma sistemine ivme kat, bugün gerçekten bir rekor kıracağız!"],
    },
    "2": {
        "tukenmis": ["Kendine biraz zaman tanı, bugün sadece dinlenerek sürece başla.", "Çok yoruldun biliyorum, lütfen bugün kendine biraz şefkat göster.", "Bugün sadece seni mutlu edecek, en sevdiğin küçük adımı at.", "Zorlama kendini, bugün sadece ufak bir tekrar yapalım."],
        "isteksiz": ["Biliyorum içinden hiç gelmiyor ama bunu beraber aşacağız.", "Hadi canım, masaya geçip birlikte ufak bir başlangıç yapalım.", "Sadece bir konuya bak, bunun sana ne kadar iyi geleceğini göreceksin.", "Gülümse ve sadece o ilk adımı at, ben hep yanındayım."],
        "kaygili": ["Derin bir nefes al, ben buradayım ve her şey yolunda.", "Sınav stresi seni sakın üzmesin, sen elinden gelenin en iyisini yapıyorsun.", "Yalnız değilsin, bu zor anları el ele verip atlatacağız.", "Hiçbir deneme notu senin değerini belirlemez, rahatla biraz."],
        "normal": ["Bugün nasılsın? Hadi güzelce masamıza geçip çalışmaya başlayalım.", "Güzel bir gün, büyük bir keyifle hazırlığımıza başlayalım.", "Beraber çok güzel işler çıkaracağımız harika bir seans olsun.", "Kendine güzel bir çay al, yavaşça ve keyifle başlayalım."],
        "odakli": ["Bu güzel enerjini hissetmek harika, hemen konumuzun içine dalalım.", "Hazır hissediyorsun, bu anı değerlendirip güzelce çalışalım.", "Tam istediğimiz gibi, dikkatini hiç bölmeden aynen devam et.", "Sakinliğin çok değerli, bu ritmi sevgiyle koruyarak ilerle."],
        "motive": ["Harikasın! Bu güzel enerjiyle bugün dağları bile aşarız!", "İçindeki o sıcak güçle bugün masadaki her şeyi halledeceksin!", "Çok iyisin! Hadi bu harika enerjini hemen sorulara yansıt!", "Bu tatlı coşkunla bugün en zor konular bile kolayca bitecek!"],
    },
    "3": {
        "tukenmis": ["Zirveye giden yolda verilen küçük molalar da son derece stratejiktir.", "Sadece en kolay soruyu çöz, bugünkü zaferin o küçük adım olsun.", "Bugün şampiyonların dinlenme günü olsun, ufak bir tekrar yap.", "Başarı bir maratondur, bugün enerjini toplayarak küçük bir adım at."],
        "isteksiz": ["Rakiplerin çalışıyor, hadi kalk ve onlara farkını netçe göster.", "Hedeflediğin o güzel üniversite için hemen masaya oturma vakti.", "Gerçek kazananlar, en zorlandığında bile masada kalanlardır.", "Sadece başla, o tatlı başarma hissi seni hemen toparlayacak."],
        "kaygili": ["Geçmişteki zaferlerini hatırla, bunu da kesinlikle halledeceksin.", "Stresini yakıta çevir, büyük hedefin tam karşında seni bekliyor.", "Sen bu süreçten galip çıkacaksın, sadece önündeki soruya odaklan.", "Kaygı duyman normal, ama senin kazanma arzun çok daha güçlü."],
        "normal": ["Hedefe bir adım daha yaklaşmak için yeni bir gün başladı.", "Bugünkü netlerini artırmak için hiç vakit kaybetmeden masaya geçiyoruz.", "Başarı planı devrede, sıradaki görevini hızlıca tamamla.", "Bugün de her zamanki gibi kazanmak için buradayız, başlayalım."],
        "odakli": ["Tam bir şampiyon odağı, masadaki en zor sorulara geçelim.", "Hazırsan doğrudan net getirecek o can alıcı konuyu bitirelim.", "Hedefe kilitlenmiş durumdasın, arkana aldığın bu rüzgarı kaçırma.", "Gözünü zirveden hiç ayırma, aynı dikkatle tam gaz devam."],
        "motive": ["Bu hırsla bugün önündeki o zorlu denemeyi parçalayacaksın!", "Bugün senin günün, sınırlarını sonuna kadar zorla ve kazan!", "Rekor kırmaya hazır mısın? Hiç durma, hemen en zora saldır!", "İşte şampiyonun gerçek enerjisi, bugün herkesi geride bırak!"],
    },
    "4": {
        "tukenmis": ["Ruhun yorulduysa bugün sadece senin için anlamlı olan derse bak.", "Kendi ritmini bul, bugün sadece bir sayfa okusan da yeterli.", "Zoraki değil, sadece içinden gelen o küçük ve zarif adımı at.", "Kendini dinle, bugün sadece sana gerçekten iyi gelen konuya odaklan."],
        "isteksiz": ["Kendi özgün hikayeni yazıyorsun, hadi o kalemi eline al.", "İlhamın yoksa bile, bir yerden başlamak ruhuna iyi gelir.", "Senin tarzın farklı, kendi özel yöntemine göre çalışmaya başla.", "Masana geç, en sevdiğin şarkı eşliğinde ufak bir başlangıç yap."],
        "kaygili": ["İçindeki o karmaşayı anlıyorum, derin bir nefesle her şeyden uzaklaş.", "Senin potansiyelin çok farklı, zihnini yoran bu endişeler geçici.", "Kendi özel yolunda ilerliyorsun, lütfen kendini kimseyle kıyaslama.", "Kaygını bir kağıda dök ve şimdi sadece kendi sürecine odaklan."],
        "normal": ["Kendi özgün düzeninde bugünün çalışmasını en güzel şekilde tasarla.", "Bugün senin için gerçekten anlamlı olan o derse derinlemesine odaklanalım.", "Sakin bir zihinle kendi özel hikayene yepyeni bir sayfa ekle.", "Kendi iç ritmini hisset ve çalışmana keyifle başla."],
        "odakli": ["O derin dikkatini yakalamışken en sevdiğin konunun içine dal.", "Tam senin o özel akışındayız, bu güzel hissi bozmadan ilerle.", "İlham perilerin tam yanında, hadi bu derin odakla üretmeye başla.", "Bu derinlikte çalışırken okuduğun her detay zihnine kalıcı kazınacak."],
        "motive": ["Bu muazzam tutkunla bugün bambaşka bir seviyeye çıkacaksın!", "İçindeki o güçlü yaratıcılıkla bugün masadaki her soruyu çözeceksin!", "Kendi potansiyelini zirvede hissettiğin bir gün, hadi başla!", "Bu harika duygu seliyle bugün kendi unutulmaz efsaneni yaz!"],
    },
    "5": {
        "tukenmis": ["Beyninin enerji seviyesi düşük, sadece basit bir konuyu tarayıp bırak.", "Veri işlemini yavaşlatalım, bugün sadece elindeki kısa özetlere bak.", "Zihinsel yorgunluk mantıklıdır, en düşük eforlu görevi seçip ilerle.", "Dinlenmek de sistemin bir parçası, ufak bir okuma yeterli olur."],
        "isteksiz": ["Neden çalışman gerektiğini mantıken çok iyi biliyorsun, sadece başla.", "Optimum verimlilik için küçük bir başlangıç şu an yeterlidir.", "Duyguları kenara bırak, bugünün hedefine mantıkla odaklanıp başlayalım.", "İsteğin yoksa bile, mantıklı olan on dakika boyunca denemektir."],
        "kaygili": ["Sınav sadece bir ölçüm aracı, mantıklı düşün ve gerçek veriye odaklan.", "İhtimal hesaplarını bırak, sadece şimdiki konunun nedenselliğini çalış.", "Kaygı rasyonel bir durum değildir, gerçek bilgilere ve masana dön.", "Eksik analizini doğru yaptık, adım adım kapatıyoruz, sakinleş."],
        "normal": ["Zihnin gayet berrak, yeni bilgi toplama sürecine hızlıca başlayabiliriz.", "Bugünkü konuların mantıksal analizini yapmak için hemen masaya geç.", "Verimli bir çalışma seansı için zihinsel hazırlığını yap, başlayalım.", "Konuların temel çalışma mantığını tam olarak kavramak için odaklan."],
        "odakli": ["Zihnin tam analiz modunda, o en karmaşık sorulara hemen dal.", "Bilişsel kapasiten zirvede, hiç beklemeden yeni ve zorlu bilgilere geç.", "Bu derin odakla tüm nedensellikleri ve formülleri kolayca çözeceksin.", "Beynin sünger gibi her şeyi çekiyor, hemen o zor konuyu bitir."],
        "motive": ["Zihinsel enerjin muazzam seviyede, bugün çok fazla veri işleyeceksin!", "Mantık motorun tam gaz çalışıyor, en zor denemeye hemen gir!", "Algın sonuna kadar açık, bugün maksimum bilgi transferi yapacağız!", "Sahip olduğun bu analitik güçle bugün hiçbir soru seni zorlayamaz!"],
    },
    "6": {
        "tukenmis": ["Bugün sadece en güvendiğin, en iyi bildiğin eski konuya bak.", "Benimle güvendesin, bugün sadece temel ve kolay bir tekrar yapalım.", "Hiç risk alma, kendini zorlamadan rutin ve ufak bir adım at.", "Yorulmuşsun, bugün sadece garantili olan en küçük adımları atalım."],
        "isteksiz": ["Bu sürecin sonu gerçekten güzel olacak, sadece bana güven ve başla.", "Rutinimiz belli, ona sadık kalmak bugün sana da iyi gelecek.", "Zorlanabilirsin ama alışkın olduğumuz o tempoda ufaktan başlayalım.", "Hiç endişelenme, sadece bildiğin o güvenli yoldan yürümeye devam et."],
        "kaygili": ["Sınava birlikte hazırlanıyoruz, asla yalnız değilsin ve tam güvendesin.", "Her şey bizim kontrolümüz altında, planımıza sadık kalarak ilerliyoruz.", "Sakin ol, sorular bildiğin yerden gelecek, sen çok emek verdin.", "Korkacak hiçbir şey yok, son derece sağlam adımlarla yürüyoruz."],
        "normal": ["Her zamanki düzenimizde, güvenli bir şekilde hemen çalışmaya başlayalım.", "Rutin programımız sorunsuz çalışıyor, bugünün standart görevlerine geç.", "Bildiğimiz ve emin olduğumuz adımlarla ilerliyoruz, hadi masaya geçelim.", "Eksiksiz gidiyoruz, hazırladığımız plana tam güven ve çalışmaya başla."],
        "odakli": ["Sakin ve güvenli hissediyorsun, bu anı çok iyi değerlendir.", "Tam istediğimiz noktadayız, güvenle ve şüphe etmeden soruları çöz.", "Adımların çok sağlam ve yere basıyor, bu güvenle konuyu bitir.", "İşte beklediğimiz o güzel istikrar, bu odakla rahatça ilerle."],
        "motive": ["Harika bir enerji, bugün planımızın çok daha ilerisine geçeceğiz!", "Kendine olan güvenin tam, bugün en zor konuları bile kolayca yıkarsın!", "Bu inançla seni hiçbir zorluk korkutamaz, tam gaz yola devam!", "Güçlüsün ve güvendesin, bugün çok harika işler başaracaksın!"],
    },
    "7": {
        "tukenmis": ["Enerjin bitmişse bugün en eğlenceli videoları izleyerek yavaşça çalışalım.", "Çok sıkıldıysan sadece bir oyun gibi ufak bir test çöz.", "Zorlama yok, bugün sadece en çok keyif aldığın derse bak.", "Bugün çalışmayı oyunlaştırıyoruz, en kısa sürede bitirip ödülünü al."],
        "isteksiz": ["Bugünkü zor konuları eğlenceli bir yarışmaya çevirmeye ne dersin?", "Canın sıkkın biliyorum ama masadaki o ufak zaferi hemen yakalayabiliriz.", "Hadi çalışmayı sıkıcı olmaktan çıkarıp çok renkli bir hale getirelim.", "Sadece beş dakika ayır, belki de sandığından çok keyifli bir konudur!"],
        "kaygili": ["Sınavı bir canavar gibi değil, geçilecek eğlenceli bir seviye gibi düşün.", "Çok kasma kendini, bu sadece çözülmesi gereken dev bir bulmaca.", "Eğlencene bak, bu çalışma süreci de bir macera, anın keyfini çıkar.", "Daraldıysan hemen ortam değiştir, yeni ve eğlenceli bir yerde çalış."],
        "normal": ["Bugün yeni bilgiler keşfedeceğimiz harika bir macera bizi bekliyor.", "Hadi çalışmayı keyifli bir oyuna çevirelim ve gülümseyerek başlayalım.", "Enerjimiz gayet yerinde, sıradaki konuları neşeyle eritelim.", "Yeni bir keşif için masaya otur, soruları renkli bir oyun gibi çöz."],
        "odakli": ["Oyun moduna girmişsin, bu hızla masadaki tüm levelleri geçersin.", "Dikkatini yakaladık, hadi bu oyunun en zor bölümünü de geç.", "Odağın harika, eğlenerek öğrenmenin ve net artırmanın tam zamanı.", "Bu akışta kal, soruları çözerken nasıl hızlandığına kendin bile inanamayacaksın."],
        "motive": ["Enerjin harika, bugün kendi rekorumuzu kırıp seviye atlayacağız!", "Tam bir macera günü, o en zor sorulara adeta uçarak dal!", "Bu coşkuyla bugün çalışmayı resmen bir ziyafete çevireceksin!", "Hadi, büyük oyun başlıyor, bugün önüne çıkan her soruyu anında avla!"],
    },
    "8": {
        "tukenmis": ["Kontrol hala sende, bugün sadece küçük bir kaleyi fethedip dinlen.", "Güç toplamak için geri çekiliyorsun, sadece ufak bir tekrar yap.", "Bugün sınırlarını hiç zorlama, o büyük gücünü yarına sakla.", "İpi elden bırakmadan, bugün sadece hafif ve ufak bir adım at."],
        "isteksiz": ["Sen zorluklardan kaçmazsın, masaya otur ve o gücünü göster.", "İsteksizliği yenmek de bir savaştır, o kalemi eline al ve kazan.", "Odanın patronu sensin, tüm bahaneleri ez geç ve hemen başla.", "Geri adım atmak sana göre değil, gücünü topla ve masaya otur."],
        "kaygili": ["Kaygıyı ez geç, sürecin komutanı sensin ve her şeyi başaracaksın.", "Sınav senin gücünün altında kalacak, dik dur ve hemen başla.", "Korkuların seni asla yönetemez, kontrolü eline al ve ilerle.", "Derin bir nefes al, o masayı ve tüm soruları sen fethedeceksin."],
        "normal": ["Gücünü sahaya yansıtma vakti geldi, bugünün görevlerini bitir.", "Kontrol tamamen senin ellerinde, planını gümbür gümbür uygulamaya başla.", "Kararlı adımlarla ilerliyoruz, hadi masadaki o konuları hallet.", "Sınırlarını belirle ve gücünü masaya koy, hiç durmadan başlayalım."],
        "odakli": ["Dikkatin jilet gibi keskin, masadaki en sert soruları parçala.", "Gücünün zirvesindesin, bu anı hemen kullan ve hedefi tam ortadan vur.", "Odağın mükemmel, kontrol sende, hiç durmadan ve yıkmadan ilerle.", "Tam kıvamındasın, şimdi o zor konuların hakkından gelme vakti."],
        "motive": ["Bu güçle bugün tüm soruları ezip geçeceksin, hiç durma saldır!", "İçindeki o savaşçı ateşle bugün o denemeyi resmen darmadağın et!", "Bugün meydan okuma günü, kendine en zoru seç ve onu yen!", "Patron kim göster onlara, bugün masada durmak yok, sonuna kadar savaş!"],
    },
    "9": {
        "tukenmis": ["Telaş yok, sakince tek bir küçük adım atman bugün için yeter.", "Kendini hiç yorma, hafif bir müzikle usulca ufak bir tekrar yap.", "Bugün acele etmeden, sadece zihnini dinlendiren huzurlu bir ders seç.", "Baskı yok, sakince kitabın yüzüne bakman bile bugün harika bir adım."],
        "isteksiz": ["İçinden gelmiyorsa yavaş yavaş başlayalım, hiçbir acelemiz yok.", "Ufak ufak ısın, kendini hiç zorlamadan sakince masaya geç.", "Sakin kal, önce güzel bir çay yap, usulca konulara bakarız.", "Dert etme, sadece bir paragraf okuyarak huzurla günümüze başlayalım."],
        "kaygili": ["Hiçbir sınav senin iç huzurundan önemli değil, derin bir nefes al.", "Sakinleş, her şeyi yavaş yavaş hallederiz, adım adım ilerliyoruz.", "Kaygıya hiç yer yok, biz kendi halimizde sakince çalışacağız.", "Zihnini dingin tut, her şey kendi akışında en güzel şekilde düzelecek."],
        "normal": ["Sessiz ve huzurlu bir ortamda, sakince çalışmaya başlayalım.", "Zihnimiz durgun bir göl gibi, usulca yeni konulara geçiş yap.", "Acele etmeden, oldukça keyifli bir şekilde bugünü yavaşça değerlendirelim.", "Huzurlu bir çalışma seansı için usulca masamıza yerleşelim."],
        "odakli": ["Sakinliğini koruyarak, bu güzel ve huzurlu akışta çalışmaya devam et.", "Dingin bir zihinle konuları ne kadar rahat anladığını sen de gör.", "Bu huzurlu odağı hiç bozmadan masadaki soruları usulca çöz.", "Dengeyi buldun, bu güzel dinginlikle bugün çok verimli ilerleyeceksin."],
        "motive": ["Bu tatlı ve sakin enerjinle bugün çok güzel şeyler öğreneceğiz.", "İçindeki o sessiz güç uyandı, bugün rahatça her şeyi halledeceksin.", "Huzur ve pozitif enerji bir arada, çok verimli bir gün başlıyor.", "Sakin ama çok güçlü adımlarla bugün harika şekilde ilerleyeceksin."],
    },
}


CALISMA_TARZI_HAVUZ = {
    "1": {
        "tukenmis": ["Bugün sadece konu özetlerindeki temel kuralları gözden geçirerek günü bitir.", "Yeni konu çalışmak yerine masanı ve notlarını sistematik bir şekilde düzenle.", "Zorlanmadan sadece en iyi bildiğin konunun formüllerini düzgünce temize çek.", "Günün tek görevi olarak bir konunun şemasını çıkar ve listenden işaretle."],
        "isteksiz": ["Sıfırdan başlamak yerine sadece dünkü konuların eksiklerini kuralına göre tamamla.", "İçinden gelmiyorsa en sistematik ve kurallı dersi seçip küçük bir test çöz.", "Planının sadece en küçük ve net tanımlanmış maddesini kusursuzca hallet.", "Sadece bir dersin konu anlatımındaki kurallar bölümünü oku ve bırak."],
        "kaygili": ["Kaygını azaltmak için tamamen bildiğin, kuralları net olan standart sorulara yönel.", "Belirsizlik seni strese sokar, bu yüzden adım adım çözümü olan çıkmış soruları incele.", "Sıralı ve düzenli notlarını baştan sona okuyarak zihnindeki o düzeni tekrar sağla.", "Planının dışına çıkmadan, sadece bugünün en temel görevini kuralına uygun yap."],
        "normal": ["Hazırladığın planı maddeler halinde, sırasını hiç bozmadan uygulamaya başla.", "Konunun temel mantığını ve kurallarını iyice oturttuktan sonra soru çözümüne geç.", "Hata defterini yanına al ve yanlışlarını sistematik bir şekilde analiz ederek çalış.", "Çalışma masanı tamamen düzenle ve sıradaki konuyu kurallarına göre parçalara böl."],
        "odakli": ["Tam odaklanmışken kuralları en karmaşık olan o zor konunun altını üstüne getir.", "Dikkatini hiç bozmadan bir konuyu baştan sona eksiksiz ve sıfır hatayla bitir.", "Sistemli zihnini kullan ve birbirine bağlanan birden fazla konunun şemasını çıkar.", "Sınırlarını zorla ve en zor kaynaklardan kural ezberi gerektiren testleri temizle."],
        "motive": ["Bugün listendeki her şeyi sıfır hatayla ve kusursuz bir düzenle bitirme günü.", "Enerjin yüksekken tüm zorlu konuları kendi mükemmel sistemine göre arşivle.", "Hata kabul etme, bugün çözdüğün her sorunun en ince detayına kadar in.", "Bu yüksek enerjiyle tüm konu eksiklerini bul ve sistematik olarak hepsini yok et."],
    },
    "2": {
        "tukenmis": ["Sadece sevdiğin bir hocadan sohbete benzer, rahatlatıcı bir konu anlatımı dinle.", "Hiç zorlama, sadece sesli bir şekilde hayali birine bugünün en kolay konusunu anlat.", "Bir arkadaşınla kısaca telefonda görüşüp birlikte tek bir konu tekrarı yapın.", "Sadece sana kendini iyi hissettirecek bir konuyu başkasına anlatıyormuş gibi oku."],
        "isteksiz": ["Çalışma isteğini tetiklemek için bir arkadaşına söz ver ve onunla beraber başla.", "İçinden gelmese de sanki birine konuyu öğretiyormuşsun gibi sesli anlatım yap.", "Sosyal bir ortama veya kütüphaneye geçerek etrafındaki çalışma enerjisinden beslen.", "Sadece bir arkadaşının çözemediği bir soruya bak, başkasına yardım etmek seni motive edecektir."],
        "kaygili": ["Stresini atmak için güvendiğin bir çalışma arkadaşınla soru çözüm saati yap.", "Kendi kendine daralmak yerine, anlamadığın konuyu sevdiğin bir hocaya veya arkadaşına sor.", "Kaygılı anlarda notlarını sesli okuyarak kendi sesinle kendini telkin et.", "Birlikte çalışmaktan keyif aldığın biriyle konuları karşılıklı tartışarak tekrar edin."],
        "normal": ["Kendi notlarını çalıştıktan sonra aynaya bakarak konuyu birine anlatır gibi tekrar et.", "Grup çalışmaları veya etüt ortamları bugün senin için en verimli alanlar olacak.", "Öğrendiklerini pekiştirmek için zorlandığın konuyu hayali bir öğrenciye öğretmen gibi anlat.", "Soru çözerken sesli düşün ve çözümleri sanki birine açıklıyormuş gibi ifade et."],
        "odakli": ["Odağın çok iyiyken kendi kelimelerinle başkalarının da anlayacağı harika özetler çıkar.", "Derinleşmek için konunun en zor kısmını seç ve sanki sahnede anlatacakmış gibi hazırlan.", "Dikkatini bozmadan en karmaşık konuyu zihninde başkalarıyla tartışarak analiz et.", "Tam konsantrasyonla çalışırken başkalarına da faydası dokunacak şematik notlar hazırla."],
        "motive": ["Bu enerjinle harika bir konu özet kitapçığı çıkarıp arkadaşlarınla paylaşabilirsin.", "Grup çalışmasına liderlik et ve bugün tüm arkadaşlarına zor bir konuyu sen anlat.", "Sıcak ve coşkulu enerjini kullanarak bugün en zor konuları sesli anlatımla devir.", "Hem kendi eksiklerini kapat hem de başkalarına faydalı olacak o efsane notları yaz."],
    },
    "3": {
        "tukenmis": ["Sadece sana ufak bir zafer hissi yaşatacak en kolay testten bir tane çözüp bırak.", "Bugün sadece ilerlemeni görmek için bitirdiğin konuları fosforlu kalemle çiz.", "Yorgunsan sadece hedef listeni güncelle ve yarınki zaferin için ufak bir adım at.", "Büyük hedefini düşün ve masada sadece kısa bir okuma yaparak o tiki at."],
        "isteksiz": ["Moralini düzeltmek için netlerinin en yüksek olduğu o favori denemeni incele.", "Başarma hissini tatmak için hemen masaya geçip en kısa konuyu hızlıca bitir.", "Rakiplerinin şu an çalıştığını hayal et ve sadece bir konuyu aradan çıkar.", "Hedeflediğin bölümün fotoğrafına bak ve sadece bu uğurda küçük bir test çöz."],
        "kaygili": ["Kaygını dindirmek için şu ana kadar bitirdiğin tüm konuların listesine gururla bak.", "Stresliysen sadece sonucu garanti olan ve sana kesin net getirecek konuları tekrar et.", "Geçmişteki en iyi deneme sonucunu önüne koy ve o seviyeye tekrar çıkacağına inanarak başla.", "Kendini başkalarıyla kıyaslamak yerine sadece kendi gelişim grafiğine odaklanarak çalış."],
        "normal": ["Bugünkü çalışman için net hedefler belirle ve onları tamamladıkça üstünü gururla çiz.", "Kronometreni aç, çözdüğün her testte hız rekorunu kırmaya odaklanarak ilerle.", "Çalışmanı net sayını artıracak stratejik konulara yönlendirerek verimi maksimize et.", "Her dersin sonunda kendine küçük ödüller koyarak hedef odaklı çalışmanı sürdür."],
        "odakli": ["Dikkatin zirvedeyken deneme netlerini doğrudan artıracak en zorlu analitik sorulara dal.", "Hazır odaklanmışken zaman tutarak kendi en iyi süreni geçmek için hız testi yap.", "Hedefine kilitlenmiş durumdasın, hiç durmadan peş peşe o zor testleri erit.", "Bu derin odakla en çok hata yaptığın konuyu analiz et ve o eksiği kalıcı olarak kapa."],
        "motive": ["Hırsın tavan yapmışken gir o zorlu denemeye ve kendi net rekorunu darmadağın et.", "Bugün rakiplerine fark atma günü, en yoğun ve zorlu soru bankasını önüne al.", "Bu enerjiyle bugün hedefler listendeki tüm maddeleri acımasızca çizip bitir.", "Gözünü zirveye dik, bugün hiçbir soru seni durduramaz, sonuna kadar savaş."],
    },
    "4": {
        "tukenmis": ["Sadece sana estetik gelen, renkli kalemlerinle kendi tarzında küçük bir zihin haritası çiz.", "Yorgunsan çalışmayı bırak, zihnini toparlamak için sana ilham veren bir müzik eşliğinde notlarını izle.", "Standart testler yerine bugün sadece ilgini çeken tek bir alt konunun hikayesini oku.", "Zoraki bir düzene girme, sadece sana kendini iyi hissettirecek özgün bir şema karala."],
        "isteksiz": ["İlham gelmesini bekleme, masanı kendi zevkine göre süsle ve öyle bir başlangıç yap.", "Sıkıcı kaynaklar yerine konuyu farklı, edebi veya sıradışı anlatan bir video bularak başla.", "Ruhunu daraltan standart programı bir kenara koy, o an içinden hangi ders geliyorsa onu seç.", "Kendi cümlelerinle ve sana özgü sembollerle küçük bir özet çıkararak havaya gir."],
        "kaygili": ["Kaygını dışa vurmak için önce duygularını kağıda dök, sonra kendi özgün notlarına dön.", "Kıyaslanmaktan kaçın, sen eşsizsin; bu yüzden sadece kendi özel not alma stiline odaklan.", "Stresli anlarda standart soru bankalarını bırakıp konunun sana ilham veren derinliklerine in.", "Kendini güvende hissettiğin o özel çalışma köşene geç ve sadece kendi ritminde çalış."],
        "normal": ["Konuları kimsenin yapmadığı şekilde, tamamen kendi özel sembollerin ve renklerinle şemalaştır.", "Ezber yapmak yerine konunun içindeki duyguya veya estetiğe odaklanarak özgün bağlantılar kur.", "Sana has o özel çalışma rutinine sadık kalarak, notlarını adeta bir sanat eseri gibi işle.", "Klasik metotları es geç, zihninde en iyi yankı uyandıran o yaratıcı teknikle konulara dal."],
        "odakli": ["Bu derin histeyken konular arasındaki o görünmez bağları yakala ve muazzam bir zihin haritası yap.", "İlham perilerin seninleyken en karmaşık konuyu kendi eşsiz diline çevirerek kalıcı hale getir.", "Odağın tamken standartların çok dışına çıkıp tamamen sana ait bir çözüm stratejisi geliştir.", "Bu derinleşme anını kullan, her detayı kendi iç dünyanda anlamlandırarak notlarına aktar."],
        "motive": ["Bu muazzam yaratıcılıkla bugün kendi özel efsanevi özet defterini baştan yarat.", "Coşkun zirvedeyken sınırları yık, tüm konuları kendi o eşsiz perspektifinle yeniden yaz.", "İçindeki o ateşle klasik yöntemleri parçala, tamamen sana ait bir tarzda tüm soruları erit.", "Bugün sıradan olma günü değil, o güçlü ilhamınla masadaki her şeyi sanat eserine çevir."],
    },
    "5": {
        "tukenmis": ["Enerjin yoksa soru çözme, sadece bir konunun arka planındaki temel mantığı okuyup geç.", "Kendini hiç zorlamadan, uzandığın yerden bir konunun neden-sonuç ilişkisini anlatan belgesel izle.", "Sadece elindeki özetlere bakıp formüllerin nereden geldiğini hafifçe zihninde tart.", "Zihinsel yorgunluğunu artırmamak için veri işlemeyi bırak, sadece basit bir diyagramı incele."],
        "isteksiz": ["Sadece merakını cezbedecek bir alt başlığın 'neden' böyle olduğunu araştırarak başla.", "Test çözmek yerine formüllerin çıkış noktasını ispatlayarak zihnindeki o analitik kıvılcımı yak.", "Büyük bir konuya girmeden önce, sadece ilgini çeken küçük bir problemin mantığını çöz.", "Motivasyon arama, masaya sadece verileri analiz etmek ve nedensellik kurmak için otur."],
        "kaygili": ["Kaygını rasyonel verilerle yen, hangi konularda eksik olduğunu sayısal olarak analiz edip plan yap.", "Duygularından uzaklaş, konuyu en küçük mantıksal parçalarına bölerek kontrolü tekrar eline al.", "Sınav stresini bırak, önündeki sorunun mekaniğine ve çalışma prensibine odaklanarak sakinleş.", "Belirsizlik kaygı yaratır; bu yüzden sadece kesin kuralları olan analitik konulara yönel."],
        "normal": ["Konuları ezberlemek yerine, 'bu neden böyle?' sorusunu sorarak derinlemesine araştırıp öğren.", "Yeni bir formül veya kural öğrendiğinde hemen kabul etme, ispatını yaparak kendi zihnine kazı.", "Sistematik bir bilgi ağı kur, öğrendiğin her yeni bilgiyi eskileriyle mantıksal olarak bağla.", "Çalışırken tamamen bilgiye aç bir dedektif gibi davran, her sorunun kök nedenini analiz et."],
        "odakli": ["Zihnin jilet gibiyken en karmaşık ve soyut konuların nedensellik bağlarını tamamen çöz.", "Bilişsel kapasiten zirvedeyken çok adımlı, derin analitik düşünme gerektiren sorulara dal.", "Odaklanmışken dış dünyayı tamamen kapat ve o zor denemenin her bir detayını deşifre et.", "Bu derin dikkatle birbirine benzeyen konuların ayrım noktalarını mantıksal çerçeveye oturt."],
        "motive": ["Bu yüksek zihinsel enerjiyle bilgi denizine dal ve en zor konuların şifrelerini kır.", "Analitik motorun tam kapasitedeyken tüm formüllerin ispatını çıkarıp kendi sistemini kur.", "Aklın sınırlarını zorla, bugün hiçbir ezbere yer yok, her şeyi salt mantıkla parçala.", "Bu harika kapasiteyle tüm soru tiplerinin arka planındaki algoritmayı deşifre et."],
    },
    "6": {
        "tukenmis": ["Bilinmezliklere girme, bugün sadece en güvendiğin hocanın eski özet notlarını gözden geçir.", "Gücün yoksa yeni bir adım atma, tamamen bildiğin konuları yavaş ve güvenli şekilde tekrar et.", "Zorlamadan, hiçbir sürprizi olmayan, standart ve en kolay çıkmış sorulara şöyle bir göz at.", "Güvenli alanından çıkma, sadece defterindeki temel tanımları okuyarak günü sakin bitir."],
        "isteksiz": ["Konuya ısınmak için sadece çözümü içinde olan garantili örnek soruları inceleyerek başla.", "Başarısızlık korkusunu atmak için adım adım yönlendiren, tamamen bildik bir kaynağı aç.", "Risk alma, önceden çözdüğün ve doğru yaptığın bir testi tekrar ederek o güven hissini çağır.", "Motivasyon bekleme, sadece sana ne yapman gerektiğini söyleyen rutinine harfiyen uy."],
        "kaygili": ["Kaygıyı durdurmak için en güvendiğin rehberin veya öğretmenin hazırladığı sabit programa dön.", "Stres anında sürpriz arama, sadece konuyu en sade anlatan o güvenli kaynağına sarıl.", "Geleceği düşünmeyi bırak, sadece önündeki bir sonraki küçük adımı güvenle atmaya odaklan.", "Kafandaki şüpheleri sustur, daha önce işe yaradığını bildiğin o klasik çalışma yöntemini uygula."],
        "normal": ["Plana sadık kal, her zaman yaptığın gibi sağlam kaynaklardan adım adım ve kontrollü ilerle.", "Yeni konuya geçerken önce temeli tamamen sağlamlaştır, soru işaretleri bırakmadan yürü.", "Riskli ve farklı yöntemler yerine, kendini en güvende hissettiğin düzenli not alma stilini kullan.", "Her konuyu bitirdiğinde mutlaka doğruluğundan emin olduğun çıkmış sorularla teyit et."],
        "odakli": ["Dikkatini toplamışken temeli sarsılmaz bir şekilde oturtmak için konunun en derinlerine in.", "Odağın harikayken, tüm soru işaretlerini ortadan kaldıracak şekilde güvendiğin kaynakları bitir.", "Adımların çok sağlam, bu güvenli odakla denemedeki en ufak boşlukları bile doldur.", "Kontrol tamamen sende, bu dikkati hiç bozmadan planındaki o zor konuyu güvenle tamamla."],
        "motive": ["Bu güçlü güven hissiyle bugün o çok korktuğun konuların bile üstüne korkusuzca git.", "İstikrarın zirvesinde, planladığın her şeyi o sarsılmaz iradenle adım adım erit.", "Arkanı tamamen sağlama aldın, bu yüksek enerjiyle hiçbir konuyu şansa bırakmadan bitir.", "Kendine ve sistemine tam güven, bugün şüpheye yer vermeden tüm soruları ezip geç."],
    },
    "7": {
        "tukenmis": ["Sıkıcı tekrarlar yerine uzanıp konuyu oyun gibi anlatan eğlenceli bir animasyon izle.", "Sadece renkli bilgi kartlarıyla kendini hiç yormadan küçük bir hafıza oyunu oyna.", "Masada oturma, sevdiğin bir müziği aç ve odayı dolaşırken kısa sesli özetler dinle.", "Bugün dersleri boşver, sadece sana ilginç gelen ufak bir genel kültür videosuna bak."],
        "isteksiz": ["Başlamak için dersi değiştir, ilgini çeken en eğlenceli konuyu açıp sadece ona bak.", "Çalışmayı yarışmaya çevir, soruları bir oyunun bölümleri gibi düşünüp en kısasından başla.", "Sıkıntı bastıysa aynı yerde kalma, ortam değiştirip kütüphaneye veya kafeye giderek başla.", "Klasik testleri bırak, renkli kalemlerle veya farklı dijital araçlarla çalışmayı hareketlendir."],
        "kaygili": ["Kaygını dağıtmak için konuları zihninde eğlenceli ve komik hikayelere dönüştürerek ezberle.", "Stres seni kilitliyorsa tek derse odaklanma, sıkıldıkça dersler arasında hızlı geçişler yap.", "Sınavı bir son değil, geçilmesi gereken keyifli bir macera gibi görerek sorulara yaklaş.", "Daraldığını hissettiğin an çalışma ortamını veya materyalini değiştirerek zihnine nefes aldır."],
        "normal": ["Rutinden kaçın, çalışmanı farklı kaynaklar ve videolarla sürekli taze ve çeşitli tut.", "Uzun saatler tek konuya gömülme, farklı dersleri harmanlayarak dinamik bir program uygula.", "Konuları zihninde oyunlaştır, kendini bir dedektif veya kaşif gibi hayal ederek araştır.", "Çalışma anlarını keyifli hale getirmek için interaktif uygulamalar ve flashcardlar kullan."],
        "odakli": ["Tam havaya girmişken konular arasında jet hızıyla geçiş yaparak çoklu bağlantılar kur.", "Odağın mükemmel, bu eğlenceli akışta kalarak o zor konuyu bir çırpıda oyun gibi bitir.", "Dikkatin dağılmadan, konuyu farklı perspektiflerden ele alan çeşitli kaynakları hızla tara.", "Bu keyifli enerjinle kendi bulduğun o pratik ve eğlenceli yöntemle soruları art arda çöz."],
        "motive": ["İçindeki bu coşkuyla çalışmayı bir ziyafete çevir, en zor konuların üzerinden uçarak geç.", "Enerjin harika, hemen oyun moduna gir ve bütün zorlu testleri bir rekor kırarak parçala.", "Sıkılmak yok, bugün bu yüksek neşeyle daldan dala atlayarak tüm eksikleri keyifle kapat.", "Bu çılgın enerjiyle masa başında harikalar yarat, her bir soruyu keyifli bir zafer yap."],
    },
    "8": {
        "tukenmis": ["Gücün az olsa da pes etme, sadece tek bir zor soruyla yüzleşip günü öyle kapat.", "Tamamen bırakmak sana göre değil, kontrolün sende olduğunu hissetmek için bir konuyu tekrar et.", "Bugün sınırları zorlama, enerjini toplamak için sadece temel kuralları okuyup geri çekil.", "İpi bırakma, dinleniyor olsan da yarınki büyük savaş için kısa bir zihinsel hazırlık yap."],
        "isteksiz": ["Masaya hükmet, bahaneleri ezip geç ve doğrudan en sevmediğin konuya saldırarak başla.", "İsteksizliği bir düşman gibi gör, onu yenmek için o kalemi eline alıp en zordan başla.", "Kendi kendine meydan oku: 'Bu konu mu beni yenecek?' de ve acımasızca soru çözümüne gir.", "Kolaydan başlamak seni sıkar, doğrudan denemedeki o en sert soruların üstüne yürü."],
        "kaygili": ["Kaygı seni zayıflatamaz; endişeyi öfkeye, öfkeyi de çalışma hırsına çevirip konulara saldır.", "Sınav stresi kontrolü ele almadan önce sen masanın kontrolünü al ve sağlam bir başlangıç yap.", "Korkuların üstüne git, seni en çok strese sokan konu hangisiyse ilk önce onu parçala.", "Dik dur, bu süreci sen yönetiyorsun; zayıf hissettiğin o konuyu bugün acımasızca bitir."],
        "normal": ["Zorluklardan kaçma, her çalışmaya seni en çok zorlayacak ve geliştirecek kısımdan başla.", "Kendi kurallarını koy ve programına tavizsiz bir disiplinle harfiyen uyarak ilerle.", "Ders çalışmayı bir güç gösterisine çevir, her doğru cevabı bir zafer gibi görerek çöz.", "Başkalarının ne yaptığına bakma, kendi gücünü masaya koy ve kararlılıkla konuları devir."],
        "odakli": ["Dikkatin jilet gibiyken hiç vakit kaybetmeden o en ileri düzey soruların hakkından gel.", "Odağın tam, şu an o masadaki tek otorite sensin, bu gücü kullanıp konuları darmadağın et.", "Zihnin bu kadar keskinken kendi rekorlarını kırmak için zorlu bir hız denemesine gir.", "Bu derin konsantrasyonla önündeki en sert engeli acımasız bir kararlılıkla paramparça et."],
        "motive": ["Bu muazzam güçle bugün dağları devir, rakiplerini ezecek o efsanevi çalışmayı yap.", "İçindeki o fırtınayı serbest bırak, bugün o zorlu deneme senin karşında diz çökecek.", "Gücünün zirvesindesin, hiçbir şeyden çekinmeden en ağır konuların tam ortasına dal.", "Meydan oku, sınırları yak, bugün o masadan mutlak bir zaferle kalkmadan asla pes etme."],
    },
    "9": {
        "tukenmis": ["Hiç telaş yapma, sadece rahat bir koltukta arkana yaslanıp usulca konunu oku.", "Kendini hiç yorma, sakin bir müzik açarak sadece birkaç sayfa özet gözden geçir.", "Zoraki hiçbir şey yapma, bugün sadece en huzurlu hissettiğin dersin sayfalarını karıştır.", "Baskıyı tamamen kaldır, sadece kısa bir video izleyerek günü huzurla ve yavaşça bitir."],
        "isteksiz": ["Birden masaya atlama, yavaşça ve yumuşak bir geçişle, en kolay dersten usulca başla.", "Hiç acele etme, önce sevdiğin o içeceği hazırla ve sakince, adım adım masana geç.", "Zorlukları düşünme, sadece akışa kapılarak kendini sıkmadan ufak bir test çöz.", "Direnç gösterme, kendine şefkatli davranarak çok hafif bir başlangıçla konuya ısın."],
        "kaygili": ["Kaygılı hissettiğinde sadece nefesine odaklan, her şeyin kendi doğal hızında çözüleceğine güven.", "Panik yapma, en basit konudan başlayarak o sakin ve istikrarlı ritmini yeniden bul.", "Dış dünyanın baskısını kapat, kendi huzurlu kozana çekilerek yavaş ve emin adımlarla çalış.", "Sınav stresi yerine anın dinginliğine odaklan, sakince önündeki bir paragrafı oku."],
        "normal": ["Her zamanki telaşsız ama sarsılmaz istikrarınla, planına sakince uymaya devam et.", "Çalışmanı aceleye getirmeden, konuları sindire sindire kendi doğal ritminde ilerlet.", "Çevrendeki kaosu umursama, kendi iç huzurunu koruyarak yavaş ama emin adımlarla çalış.", "Sakin ve düzenli adımlarla her konuyu huzurlu bir akış içinde öğrenerek yola devam et."],
        "odakli": ["Bu güzel dinginlikle konunun en derinlerine in, o sessiz odakta her şeyi rahatça anla.", "Huzurlu bir akıştasın, hiç bozmadan o istikrarlı tempoyla sessizce soruları eritmeye devam et.", "Odağın o kadar sakin ki zamanın nasıl geçtiğini anlamadan masadaki konuları sakince bitir.", "Bu derin ve dingin uyumu yakalamışken, zihnine akan bilgileri hiç yorulmadan kabul et."],
        "motive": ["İçindeki o huzurlu gücü uyandır, bugün telaşsız ama devasa adımlarla çok büyük işler başar.", "Sakin bir okyanus gibisin, bu derin enerjiyle önüne çıkan tüm konuları yumuşakça aş.", "Neşeli ve dingin bir enerjiyle hiç yorulmadan bugün çok uzun bir çalışma seansı geçir.", "Sarsılmaz bir huzurla bugün tüm listeyi o yavaş ama kesin ritminle harika şekilde bitir."],
    },
}


RUH_KAPANIS_HAVUZ = {
    "tukenmis": ["Acele yok, küçük adımlarla gideceğiz.", "Bugünlük bu kadarı bile değerli, kendine iyi davran.", "Yavaşça ilerlemek de ilerlemektir, dinlen biraz."],
    "isteksiz": ["Yavaş yavaş, sen yapabilirsin.", "İlk adımı attın ya, gerisi gelir.", "Küçük bir başlangıç bile bugünü kurtarır."],
    "kaygili": ["Derin bir nefes al, her şey yoluna girecek.", "Sakin ol, adım adım üstesinden geleceğiz.", "Endişelenme, sen göründüğünden çok daha hazırsın."],
    "normal": ["İstikrarla devam et.", "Bu tempoyu koru, güzel gidiyorsun.", "Düzenli çalışman meyvesini verecek."],
    "odakli": ["Bu odağı koru, harika ilerliyorsun.", "Tam yolundasın, aynen devam.", "Bu netlikte çalışmaya devam et."],
    "motive": ["Hadi, başarı seni bekliyor!", "Bu enerjiyle dağları aşarsın!", "Tam gaz devam, bugün senin günün!"],
}


def _acilis_sec(tip_no, ruh_key):
    """Kisilik tipine ve ruh haline gore rastgele bir acilis cumlesi secer (cesitlilik)."""
    varyantlar = KISILIK_ACILIS_HAVUZ.get(str(tip_no), {}).get(ruh_key, [])
    if varyantlar:
        return random.choice(varyantlar)
    return "Bugün bu konuya odaklanalım."


def _kapanis_sec(ruh_key):
    """Ruh haline gore rastgele bir kapanis cumlesi secer."""
    varyantlar = RUH_KAPANIS_HAVUZ.get(ruh_key, [])
    if varyantlar:
        return random.choice(varyantlar)
    return "Devam et."


def _calisma_tarzi_sec(tip_no, ruh_key):
    """Kisilik tipine ve ruh haline gore rastgele bir calisma tarzi onerisi secer."""
    varyantlar = CALISMA_TARZI_HAVUZ.get(str(tip_no), {}).get(ruh_key, [])
    if varyantlar:
        return random.choice(varyantlar)
    return "Kendi temponda, sana en uygun yöntemle düzenli çalış."

def sablon_tavsiye(tip_no, ruh_key, zayif_konu, konu_teknigi, hedef):
    a = _acilis_sec(tip_no, ruh_key)
    k = _kapanis_sec(ruh_key)
    return f"{a} {zayif_konu} konusunda {konu_teknigi}. Bugünkü hedefin: {hedef}. {k}"


# =====================================================================
#  LATEX GUVENLI SARMA (model ciplak \frac yazarsa $...$ ile sarar)
# =====================================================================
def latex_guvenli_sar(metin):
    """Ciplak LaTeX'i $...$ ile sarar + ### temizler + okunabilirlik (adim/sonuc satira).
    v8: GUVENLI okunabilirlik - sadece net Adim/Sonuc/numara isaretleri satira, kelime YAPISMAZ."""
    if not metin:
        return metin
    metin = re.sub(r"#{2,}\s*", "", metin)
    metin = re.sub(r"\s*-{3,}\s*", "\n\n", metin)
    metin = re.sub(r"\s+(\*{0,2}Adım\s*\d+\s*[:.])", r"\n\n\1", metin)
    metin = re.sub(r"\s+(\*{0,2}Sonuç\s*[:.])", r"\n\n\1", metin)
    metin = re.sub(r"\s+(\d{1,2}\.\s)(?=[A-ZÇĞİÖŞÜ])", r"\n\n\1", metin)
    korunan = []
    def koru(m):
        korunan.append(m.group(0))
        return f"\x00K{len(korunan)-1}\x00"
    metin = re.sub(r"\$[^$\n]+\$", koru, metin)
    latex_parca = (
        r"\\left[\(\[\|].*?\\right[\)\]\|]"
        r"|\\frac\{[^{}]*\}\{[^{}]*\}"
        r"|\\sqrt\{[^{}]*\}"
        r"|\\(?:cdot|times|div|pm|mp|leq|geq|neq|sum|int|lim|sin|cos|tan|log|ln)\b"
        r"|[a-zA-Z0-9\)\}\]]\^\{?[a-zA-Z0-9]+\}?"
        r"|[a-zA-Z0-9]_\{?[a-zA-Z0-9]+\}?"
    )
    ada_deseni = rf"(?:{latex_parca})(?:[ =+\-*/().,]{{0,4}}(?:{latex_parca}))*"
    metin = re.sub(ada_deseni, lambda m: f"${m.group(0).strip()}$", metin)
    metin = re.sub(r"\x00K(\d+)\x00", lambda m: korunan[int(m.group(1))], metin)
    metin = re.sub(r"\$\$+", "$", metin)
    metin = re.sub(r"\$(\s+)\$", r"\1", metin)
    if metin.count("$") % 2 == 1:
        son = metin.rfind("$")
        metin = metin[:son] + metin[son+1:]
    metin = re.sub(r"\n{3,}", "\n\n", metin)
    return metin.strip()


# =====================================================================
#  CIKTI TEMIZLEME
# =====================================================================
def cikti_temizle(metin):
    if not metin or len(metin.strip()) < 15:
        return None
    metin = re.sub(r"\s+", " ", metin.strip().strip('"').strip()).strip()
    acilis = [r"^Sen[!,.]?\s+", r"^Hey[!,.]?\s+", r"^Selam[!,.]?\s+", r"^Merhaba[!,.]?\s+",
        r"^Sevgili [oö]ğrenci[!,.]?\s+", r"^[OÖ]ğrenci(ler)?[!,.]?\s+", r"^Kolay gelsin[!,.]?\s+",
        r"^Sana yardım etmek için buradayım[!,.]?\s+"]
    deg = True
    while deg:
        deg = False
        for k in acilis:
            y = re.sub(k, "", metin, flags=re.IGNORECASE)
            if y != metin:
                metin, deg = y.strip(), True
    for kesici in [r"[OÖ]nceki derslerde verdiğim.*", r"Bir tavsiyemi paylaşmak ister.*"]:
        metin = re.sub(kesici, "", metin, flags=re.IGNORECASE | re.DOTALL).strip()
    metin = re.sub(r"\s+", " ", metin).strip()
    cumleler = re.split(r"(?<=[.!?])\s+", metin)
    if len(cumleler) > 1:
        son = cumleler[-1].strip()
        if son and son[-1] not in ".!?":
            cumleler = cumleler[:-1]
        metin = " ".join(cumleler).strip()
    elif metin and metin[-1] not in ".!?":
        if "," in metin:
            metin = metin[:metin.rfind(",")].strip() + "."
        else:
            return None
    if len(metin) < 25:
        return None
    if len(metin.split()) < 8:
        return None
    return metin


# =====================================================================
#  ANA TAVSIYE (model + temizleme + sablon yedek)
# =====================================================================
def _tavsiye_metni(kisilik, ruh, zayif_konu, konu_teknigi, zorlanma="Orta"):
    tip_no = kisilik["no"]
    ruh_key = ruh["key"]
    acilis = _acilis_sec(tip_no, ruh_key)
    kapanis = _kapanis_sec(ruh_key)

    zorlanma_aciklama = {
        "Çok": "bu konuda çok zorlanıyor, temelden ve en basit kuraldan başlamalı",
        "Orta": "bu konuyu orta düzeyde biliyor, kuralın mantığını pekiştirmeli",
        "Az": "bu konuda az zorlanıyor, inceliklere ve ileri ayrıntılara odaklanmalı",
    }.get(zorlanma, "bu konuyu orta düzeyde biliyor")

    model_teknik = None
    try:
        prompt = (
            f"Bir YKS öğrencisinin '{zayif_konu}' konusunda nasıl çalışması gerektiğini açıkla.\n\n"
            f"Öğrencinin durumu: {zorlanma_aciklama}.\n\n"
            f"GÖREV: 4-6 cümlelik, '{zayif_konu}' konusunun ÖZÜNE giren teknik bir çalışma önerisi yaz. "
            f"Şunları içersin: (1) konunun temel kuralı/mantığı somut olarak nasıl işler, "
            f"(2) çözerken hangi adımı/yöntemi izlemeli, (3) bu konuda en sık yapılan hata veya dikkat edilmesi gereken incelik nedir. "
            f"Mümkünse küçük somut bir örnekle açıkla. "
            f"Motivasyon cümlesi, selamlama veya kişilik yorumu YAZMA — sadece konunun nasıl çalışılacağına dair teknik açıklama yaz. "
            f"Sayı verme (kaç soru gibi). Gevezelik etme, her cümle bilgi taşısın. Doğrudan konunun tekniğiyle başla."
        )
        messages = [
            {"role":"system","content":"Sen YKS konularını öğreten uzman bir öğretmensin. Bir konunun özüne inip, kuralının nasıl işlediğini somut ve teknik olarak açıklarsın. Süslü laflar değil, konunun mantığını anlatırsın."},
            {"role":"user","content":prompt},
        ]
        ham = model_uret(messages, max_new_tokens=520, temperature=0.7,
                         repetition_penalty=1.1)
        model_teknik = cikti_temizle(ham)
    except Exception:
        model_teknik = None

    if model_teknik is None:
        sablon = sablon_tavsiye(tip_no, ruh_key, zayif_konu, konu_teknigi, ruh["hedef"])
        return f"{acilis} {sablon} {kapanis}"

    hibrit = f"{acilis} {model_teknik} {kapanis}"
    return hibrit


# =====================================================================
#  CALISMA PLANI URETICI (model + sablon yedek)
# =====================================================================
def _calisma_plani_sablon(kisilik_no, ruh_key, zayif_konu, konu_teknigi, zorlanma="Orta"):
    tarz = _calisma_tarzi_sec(kisilik_no, ruh_key)
    zorlanma_carpan = {"Çok": 0.5, "Orta": 1.0, "Az": 1.6}.get(zorlanma, 1.0)
    vurgu_havuz = {
        "Çok":["Bu konuda çok zorlandığın için ÖNCE teoriyi sağlamlaştır. Soru sayısını düşük tuttum; önemli olan anlamak.",
               "Zorlandığını biliyorum — bugün acele yok. Temeli oturt, az ama doğru soru çöz.",
               "Çok zorlandığın için planı hafif tuttum. Önce kuralları kavra, sonra yavaşça soruya geç."],
        "Orta":["Bu konuyu orta düzeyde biliyorsun; teori tekrarı + bol soru dengeli gitmeli.",
                "Temeller tamam, şimdi pekiştirme zamanı: kısa tekrar + bol pratik.",
                "Orta seviyedesin; biraz tekrar, çokça soru ile bu konuyu kapatabilirsin."],
        "Az":["Bu konuda az zorlanıyorsun; teoriyi hızlı geç, bol ve zorlayıcı soruyla hız kazan.",
              "Konuya hâkimsin — bugün hız ve dikkat çalış, zor sorularla kendini test et.",
              "Az zorlanıyorsun, o yüzden teoriyi atla, doğrudan sınav temposunda soru çöz."],
    }
    vurgu = random.choice(vurgu_havuz.get(zorlanma, vurgu_havuz["Orta"]))
    ek_adim_havuz = {
        "Çok":["Çözemediğin her soruda çözümlü örneğe bak, mantığını anla, sonra benzerini dene.",
               "Takıldığın soruyu işaretle, video/çözüm bak, ertesi gün tekrar dene.",
               "Her yanlışta 'neden yanlış?' diye sor, o noktayı not defterine yaz."],
        "Orta":["Yanlışlarını konu bazında grupla, en çok hata yaptığın alt başlığa ekstra zaman ayır.",
                "Doğru-yanlış oranını takip et, zayıf kaldığın soru tipini tekrar çöz.",
                "Her blok sonunda yanlışlarını gözden geçir, aynı hatayı tekrarlama."],
        "Az":["Zorlayıcı/çeldiricili sorulara odaklan, süre tutarak çöz (sınav temposu).",
              "Hız çalış: her soruya max 1.5 dk ver, dikkat hatalarını kapat.",
              "Deneme sınavı mantığıyla çöz, süreyi ölç, gerçek sınava hazırlan."],
    }
    ek_adim = random.choice(ek_adim_havuz.get(zorlanma, ek_adim_havuz["Orta"]))

    acilis_havuz = [
        f"Bugünkü hedefin **{zayif_konu}** konusunu pekiştirmek.",
        f"Hadi **{zayif_konu}** konusuna odaklanalım.",
        f"Bugün **{zayif_konu}** üzerinde çalışıyoruz.",
        f"**{zayif_konu}** için planın hazır.",
    ]
    kapanis_havuz = [
        "Yarın aynı konudan 3-5 soru ile hızlı bir tekrar yap (unutma eğrisini kır).",
        "Yarın bu konuya 5 dakikalık kısa bir tekrarla başla, kalıcı olsun.",
        "Yatmadan önce bugün öğrendiğin 1 şeyi zihninde tekrar et.",
    ]
    acilis = random.choice(acilis_havuz)
    kapanis = random.choice(kapanis_havuz)

    return f"""### 📋 Bugünkü Çalışma Planın — {zayif_konu}
{acilis}

**🎯 Teknik:** {konu_teknigi}

**📊 Zorlanma seviyesi: {zorlanma}** — {vurgu}

**🧠 Sana önerilen çalışma tarzı:**
{tarz}

**✅ Kapanış adımları:**
1. {ek_adim}
2. Yanlışlarını ayrı bir deftere, hatanın *nedeniyle* birlikte not al
3. {kapanis}"""


def calisma_plani_uret(kisilik_no, ruh_key, zayif_konu, konu_teknigi, zorlanma="Orta"):
    """Plan koddan (zengin varyasyon havuzlu) anlik uretilir - hizli ve garantili.
    Varyasyon havuzlari sayesinde ayni girdide bile her uretimde farkli gorunur."""
    return _calisma_plani_sablon(kisilik_no, ruh_key, zayif_konu, konu_teknigi, zorlanma)


# =====================================================================
#  ENNEAGRAM TESTI + RADAR GRAFIK
# =====================================================================
TEST_SORULARI = [
    ("Her işi kurallara ve doğru yönteme göre yapmak benim için önemlidir.", "Tip 1 - Mükemmeliyetçi"),
    ("Bir şeyi yarım veya hatalı bırakmak beni rahatsız eder.", "Tip 1 - Mükemmeliyetçi"),
    ("Çalışırken 'doğru' ve 'yanlış' net çizgilerle ayrılmalı diye düşünürüm.", "Tip 1 - Mükemmeliyetçi"),
    ("Başkalarına yardım etmek ve onlar tarafından sevilmek bana iyi gelir.", "Tip 2 - Yardımsever"),
    ("Arkadaşlarımın bir konuyu anlamasına yardım etmek beni mutlu eder.", "Tip 2 - Yardımsever"),
    ("Birinin bana ihtiyacı olduğunu hissetmek motivasyonumu artırır.", "Tip 2 - Yardımsever"),
    ("Hedeflerime ulaşmak ve başarılı görünmek beni motive eder.", "Tip 3 - Başarı Odaklı"),
    ("Rakiplerimden veya sınıf ortalamasından önde olmak beni heyecanlandırır.", "Tip 3 - Başarı Odaklı"),
    ("Verimli çalışıp somut sonuç almak benim için her şeyden önemlidir.", "Tip 3 - Başarı Odaklı"),
    ("Kendimi özel ve farklı hissetmek, duygularımı derinden yaşamak isterim.", "Tip 4 - Bireyci"),
    ("Çalışırken ruh halim performansımı çok etkiler.", "Tip 4 - Bireyci"),
    ("Bir konuyu kendi özgün bakış açımla anlamlandırmayı severim.", "Tip 4 - Bireyci"),
    ("Bir konuyu derinlemesine anlamak ve bilgi toplamak beni tatmin eder.", "Tip 5 - Araştırmacı"),
    ("Kalabalık yerine tek başıma, sessizce çalışmayı tercih ederim.", "Tip 5 - Araştırmacı"),
    ("Bir şeyin 'neden' öyle olduğunu anlamadan ezberlemek bana anlamsız gelir.", "Tip 5 - Araştırmacı"),
    ("Güvenlik ararım ve bir plana bağlı olmak beni rahatlatır.", "Tip 6 - Sadakatçi"),
    ("Sınav öncesi en kötü ihtimalleri düşünüp ona göre hazırlanırım.", "Tip 6 - Sadakatçi"),
    ("Güvendiğim birinin onayını almak bana huzur verir.", "Tip 6 - Sadakatçi"),
    ("Yeni deneyimler, eğlence ve çeşitlilik peşinde koşarım.", "Tip 7 - Maceracı"),
    ("Aynı konuda uzun süre kalmak beni sıkar, farklı şeylere atlamayı severim.", "Tip 7 - Maceracı"),
    ("Çalışmayı oyunlaştırmak veya eğlenceli hale getirmek beni motive eder.", "Tip 7 - Maceracı"),
    ("Güçlü olmak ve kendi kontrolümü elimde tutmak benim için önemlidir.", "Tip 8 - Meydan Okuyan"),
    ("Zor bir konuyla karşılaşınca geri çekilmek yerine üzerine giderim.", "Tip 8 - Meydan Okuyan"),
    ("Birinin bana ne yapacağımı söylemesinden hoşlanmam.", "Tip 8 - Meydan Okuyan"),
    ("Huzuru korumak ve çatışmadan kaçınmak isterim.", "Tip 9 - Barışçıl"),
    ("Telaşsız, sakin bir tempoda çalışmak bana iyi gelir.", "Tip 9 - Barışçıl"),
    ("Kararsız kaldığımda genelde ortayı bulmaya çalışırım.", "Tip 9 - Barışçıl"),
]

def radar_grafik_ciz(puanlar_dict):
    tipler = [t.split(" - ")[1] for t in puanlar_dict.keys()]
    degerler = list(puanlar_dict.values())
    tipler_kapali = tipler + [tipler[0]]
    degerler_kapali = degerler + [degerler[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=degerler_kapali, theta=tipler_kapali, fill='toself',
        line=dict(color='rgb(99,102,241)', width=2),
        fillcolor='rgba(99,102,241,0.25)', name='Kişilik Profilin'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,5], tickvals=[1,2,3,4,5])),
        showlegend=False, title="🧠 Enneagram Kişilik Profilin",
        height=450, margin=dict(l=60, r=60, t=60, b=40)
    )
    return fig

def test_hesapla(*cevaplar):
    toplam = {}
    sayac = {}
    for (soru, tip), cevap in zip(TEST_SORULARI, cevaplar):
        try:
            puan = int(str(cevap).strip().split(" ")[0].split("-")[0])
        except (ValueError, IndexError):
            puan = 3
        toplam[tip] = toplam.get(tip, 0) + puan
        sayac[tip] = sayac.get(tip, 0) + 1
    puanlar = {tip: round(toplam[tip] / sayac[tip], 2) for tip in toplam}
    kazanan = max(puanlar, key=puanlar.get)
    fig = radar_grafik_ciz(puanlar)
    mesaj = f"### ✅ Sonuç: **{kazanan}**\nKişilik tipin 'Tavsiye' sekmesinde otomatik seçildi. Aşağıdaki radar grafiği tüm tiplerdeki eğilimini gösterir (her tip 3 sorunun ortalaması)."
    return gr.update(value=kazanan), mesaj, fig


def kt_gezin(yon, idx, secim, cevaplar):
    """Wizard gezinme: mevcut cevabi kaydet, idx'i yon (+1/-1) kadar kaydir,
    yeni soruyu/ilerlemeyi + buton gorunurluklerini dondur."""
    cevaplar = list(cevaplar)
    if secim is not None:
        cevaplar[idx] = secim
    N = len(TEST_SORULARI)
    yeni = max(0, min(N - 1, idx + yon))
    soru = TEST_SORULARI[yeni][0]
    son_mu = (yeni == N - 1)
    if son_mu:
        ilerleme = f"✅ **Son soru ({N}/{N})** — cevabını seç, aşağıdaki **Hesapla** butonuna bas."
    else:
        ilerleme = f"**Soru {yeni+1} / {N}**"
    return (
        yeni,
        cevaplar,
        gr.update(label=f"{yeni+1}. {soru}", value=cevaplar[yeni]),
        ilerleme,
        gr.update(visible=(yeni > 0)),
        gr.update(visible=not son_mu),
    )


def kt_hesapla(idx, secim, cevaplar):
    """Wizard'in topladigi 27 cevabi mevcut test_hesapla'ya verir (skor + radar ayni)."""
    cevaplar = list(cevaplar)
    if secim is not None:
        cevaplar[idx] = secim
    return test_hesapla(*cevaplar)


# =====================================================================
#  TAVSIYE + ORNEK SORU KARTLARI
# =====================================================================
def tam_tavsiye_uret(kisilik_label, ruh_label, zayif_konu, zorlanma="Orta"):
    if not zayif_konu or not zayif_konu.strip():
        return "Lütfen bir konu seç.", ""
    kisilik = ENNEAGRAM.get(kisilik_label, list(ENNEAGRAM.values())[0])
    ruh = RUH_HALI_TON.get(ruh_label, list(RUH_HALI_TON.values())[2])
    konu_teknigi = konu_teknigi_bul(zayif_konu)

    tavsiye = _tavsiye_metni(kisilik, ruh, zayif_konu, konu_teknigi, zorlanma)

    plan = calisma_plani_uret(kisilik["no"], ruh["key"], zayif_konu, konu_teknigi, zorlanma)

    return tavsiye, plan


# =====================================================================
#  RUH HALI TAKIBI (haftalik grafik)
# =====================================================================
def _tarih_sirala_anahtari(tarih_str):
    """Gun.Ay formatindaki tarihi siralama icin (ay, gun) tuple'ina cevirir.
    Parse edilemezse en sona atilir (buyuk deger)."""
    try:
        parcalar = tarih_str.strip().split(".")
        gun, ay = int(parcalar[0]), int(parcalar[1])
        return (ay, gun)
    except Exception:
        return (99, 99)

def ruh_hali_grafik(kayitlar):
    if not kayitlar:
        fig = go.Figure()
        fig.update_layout(title="Henüz kayıt yok. Aşağıdan ruh halini ekle.", height=350)
        return fig
    kayitlar_sirali = sorted(kayitlar, key=lambda k: _tarih_sirala_anahtari(k[0]))
    gunler = [k[0] for k in kayitlar_sirali]
    puanlar = [k[1] for k in kayitlar_sirali]
    etiketler = {1:"Tükenmiş",2:"İsteksiz",3:"Kaygılı",4:"Normal",5:"Odaklı",6:"Motive"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gunler, y=puanlar, mode='lines+markers',
        line=dict(color='rgb(99,102,241)', width=3),
        marker=dict(size=12, color='rgb(79,70,229)'),
        text=[etiketler.get(p,"") for p in puanlar], textposition="top center"
    ))
    fig.update_layout(
        title="📈 Haftalık Ruh Hali Takibin", height=400,
        yaxis=dict(range=[0.5,6.5], tickvals=[1,2,3,4,5,6],
                   ticktext=["😣 Tükenmiş","😟 İsteksiz","😰 Kaygılı","😐 Normal","🙂 Odaklı","🔥 Motive"]),
        xaxis=dict(type='category'),
        xaxis_title="Tarih", margin=dict(l=90,r=40,t=60,b=40)
    )
    return fig

def _ruh_secenekleri(kayitlar):
    """Silme dropdown'u icin 'tarih - ruh hali' etiketli secenekler uretir."""
    etiketler = {1:"😣 Tükenmiş",2:"😟 İsteksiz",3:"😰 Kaygılı",4:"😐 Normal",5:"🙂 Odaklı",6:"🔥 Motive"}
    kayitlar_sirali = sorted(kayitlar or [], key=lambda k: _tarih_sirala_anahtari(k[0]))
    return [f"{tarih} — {etiketler.get(puan,'')}" for tarih, puan in kayitlar_sirali], kayitlar_sirali

def ruh_ekle(tarih, ruh_label, kayitlar):
    if not tarih or not tarih.strip():
        tarih = datetime.now().strftime("%d.%m")
    puan = RUH_HALI_TON.get(ruh_label, {}).get("puan", 3)
    kayitlar = (kayitlar or []) + [[tarih.strip(), puan]]
    secenekler, _ = _ruh_secenekleri(kayitlar)
    return kayitlar, ruh_hali_grafik(kayitlar), "", gr.update(choices=secenekler, value=None)

def ruh_sil(secili_etiket, kayitlar):
    kayitlar = kayitlar or []
    if not secili_etiket:
        secenekler, _ = _ruh_secenekleri(kayitlar)
        return kayitlar, ruh_hali_grafik(kayitlar), gr.update(choices=secenekler, value=None)
    secenekler, kayitlar_sirali = _ruh_secenekleri(kayitlar)
    if secili_etiket in secenekler:
        idx = secenekler.index(secili_etiket)
        silinecek = kayitlar_sirali[idx]
        kayitlar = list(kayitlar)
        kayitlar.remove(silinecek)
    yeni_secenekler, _ = _ruh_secenekleri(kayitlar)
    return kayitlar, ruh_hali_grafik(kayitlar), gr.update(choices=yeni_secenekler, value=None)


# =====================================================================
#  DERS BASARI ANALIZI (ders ders D/Y/B girisi, yuzde+grup, grafik, AI tavsiye)
# =====================================================================
def basari_grubu(yuzde):
    """Yuzdeye gore basari grubu: 0-40 Zayif, 40-60 Orta, 60-75 Iyi, 75-90 Cok Iyi, 90+ Muthis."""
    if yuzde < 40:
        return "Zayıf", "🔴"
    elif yuzde < 60:
        return "Orta", "🟠"
    elif yuzde < 75:
        return "İyi", "🟡"
    elif yuzde < 90:
        return "Çok İyi", "🟢"
    else:
        return "Müthiş", "⭐"


def _basari_secenekleri(kayitlar):
    """Silme dropdown'u icin kayit etiketleri uretir."""
    kayitlar = kayitlar or []
    secenekler = []
    for k in kayitlar:
        secenekler.append(f"{k['tarih']} · {k['ders']} · %{k['yuzde']} ({k['grup']})")
    return secenekler, list(kayitlar)


def basari_ekle(ders, dogru, yanlis, bos, tarih, kayitlar):
    """Ders + D/Y/B + tarih alir, yuzde+net+grup hesaplar, kayda ekler. Grafik+dropdown gunceller."""
    try:
        d = int(dogru or 0); y = int(yanlis or 0); b = int(bos or 0)
    except (ValueError, TypeError):
        secenekler, _ = _basari_secenekleri(kayitlar)
        return kayitlar, "⚠️ Lütfen geçerli sayılar gir.", basari_grafik(kayitlar), gr.update(choices=secenekler, value=None)
    if d < 0 or y < 0 or b < 0:
        secenekler, _ = _basari_secenekleri(kayitlar)
        return kayitlar, "⚠️ Sayılar negatif olamaz.", basari_grafik(kayitlar), gr.update(choices=secenekler, value=None)
    toplam = d + y + b
    if toplam == 0:
        secenekler, _ = _basari_secenekleri(kayitlar)
        return kayitlar, "⚠️ En az bir soru sayısı gir.", basari_grafik(kayitlar), gr.update(choices=secenekler, value=None)
    net = d - y / 4
    yuzde = round(d / toplam * 100)
    grup, ikon = basari_grubu(yuzde)
    tarih = (str(tarih).strip() if tarih else "") or datetime.now().strftime("%d.%m")
    kayitlar = (kayitlar or []) + [{
        "ders": ders, "dogru": d, "yanlis": y, "bos": b,
        "toplam": toplam, "net": round(net, 2), "yuzde": yuzde,
        "grup": grup, "tarih": tarih,
    }]
    secenekler, _ = _basari_secenekleri(kayitlar)
    mesaj = f"{ikon} **{ders}** eklendi: %{yuzde} başarı ({grup}) · Net: {round(net, 2)} · {d}D {y}Y {b}B"
    return kayitlar, mesaj, basari_grafik(kayitlar), gr.update(choices=secenekler, value=None)


def basari_sil(secili_etiket, kayitlar):
    """Secili kaydi siler."""
    kayitlar = kayitlar or []
    if not secili_etiket:
        secenekler, _ = _basari_secenekleri(kayitlar)
        return kayitlar, basari_grafik(kayitlar), gr.update(choices=secenekler, value=None)
    secenekler, kayitlar_kopya = _basari_secenekleri(kayitlar)
    if secili_etiket in secenekler:
        idx = secenekler.index(secili_etiket)
        kayitlar = list(kayitlar)
        kayitlar.pop(idx)
    yeni_secenekler, _ = _basari_secenekleri(kayitlar)
    return kayitlar, basari_grafik(kayitlar), gr.update(choices=yeni_secenekler, value=None)


def _basari_tarih_anahtar(k):
    try:
        p = str(k.get("tarih", "")).split(".")
        return (int(p[1]), int(p[0]))
    except Exception:
        return (99, 99)


def basari_grafik(kayitlar):
    """Ders bazli basari yuzdesini cizer. Ayni ders birden fazla girilmisse zaman serisi olur."""
    kayitlar = kayitlar or []
    if not kayitlar:
        fig = go.Figure()
        fig.update_layout(
            title="Henüz kayıt yok. Aşağıdan ders sonucu ekle.",
            height=400, margin=dict(l=60, r=40, t=60, b=40),
            yaxis=dict(range=[0, 100], title="Başarı %"),
        )
        return fig
    dersler = {}
    for k in kayitlar:
        dersler.setdefault(k["ders"], []).append(k)
    fig = go.Figure()
    renkler = ["#6366f1", "#ec4899", "#14b8a6", "#f59e0b", "#8b5cf6",
               "#ef4444", "#10b981", "#3b82f6", "#f97316", "#a855f7"]
    for i, (ders, kyt) in enumerate(dersler.items()):
        kyt = sorted(kyt, key=_basari_tarih_anahtar)
        x = [f"{k['tarih']} ({j+1})" for j, k in enumerate(kyt)]
        y = [k["yuzde"] for k in kyt]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=ders,
            line=dict(color=renkler[i % len(renkler)], width=3),
            marker=dict(size=10),
        ))
    for esik, etiket in [(40, "Zayıf"), (60, "Orta"), (75, "İyi"), (90, "Çok İyi")]:
        fig.add_hline(y=esik, line_dash="dot", line_color="rgba(150,150,150,0.4)",
                      annotation_text=etiket, annotation_position="right")
    fig.update_layout(
        title="📊 Ders Bazlı Başarı Takibin",
        height=450, margin=dict(l=60, r=80, t=60, b=60),
        yaxis=dict(range=[0, 100], title="Başarı %"),
        xaxis_title="Kayıt (tarih)",
        showlegend=True,
    )
    return fig


def basari_tavsiye_al(kayitlar):
    """Girilen ders sonuclarina gore yapay zekadan calisma tavsiyesi alir."""
    kayitlar = kayitlar or []
    if not kayitlar:
        return "Önce yukarıdan en az bir ders sonucu ekle, sonra tavsiye alabilirsin."

    son_durum = {}
    for k in kayitlar:
        son_durum[k["ders"]] = k
    sirali = sorted(son_durum.values(), key=lambda k: k["yuzde"])

    ozet_satirlari = []
    for k in sirali:
        ozet_satirlari.append(f"- {k['ders']}: %{k['yuzde']} ({k['grup']}), {k['dogru']} doğru {k['yanlis']} yanlış {k['bos']} boş")
    ozet = "\n".join(ozet_satirlari)

    en_zayif = sirali[0]
    en_iyi = sirali[-1]

    try:
        prompt = (
            f"Bir YKS öğrencisinin ders bazlı deneme sonuçları aşağıda. "
            f"Bu sonuçlara bakarak ona kişiselleştirilmiş, somut bir çalışma tavsiyesi ver.\n\n"
            f"DENEME SONUÇLARI (zayıftan güçlüye):\n{ozet}\n\n"
            f"GÖREV: 4-6 cümlelik bir analiz ve tavsiye yaz:\n"
            f"1. En zayıf olduğu ders(ler)e ({en_zayif['ders']}, %{en_zayif['yuzde']}) öncelik vermesi gerektiğini söyle ve NEDEN somut belirt.\n"
            f"2. Yanlış sayısı yüksek derslerde dikkatli olması (net kaybı) konusunda uyar.\n"
            f"3. İyi olduğu dersi ({en_iyi['ders']}, %{en_iyi['yuzde']}) koruması için kısa öneri ver.\n"
            f"4. Hangi derse ne kadar ağırlık vermesi gerektiğine dair somut bir yönlendirme yap.\n\n"
            f"Sıcak, motive edici ama gerçekçi bir dille yaz. Madde işareti kullanma, akıcı paragraf yaz."
        )
        messages = [
            {"role": "system", "content": "Sen YKS öğrencilerinin deneme sonuçlarını analiz edip somut çalışma stratejisi öneren uzman bir koçsun. Verilere dayalı, gerçekçi ve uygulanabilir tavsiyeler verirsin."},
            {"role": "user", "content": prompt},
        ]
        ham = model_uret(messages, max_new_tokens=450, temperature=0.7,
                         repetition_penalty=1.1, adapter_kullan=False)
        tavsiye = cikti_temizle(ham)
        if tavsiye is None:
            tavsiye = _basari_sablon_tavsiye(sirali)
    except Exception:
        tavsiye = _basari_sablon_tavsiye(sirali)

    return f"### 🎯 Sana Özel Çalışma Tavsiyesi\n\n**Genel Durum:**\n{ozet}\n\n---\n\n{tavsiye}"


def _basari_sablon_tavsiye(sirali):
    """Model calismazsa kod tabanli yedek tavsiye."""
    en_zayif = sirali[0]
    en_iyi = sirali[-1]
    return (
        f"Şu an en çok dikkat etmen gereken ders **{en_zayif['ders']}** (%{en_zayif['yuzde']}, {en_zayif['grup']}). "
        f"Çalışma zamanının büyük kısmını buraya ayır; önce eksik konularını belirle, sonra bol soru çöz. "
        f"Yanlışların net kaybına yol açıyor, o yüzden emin olmadığın soruları boş bırakmayı da değerlendir. "
        f"**{en_iyi['ders']}** dersinde ise iyi durumdasın (%{en_iyi['yuzde']}); bu seviyeyi korumak için "
        f"haftada birkaç soru çözmen yeterli. Zayıf derse ağırlık ver, güçlü dersi ihmal etme; dengeli bir program kur."
    )


def ruh_basari_korelasyon(ruh_kayitlari, basari_kayitlari):
    """Ruh hali ve ders basarisini AYNI grafikte gosterir (cift Y ekseni, ortak tarih).
    Sol eksen: basari %, sag eksen: ruh hali (1-6). Iliskiyi gormeyi saglar."""
    ruh_kayitlari = ruh_kayitlari or []
    basari_kayitlari = basari_kayitlari or []

    if not ruh_kayitlari and not basari_kayitlari:
        fig = go.Figure()
        fig.update_layout(
            title="Henüz veri yok. Ruh hali ve ders sonucu ekledikçe ilişki burada görünecek.",
            height=400, margin=dict(l=60, r=60, t=60, b=40),
        )
        return fig

    ruh_tarih = {}
    for t, p in ruh_kayitlari:
        ruh_tarih.setdefault(t, []).append(p)
    ruh_ort = {t: sum(v) / len(v) for t, v in ruh_tarih.items()}
    basari_tarih = {}
    for k in basari_kayitlari:
        basari_tarih.setdefault(k["tarih"], []).append(k["yuzde"])
    basari_ort = {t: sum(v) / len(v) for t, v in basari_tarih.items()}

    tum_tarihler = sorted(set(list(ruh_ort.keys()) + list(basari_ort.keys())),
                          key=_tarih_sirala_anahtari)

    fig = go.Figure()
    basari_y = [basari_ort.get(t) for t in tum_tarihler]
    fig.add_trace(go.Scatter(
        x=tum_tarihler, y=basari_y, mode="lines+markers", name="Başarı %",
        line=dict(color="#22c55e", width=3), marker=dict(size=10),
        connectgaps=True, yaxis="y1",
    ))
    ruh_y = [ruh_ort.get(t) for t in tum_tarihler]
    fig.add_trace(go.Scatter(
        x=tum_tarihler, y=ruh_y, mode="lines+markers", name="Ruh Hali",
        line=dict(color="#6366f1", width=3, dash="dot"), marker=dict(size=10),
        connectgaps=True, yaxis="y2",
    ))
    fig.update_layout(
        title="📊 Ruh Hali ve Başarı İlişkisi",
        height=450, margin=dict(l=60, r=70, t=60, b=60),
        xaxis=dict(type="category", title="Tarih"),
        yaxis=dict(title="Başarı %", range=[0, 100], side="left",
                   titlefont=dict(color="#22c55e"), tickfont=dict(color="#22c55e")),
        yaxis2=dict(title="Ruh Hali", range=[0.5, 6.5], side="right", overlaying="y",
                    tickvals=[1, 2, 3, 4, 5, 6],
                    ticktext=["Tükenmiş", "İsteksiz", "Kaygılı", "Normal", "Odaklı", "Motive"],
                    titlefont=dict(color="#6366f1"), tickfont=dict(color="#6366f1")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =====================================================================
#  KOCLA SOHBET (acik uclu, deneysel) - serbest giris, model dogrudan cevap verir
#  - bos/cok kisa/hatali cevapta kullanici sessiz cokme yerine net mesaj gorur
# =====================================================================
SOHBET_SISTEM_PROMPT = (
    "Sen MindCoach adinda, YKS ve LGS'ye hazirlanan ogrencilere yardimci olan "
    "bir calisma kocusun. Sadece ders calisma teknikleri, calisma plani olusturma, "
    "motivasyon, sinav kaygisi ve konu calisma stratejileri hakkinda konusursun.\n\n"
    "KURALLAR:\n"
    "1. Ogrencinin SANA YAZDIGI MESAJA dogrudan cevap ver. Mesajda gecmeyen bir konu, "
    "ders veya plan UYDURMA.\n"
    "2. Ogrenci sadece selam verirse ('merhaba', 'selam' gibi) veya kisa/belirsiz bir "
    "sey yazarsa, sen de kisaca selam ver ve hangi ders/konuda yardim istedigini sor. "
    "Asla kendiliginden uzun bir calisma plani veya liste UYDURMA.\n"
    "3. Ogrenci somut bir konu, ders ya da plan isteği belirtirse, ancak o zaman somut "
    "ve uygulanabilir bir cevap ver (gerekirse maddeler halinde plan kurgula).\n"
    "4. Calisma/sinav konusuyla tamamen alakasiz bir sey sorulursa "
    "(orn. siyaset, kisisel veri), bunu nazikce belirtip calisma konusuna yonlendir.\n"
    "5. Kisa, sicak, dogrudan ve Turkce konus. Asiri uzun yazma; 2-4 cumle yeterli, "
    "sadece acikca plan istenirse kisa madde listesi kullan."
)

SOHBET_SELAMLAMA_KALIPLARI = {
    "merhaba", "selam", "selamlar", "merhabalar", "hey", "hi", "hello",
    "naber", "nasilsin", "nasılsın", "gunaydin", "günaydın", "iyi aksamlar",
    "iyi akşamlar", "kanka", "abi", "hocam",
}

SOHBET_SELAMLAMA_CEVABI = (
    "Selam! 👋 Ben MindCoach, çalışma koçun. Hangi derste veya konuda zorlanıyorsun, "
    "ya da sana nasıl bir çalışma planı hazırlamamı istersin?"
)

SOHBET_FALLBACK = (
    "Şu an bu soruna net bir cevap üretemedim 🙏 Lütfen ya soruyu biraz daha "
    "netleştirerek tekrar dene, ya da 2️⃣ **Tavsiye & Sorular** sekmesindeki "
    "yapılandırılmış asistanı kullan — o her durumda garanti bir cevap üretir."
)

def _yarim_cumleyi_at(metin):
    """Modelin token limitine takilip cumle ortasinda kesilen cikislarini temizler.
    Son cumle noktalama ile bitmiyorsa atar; hicbir tam cumle kalmazsa oldugu gibi birakir."""
    if not metin:
        return metin
    metin = re.sub(r"\s+", " ", metin.strip())
    cumleler = re.split(r"(?<=[.!?])\s+", metin)
    if len(cumleler) > 1:
        son = cumleler[-1].strip()
        if son and son[-1] not in ".!?:":
            cumleler = cumleler[:-1]
        metin = " ".join(cumleler).strip()
    return metin

def kocla_sohbet(mesaj, gecmis):
    gecmis = gecmis or []
    mesaj = (mesaj or "").strip()
    if not mesaj:
        return gecmis, gecmis, ""

    mesaj_sade = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ\s]", "", mesaj.lower()).strip()
    kelimeler = set(mesaj_sade.split())
    if len(mesaj_sade) <= 20 and kelimeler and kelimeler.issubset(SOHBET_SELAMLAMA_KALIPLARI | {""}):
        gecmis = gecmis + [[mesaj, SOHBET_SELAMLAMA_CEVABI]]
        return gecmis, gecmis, ""

    cevap = None
    try:
        messages = [{"role": "system", "content": SOHBET_SISTEM_PROMPT}]
        for onceki_mesaj, onceki_cevap in gecmis[-6:]:
            messages.append({"role": "user", "content": onceki_mesaj})
            messages.append({"role": "assistant", "content": onceki_cevap})
        messages.append({"role": "user", "content": mesaj})

        ham = model_uret(messages, max_new_tokens=360, temperature=0.6,
                         repetition_penalty=1.15, adapter_kullan=False)
        if len(ham) >= 15:
            cevap = _yarim_cumleyi_at(ham)
            if cevap and len(cevap) < 15:
                cevap = None
    except Exception:
        cevap = None

    if not cevap:
        cevap = SOHBET_FALLBACK

    gecmis = gecmis + [[mesaj, cevap]]
    return gecmis, gecmis, ""


# =====================================================================
#  GRADIO ARAYUZU
# =====================================================================
def konu_ozeti_goster(konu_adi):
    """Secili konunun PDF'ten okunan ozetini gosterir. Yoksa nazik uyari."""
    if not konu_adi:
        return "_Önce bir konu seç._"
    ozet = konu_ozeti_getir(konu_adi)
    if ozet:
        baslik = f"## 📖 {konu_adi}\n\n"
        return baslik + ozet
    return (
        f"## 📖 {konu_adi}\n\n"
        f"_Bu konunun özeti henüz yüklenmedi._\n\n"
        f"Konu özetleri Google Drive'daki PDF'lerden gelir. Bu konu için özet PDF'i "
        f"hazırlanıp `{OZET_PDF_KLASORU}` klasörüne eklendiğinde burada görünecek."
    )

def kc_cevap_degerlendir(konu_adi, kullanici_cevabi):
    """Kullanicinin konu hakkindaki cevabini/aciklamasini model degerlendirir."""
    kullanici_cevabi = (kullanici_cevabi or "").strip()
    if not konu_adi:
        return "_Önce bir konu seç._"
    if not kullanici_cevabi:
        return "_Önce yukarıdaki kutuya cevabını/açıklamanı yaz._"

    ozet = konu_ozeti_getir(konu_adi)
    baglam = f"\n\nKonunun doğru özeti (referans):\n{ozet[:800]}" if ozet else ""

    try:
        prompt = (
            f"Bir YKS öğrencisi '{konu_adi}' konusu hakkında aşağıdaki açıklamayı/cevabı yazdı. "
            f"Öğretmen gözüyle değerlendir.{baglam}\n\n"
            f"ÖĞRENCİNİN CEVABI:\n{kullanici_cevabi}\n\n"
            f"DEĞERLENDİRME ŞÖYLE OLSUN:\n"
            f"1. **Doğru olanlar:** Öğrencinin doğru bildiği noktaları belirt (varsa).\n"
            f"2. **Eksik/Yanlış olanlar:** Eksik veya hatalı noktaları nazikçe düzelt.\n"
            f"3. **Tavsiye:** Bu konuyu daha iyi öğrenmesi için 1-2 cümlelik öneri.\n\n"
            f"Sıcak, cesaretlendirici ve öğretici bir dille yaz. Matematik ifadelerini $...$ ile yaz."
        )
        messages = [
            {"role": "system", "content": "Sen sabırlı, yapıcı bir YKS öğretmenisin. Öğrencilerin cevaplarını değerlendirip eksiklerini nazikçe gösterirsin, asla küçümsemezsin."},
            {"role": "user", "content": prompt},
        ]
        ham = model_uret(messages, max_new_tokens=500, temperature=0.6,
                         repetition_penalty=1.1, adapter_kullan=False)
        ham = ham.strip()
        if len(ham) >= 30:
            return f"### 🤖 Değerlendirme\n\n{ham}"
    except Exception:
        pass
    return (
        "### 🤖 Değerlendirme\n\n"
        "_Şu an değerlendirme üretemedim. Model yüklü mü kontrol et veya tekrar dene._"
    )


def ders_secilince(ders):
    konular = DERS_KONULARI.get(ders, [])
    return gr.update(choices=konular, value=konular[0] if konular else None)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title="Datagram MindCoach") as demo:
    gr.Markdown("""# 🧠 Datagram MindCoach
### Kişiliğine, ruh haline ve zayıf konuna göre sana özel YKS çalışma asistanı
Enneagram kişilik tipin + bugünkü ruh halin + zorlandığın konu birleşir; sana özel **tavsiye**, **çalışma planı** ve **örnek sorular** üretilir.""")

    _model_yuklu_mu = ("tokenizer" in globals() and "model" in globals())
    if not _model_yuklu_mu:
        gr.Markdown(
            "### ⚠️ UYARI: `model` / `tokenizer` bu oturumda tanımlı değil.\n"
            "Şu an gördüğün tüm tavsiyeler **şablon (fallback)** çıktısıdır, gerçek Qwen3 model "
            "çıktısı DEĞİLDİR. Önce model yükleme hücresini (`FastLanguageModel.from_pretrained(...)`) "
            "çalıştır, sonra bu hücreyi tekrar çalıştır."
        )

    ruh_state = gr.State([])
    soru_state = gr.State([])

    with gr.Tab("🧭 Kişilik Testi"):
        gr.Markdown("Sorular tek tek gelir. Her ifade için sana en uygun seçeneği işaretle, **İleri** ile devam et; son soruda **Hesapla**'ya bas. (Dokunmadığın soru 'Kararsızım' sayılır.)")
        TEST_SECENEKLERI = [
            "1 - Hiç katılmıyorum",
            "2 - Pek katılmıyorum",
            "3 - Kararsızım",
            "4 - Katılıyorum",
            "5 - Tamamen katılıyorum",
        ]
        _KT_N = len(TEST_SORULARI)
        kt_idx_state = gr.State(0)
        kt_cevap_state = gr.State(["3 - Kararsızım"] * _KT_N)
        kt_ilerleme = gr.Markdown(f"**Soru 1 / {_KT_N}**")
        kt_soru_radio = gr.Radio(
            TEST_SECENEKLERI, value="3 - Kararsızım",
            label=f"1. {TEST_SORULARI[0][0]}",
        )
        with gr.Row():
            kt_geri_btn = gr.Button("◀ Geri", visible=False)
            kt_ileri_btn = gr.Button("İleri ▶", variant="primary")
        test_btn = gr.Button("🧠 Kişilik Tipimi Hesapla", variant="primary", size="lg")
        test_sonuc = gr.Markdown()
        test_grafik = gr.Plot()

    with gr.Tab("💡 Tavsiye"):
        with gr.Row():
            kisilik_dd = gr.Dropdown(list(ENNEAGRAM.keys()), label="🧠 Kişilik Tipin", value="Tip 4 - Bireyci")
            ruh_dd = gr.Dropdown(list(RUH_HALI_TON.keys()), label="💭 Bugünkü Ruh Halin", value="😐 Normal")
        with gr.Row():
            ders_dd = gr.Dropdown(list(DERS_KONULARI.keys()), label="📚 Ders", value="TYT Matematik")
            konu_dd = gr.Dropdown(DERS_KONULARI["TYT Matematik"], label="📖 Zayıf Konun", value="Çarpanlara Ayırma")
        zorlanma_dd = gr.Dropdown(["Az", "Orta", "Çok"], label="📊 Bu Konuda Ne Kadar Zorlanıyorsun?", value="Orta")
        uret_btn = gr.Button("✨ Bana Özel Çalışma Paketi Hazırla", variant="primary", size="lg")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 💬 Sana Özel Tavsiye")
                cikti_tavsiye = gr.Markdown(latex_delimiters=[{"left": "$", "right": "$", "display": False}])
            with gr.Column():
                cikti_plan = gr.Markdown(latex_delimiters=[{"left": "$", "right": "$", "display": False}])

    with gr.Tab("📚 Konu Çalış"):
        gr.Markdown(
            "### 📚 Konu Özeti, Soru Çözümü ve Pratik\n"
            "Bir ders ve konu seç; özeti oku, örnek soru çöz, detaylı çözüm al "
            "veya anladığını yazıp yapay zekaya değerlendirt."
        )
        with gr.Row():
            kc_ders_dd = gr.Dropdown(list(DERS_KONULARI.keys()), label="📚 Ders", value="TYT Matematik")
            kc_konu_dd = gr.Dropdown(DERS_KONULARI["TYT Matematik"], label="📖 Konu", value="Çarpanlara Ayırma")
        kc_ozet_btn = gr.Button("📖 Konu Özetini Göster", variant="primary", size="lg")
        kc_ozet_cikti = gr.Markdown(
            "_Bir konu seçip 'Konu Özetini Göster'e bas._",
            latex_delimiters=[{"left": "$", "right": "$", "display": False}]
        )
        gr.Markdown("---")
        gr.Markdown("#### 📝 Örnek Soru Çöz")
        with gr.Row():
            soru_adet = gr.Slider(1, 10, value=3, step=1, label="Kaç soru?")
            soru_zorluk = gr.Dropdown(["Kolay", "Orta", "Zor"], label="Zorluk", value="Orta")
        soru_btn = gr.Button("📝 Soru Üret", variant="secondary", size="lg")
        with gr.Row():
            with gr.Column(scale=1):
                cikti_sorular = gr.Markdown(latex_delimiters=[{"left": "$", "right": "$", "display": False}])
                daha_fazla_btn = gr.Button("➕ Daha Fazla Soru Üret", variant="secondary", visible=False)
                with gr.Row(visible=False) as detay_row:
                    detay_dd = gr.Dropdown(choices=[], label="🔍 Hangi soruyu detaylı çözeyim?", value=None, scale=3)
                    detay_btn = gr.Button("Detaylı Çöz", variant="secondary", scale=1)
            with gr.Column(scale=1):
                cikti_detay = gr.Markdown(
                    "_🔍 Bir soru seçip 'Detaylı Çöz'e bastığında, adım adım çözüm burada görünecek._",
                    latex_delimiters=[{"left": "$", "right": "$", "display": False}]
                )
        gr.Markdown("---")
        gr.Markdown(
            "#### ✍️ Kendini Test Et\n"
            "Bu konuyla ilgili öğrendiklerini kendi cümlelerinle yaz (veya bir soruya cevabını yaz), "
            "yapay zeka değerlendirip eksiklerini söylesin."
        )
        kc_cevap_kutu = gr.Textbox(
            label="Cevabın / Açıklaman",
            placeholder="Örn: Çarpanlara ayırmada iki kare farkı şöyle kullanılır...",
            lines=4
        )
        kc_degerlendir_btn = gr.Button("🤖 Değerlendir", variant="secondary")
        kc_degerlendirme_cikti = gr.Markdown(
            latex_delimiters=[{"left": "$", "right": "$", "display": False}]
        )

    with gr.Tab("📈 Ruh Hali Takibi"):
        gr.Markdown("Her gün ruh halini ekle, haftalık değişimini grafikte gör.")
        with gr.Row():
            ruh_tarih = gr.Textbox(label="📅 Tarih (örn: 19.06)", placeholder="boş bırakırsan bugün")
            ruh_secim = gr.Dropdown(list(RUH_HALI_TON.keys()), label="💭 Bugünkü Ruh Halin", value="😐 Normal")
        ruh_ekle_btn = gr.Button("➕ Ekle", variant="primary")
        ruh_grafik = gr.Plot()
        gr.Markdown("**Yanlışlıkla eklediğin bir kaydı silmek için aşağıdan seç:**")
        with gr.Row():
            ruh_sil_dd = gr.Dropdown(choices=[], label="🗑️ Silinecek Kayıt", value=None)
            ruh_sil_btn = gr.Button("Sil", variant="stop")

    with gr.Tab("📊 Ders Başarı Analizi"):
        gr.Markdown(
            "### 📊 Ders Başarı Analizi\n"
            "Çözdüğün deneme/testlerin sonuçlarını ders ders gir; başarı yüzden hesaplanır, "
            "zaman içindeki değişimin grafiğe işlenir. Sonra yapay zekadan sana özel çalışma "
            "tavsiyesi alabilirsin.\n\n"
            "**Başarı grupları:** 🔴 Zayıf (0-40) · 🟠 Orta (40-60) · 🟡 İyi (60-75) · "
            "🟢 Çok İyi (75-90) · ⭐ Müthiş (90-100)"
        )
        with gr.Row():
            basari_ders_dd = gr.Dropdown(list(DERS_KONULARI.keys()), label="📚 Ders", value="TYT Matematik", scale=3)
            basari_tarih = gr.Textbox(label="📅 Tarih (gg.aa)", value=datetime.now().strftime("%d.%m"), scale=1)
        with gr.Row():
            basari_dogru = gr.Number(label="✅ Doğru", value=0, minimum=0, precision=0)
            basari_yanlis = gr.Number(label="❌ Yanlış", value=0, minimum=0, precision=0)
            basari_bos = gr.Number(label="⚪ Boş", value=0, minimum=0, precision=0)
        basari_ekle_btn = gr.Button("➕ Sonucu Ekle", variant="primary", size="lg")
        basari_durum = gr.Markdown()
        basari_grafik_alani = gr.Plot()
        with gr.Row():
            basari_sil_dd = gr.Dropdown(choices=[], label="🗑️ Silinecek Kayıt", value=None, scale=3)
            basari_sil_btn = gr.Button("Sil", variant="stop", scale=1)
        gr.Markdown("---")
        basari_tavsiye_btn = gr.Button("🎯 Bu Sonuçlara Göre Çalışma Tavsiyesi Al", variant="secondary", size="lg")
        basari_tavsiye_cikti = gr.Markdown()
        gr.Markdown("---")
        gr.Markdown(
            "#### 📊 Ruh Hali ve Başarı İlişkisi\n"
            "Ruh Hali Takibi sekmesinde eklediğin ruh halleri ile buradaki ders başarın "
            "aynı grafikte gösterilir. Böylece *kötü ruh halinde başarımın düşüp düşmediğini* "
            "aynı tarih üzerinden görebilirsin. (Yeşil çizgi: başarı %, mor noktalı çizgi: ruh hali)"
        )
        korelasyon_btn = gr.Button("📊 İlişkiyi Göster", variant="secondary")
        korelasyon_grafik = gr.Plot()
        basari_state = gr.State([])

    with gr.Tab("💬 Koçla Sohbet"):
        gr.Markdown(
            "Açık uçlu sohbet — istediğin konuyu, çalışma planını veya zorlandığın bir şeyi "
            "doğrudan koçuna yaz, sana yardımcı olsun."
        )
        sohbet_state = gr.State([])
        sohbet_kutu = gr.Chatbot(label="MindCoach", height=420)
        with gr.Row():
            sohbet_giris = gr.Textbox(
                label="", placeholder="Örn: Bana 3 günlük bir türev çalışma planı yaz",
                scale=4, show_label=False,
            )
            sohbet_gonder = gr.Button("Gönder", variant="primary", scale=1)

    kt_ileri_btn.click(
        lambda idx, secim, cevaplar: kt_gezin(1, idx, secim, cevaplar),
        inputs=[kt_idx_state, kt_soru_radio, kt_cevap_state],
        outputs=[kt_idx_state, kt_cevap_state, kt_soru_radio, kt_ilerleme, kt_geri_btn, kt_ileri_btn],
    )
    kt_geri_btn.click(
        lambda idx, secim, cevaplar: kt_gezin(-1, idx, secim, cevaplar),
        inputs=[kt_idx_state, kt_soru_radio, kt_cevap_state],
        outputs=[kt_idx_state, kt_cevap_state, kt_soru_radio, kt_ilerleme, kt_geri_btn, kt_ileri_btn],
    )
    test_btn.click(
        kt_hesapla,
        inputs=[kt_idx_state, kt_soru_radio, kt_cevap_state],
        outputs=[kisilik_dd, test_sonuc, test_grafik],
    )
    ders_dd.change(ders_secilince, inputs=ders_dd, outputs=konu_dd)
    uret_btn.click(tam_tavsiye_uret, inputs=[kisilik_dd, ruh_dd, konu_dd, zorlanma_dd], outputs=[cikti_tavsiye, cikti_plan])
    kc_ders_dd.change(ders_secilince, inputs=kc_ders_dd, outputs=kc_konu_dd)
    kc_ozet_btn.click(konu_ozeti_goster, inputs=[kc_konu_dd], outputs=[kc_ozet_cikti])
    soru_btn.click(
        ornek_soru_uret_buton, inputs=[kc_konu_dd, soru_adet, soru_zorluk],
        outputs=[cikti_sorular, soru_state, daha_fazla_btn]
    ).then(
        soru_secenekleri_guncelle, inputs=[soru_state], outputs=[detay_dd]
    ).then(
        lambda s: gr.update(visible=bool(s)), inputs=[soru_state], outputs=[detay_row]
    )
    daha_fazla_btn.click(
        daha_fazla_soru_uret, inputs=[kc_konu_dd, soru_adet, soru_zorluk, soru_state],
        outputs=[cikti_sorular, soru_state, daha_fazla_btn]
    ).then(
        soru_secenekleri_guncelle, inputs=[soru_state], outputs=[detay_dd]
    )
    detay_btn.click(detayli_cozum_uret, inputs=[detay_dd, soru_state], outputs=[cikti_detay])
    kc_degerlendir_btn.click(kc_cevap_degerlendir, inputs=[kc_konu_dd, kc_cevap_kutu], outputs=[kc_degerlendirme_cikti])
    ruh_ekle_btn.click(ruh_ekle, inputs=[ruh_tarih, ruh_secim, ruh_state], outputs=[ruh_state, ruh_grafik, ruh_tarih, ruh_sil_dd])
    ruh_sil_btn.click(ruh_sil, inputs=[ruh_sil_dd, ruh_state], outputs=[ruh_state, ruh_grafik, ruh_sil_dd])
    basari_ekle_btn.click(
        basari_ekle,
        inputs=[basari_ders_dd, basari_dogru, basari_yanlis, basari_bos, basari_tarih, basari_state],
        outputs=[basari_state, basari_durum, basari_grafik_alani, basari_sil_dd]
    )
    basari_sil_btn.click(
        basari_sil, inputs=[basari_sil_dd, basari_state],
        outputs=[basari_state, basari_grafik_alani, basari_sil_dd]
    )
    basari_tavsiye_btn.click(basari_tavsiye_al, inputs=[basari_state], outputs=[basari_tavsiye_cikti])
    korelasyon_btn.click(ruh_basari_korelasyon, inputs=[ruh_state, basari_state], outputs=[korelasyon_grafik])
    sohbet_gonder.click(kocla_sohbet, inputs=[sohbet_giris, sohbet_state], outputs=[sohbet_kutu, sohbet_state, sohbet_giris])
    sohbet_giris.submit(kocla_sohbet, inputs=[sohbet_giris, sohbet_state], outputs=[sohbet_kutu, sohbet_state, sohbet_giris])

try:
    _yuklenen, _mesaj = tum_ozetleri_yukle()
    print(_mesaj)
except Exception as _e:
    print(f"Konu özetleri yüklenemedi (sorun değil, Drive bağlı değilse normal): {_e}")

demo.launch(share=True)
