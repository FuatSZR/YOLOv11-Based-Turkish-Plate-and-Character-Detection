# 🇹🇷 Gelişmiş Türk Plaka Tanıma Sistemi (Advanced Turkish License Plate Recognition - ANPR)

**Etiketler:** `YOLOv11` `YOLOv8` `Streamlit` `Computer-Vision` `ANPR` `License-Plate-Recognition`

Bu proje, görüntü ve videolardaki Türk plakalarını yüksek doğrulukla tespit eden ve okuyan uçtan uca (end-to-end) bir Otomatik Plaka Tanıma (OPT / ANPR) sistemidir. Geliştirilen çift aşamalı YOLO mimarisi, kullanıcı dostu bir arayüz ile **Streamlit** üzerinden gerçek zamanlı olarak sunulur.

## 🌟 Öne Çıkan Özellikler

  * **Çift Aşamalı YOLO Mimarisi:** Plaka tespiti için **YOLOv11n** ve karakter tanıma için **YOLOv8n** olmak üzere iki ayrı optimize edilmiş model kullanılır.
  * **Web Uygulaması (Streamlit):** Kullanıcının resim yükleyerek anlık sonuç alabileceği, temiz ve etkileşimli bir web arayüzü sunar.
  * **Yüksek Doğruluk:** Türkiye plaka standardına özel verilerle eğitilmiş, yerelleştirilmiş çözümdür.
  * **Uçtan Uca Çözüm:** Tespitten (Detection) okumaya (Reading) kadar tüm süreç otomatikleştirilmiştir.

-----

## 🏗️ Proje Mimarisi ve Çalışma Prensibi

Sistem, ana Streamlit uygulaması (**`main.py`**) tarafından koordine edilen, sırasıyla **Plaka Tespiti** ve **Karakter Tanıma** olmak üzere iki ana aşamada çalışır.

### 📂 Proje Dosya Yapısı

```
.
├── main.py                   # Streamlit ana uygulama (UI) ve iş akışı koordinasyonu
├── helper.py                 # Arka plan algılama ve okuma fonksiyonlarını içerir
├── requirements.txt          # Tüm Python bağımlılıkları
├── models/
│   ├── plate_detection.pt    # Aşama 1: Plaka tespit modeli (YOLOv11n ağırlıkları)
│   └── plate_read.pt         # Aşama 2: Karakter tanıma modeli (YOLOv8n ağırlıkları)
└── README.md                 # Bu dosya
```

### 🎯 Aşama 1: Plaka Tespiti (Plate Detection)

Bu aşama, giriş görüntüsündeki plaka bölgelerini izole etmeyi hedefler.

| Özellik | Detay |
| :--- | :--- |
| **Model** | **YOLOv11n** (Nano) |
| **Amaç** | Görüntü içindeki plakanın Bounding Box koordinatlarını bulmak. |
| **Eğitim Verisi** | Kaggle - [Turkish License Plate Dataset] |
| **Eğitim Parametreleri** | `epochs=50`, `imgsz=640`, `batch=16` |
| **Çıktı** | Tespit edilen plaka bölgelerinin kırpılmış (cropped) görüntüleri. |

### 📝 Aşama 2: Plaka Okuma (Character Recognition)

Kırpılmış plaka görüntüleri üzerinde çalışarak her bir karakteri tanır ve sıralar.

| Özellik | Detay |
| :--- | :--- |
| **Model** | **YOLOv8n** (Nano) |
| **Amaç** | Plaka üzerindeki her bir harfi ve rakamı sınıflandırmak. |
| **Eğitim Parametreleri** | `epochs=50`, `imgsz=640`, `batch=16` |
| **Post-Processing** | Karakterler, doğru plaka metnini oluşturmak için **x-ekseni koordinatlarına göre** sıralanır. |
| **Nihai Çıktı** | Plakanın metin hali (`AB34ABC`). |

-----

## 💻 Kurulum ve Kullanım

Projenin yerel makinenizde çalıştırılması için aşağıdaki adımları izleyin.

### Adım 1: Depoyu Klonlama

```bash
git clone <depo-adresiniz>
cd <proje-klasörü>
```

### Adım 2: Bağımlılıkları Yükleme

Proje, `ultralytics` ve `streamlit` dahil olmak üzere gerekli tüm kütüphaneleri `requirements.txt` dosyası üzerinden yönetir.

```bash
pip install -r requirements.txt
```

### Adım 3: Model Ağırlıklarını Yerleştirme

Eğittiğiniz model ağırlıklarını, `main.py` dosyasında tanımlı yollara uygun olarak **`models/`** klasörüne yerleştirin.

| Model Ağırlığı | Konum |
| :--- | :--- |
| Plaka Tespiti | `models/plate_detection.pt` |
| Karakter Tanıma | `models/plate_read.pt` |

### Adım 4: Streamlit Uygulamasını Başlatma

Ana uygulamayı Streamlit komutu ile çalıştırın:

```bash
streamlit run main.py
```

Uygulama, yerel sunucunuzda (genellikle `http://localhost:8501`) otomatik olarak açılacaktır.

### 🖼️ Uygulama Kullanımı

1.  Uygulama arayüzündeki **"Upload an image"** bölümünü kullanarak bir resim yükleyin (`.png`, `.jpg`, `.jpeg`).
2.  Uygulama, resmi yüklendikten sonra otomatik olarak **`detect_plate`** fonksiyonunu çağırır.
3.  Tespit edilen plaka (varsa), kırpılmış görüntü olarak **"Result"** başlığı altında gösterilir.
4.  Kırpılan görüntü, **`plate_read`** fonksiyonu ile karakterlere ayrılır ve nihai metin **"Plate Text"** başlığı altında görüntülenir.

-----

## 🛠️ Temel Kod Blokları

Projenin temel iş mantığı, **`helper.py`** dosyasındaki iki ana fonksiyon üzerine kuruludur:

### 1\. `detect_plate(image, plate_detction_model)`

Görüntüdeki plaka kutularını tespit eder ve kırparak döndürür.

### 2\. `plate_read(plate_imgs, plate_read_model)`

Gelen kırpılmış plaka görüntülerini işler, karakterleri tanır ve x koordinatına göre sıralayarak okunabilir metne dönüştürür.

```python
# helper.py içinden karakter sıralama mantığı
# ...
for result in results.boxes.data.tolist():
    # Karakter koordinatlarını ve sınıfını al
    x1, y1, x2, y2, score, class_id = result
    class_name = model.names[class_id]
    
    # x koordinatına ve karaktere göre sıralamak için veri yapısına ekle
    # (Bu kısım, helper.py'daki sıralama mekanizmasının temelini oluşturur)
    plate_text_location.append((x1, class_name)) 

# x1 (sol koordinat) değerine göre sıralama
plate_text_location.sort(key=lambda x: x[0]) 

# Sıralanmış karakterleri birleştirerek plaka metnini oluştur
# ...
```

## 💡 Gelecek Geliştirme Fikirleri

Mevcut proje, çift aşamalı YOLO mimarisi ve Streamlit arayüzü ile güçlü bir temel sunmaktadır. Doğruluğu, hızı ve işlevselliği daha da artırmak için aşağıdaki geliştirmeler yapılabilir:

### 1. Karakter Tanıma Doğruluğunu Artırma (YOLOv11L Varyantı)

Projenin en önemli geliştirme alanı, **Plaka Okuma (Aşama 2)** modelinin performansını artırmaktır.

* **Model Yükseltme:** Mevcut **YOLOv8n** (Nano) modelini, daha yüksek doğruluk potansiyeli sunan **YOLOv11L** (Large) veya **YOLOv11X** (Extra Large) gibi daha güçlü bir YOLO varyantı ile değiştirmek. Daha büyük bir model, daha karmaşık karakter biçimlerini ve düşük çözünürlüklü görüntüleri daha iyi işleyebilir.
* **Özel Türkçe Karakter Veri Seti:** Karakter tanıma modelini eğitmek için, **yalnızca Türkçe plakalardan kırpılmış, zenginleştirilmiş ve özel olarak etiketlenmiş** büyük bir veri seti kullanılmalıdır. Bu, modelin özellikle zorlu koşullarda (eğik, kirli, hasarlı plakalar) bile yüksek doğrulukta sonuç vermesini sağlayacaktır.

### 2. Gerçek Zamanlı Video Akışı Desteği

* Mevcut Streamlit uygulamasını, yüklenen görüntüler yerine **webcam** veya **RTSP/IP kamera** akışını doğrudan işleyebilecek şekilde genişletmek. Bu, sistemi gerçek bir gözetim veya trafik yönetimi senaryosunda kullanıma uygun hale getirir.

### 3. Hız Optimizasyonu ve Model Dönüşümü

* Daha hızlı çıkarım (inference) süreleri elde etmek için, eğitilen PyTorch ağırlıklarını **ONNX** veya **TensorRT** gibi daha hafif formatlara dönüştürmek ve bu optimize edilmiş modelleri Streamlit uygulamasında kullanmak. Bu, özellikle CPU tabanlı ortamlarda performansı ciddi ölçüde artırabilir.

### 4. Plaka Veritabanı Entegrasyonu

* Tanınan plaka metinlerini (örneğin `34ABC123`), bir **SQL veritabanı** ile entegre etmek. Bu, sistemin plaka numarasını bir kayda (araç sahibi, giriş zamanı vb.) bağlamasını sağlayarak erişim kontrolü veya trafik izleme uygulamaları için tam teşekküllü bir çözüm haline gelmesini sağlar.
