# 🚗 Smart License Plate Expiration Detection

Sistem deteksi masa berlaku plat nomor kendaraan Indonesia menggunakan **metode Computer Vision klasik** (tanpa deep learning). Pipeline membaca nomor plat dan tanggal kedaluwarsa (MM.YY) dari foto, lalu menentukan status **EXPIRED** atau **VALID**.

🔗 **Live Demo:** https://lisence-plate-detector-2z9sxpciywtobpapsjufwm.streamlit.app/

---

## 📋 Daftar Isi
- [Gambaran Umum](#-gambaran-umum)
- [Pipeline](#-pipeline)
- [Struktur Project](#-struktur-project)
- [Instalasi](#️-instalasi)
- [Menjalankan Aplikasi](#️-menjalankan-aplikasi)
- [Melatih Model](#-melatih-model)
- [Evaluasi](#-evaluasi)
- [Hasil](#-hasil)
- [Teknologi](#️-teknologi)
- [Tim](#-tim)

---

## 🎯 Gambaran Umum

Aplikasi ini mendeteksi apakah plat nomor kendaraan sudah kedaluwarsa berdasarkan tanggal MM.YY yang tercetak di plat. Seluruh proses menggunakan teknik Computer Vision klasik:

- **Plate detection** — lokalisasi plat dengan Canny edge + analisis kontur
- **Character segmentation** — pemisahan karakter via connected components & projection profile
- **OCR** — pengenalan karakter dengan fitur **HOG** + classifier **SVM**
- **Expiry parsing** — membaca MM.YY dan membandingkan dengan tanggal saat ini

---

## 🔄 Pipeline

```
Input Image
    │
    ▼
┌─────────────────┐
│ Preprocessing   │  Grayscale + Gaussian blur
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Plate Detection │  Canny edge → contour → uniform char-run
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Segmentation    │  Number row + expiry row (dual threshold)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ OCR (HOG + SVM) │  Per-character recognition
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Expiry Parsing  │  MM.YY → compare with today
└─────────────────┘
    │
    ▼
  EXPIRED / VALID
```

Aplikasi otomatis mendeteksi dua jenis input:
- **Foto kendaraan penuh** → mencari plat di dalam gambar
- **Plat ter-crop (close-up)** → langsung segmentasi karakter

---

## 📁 Struktur Project

```
license-plate-expiry/
├── app/
│   └── streamlit_app.py          # Aplikasi Streamlit (self-contained)
├── src/                          # Modul inti
│   ├── preprocessing.py          # Grayscale, blur, threshold
│   ├── plate_detection.py        # Lokalisasi plat
│   ├── char_segmentation.py      # Segmentasi karakter
│   ├── features.py               # Ekstraksi fitur HOG
│   ├── ocr.py                    # OCR classifier (SVM)
│   ├── expiry.py                 # Parsing tanggal MM.YY
│   ├── plate_classifier.py       # Plat vs non-plat
│   ├── evaluation.py             # Metrik evaluasi
│   ├── pipeline.py               # Pipeline end-to-end
│   └── synthetic.py              # Generator data sintetis
├── scripts/
│   ├── train_ocr_from_cropped_plates.py  # Training OCR (utama)
│   ├── train_ocr_improved.py             # Training dengan augmentasi
│   ├── evaluate_ocr.py                   # Evaluasi pada foto plat
│   ├── evaluate_dataset_char.py          # Evaluasi pada karakter
│   ├── run_inference.py                  # Inferensi 1 gambar
│   └── test_on_dataset.py                # Tes batch
├── models/
│   └── ocr_svm.joblib            # Model HOG+SVM terlatih (5.2 MB)
├── notebooks/
│   └── demo.ipynb                # Notebook demo
├── tests/
│   └── smoke_test.py             # Smoke test
├── config.py                     # Konfigurasi global
├── requirements.txt              # Dependencies Python
└── packages.txt                  # System packages (Streamlit Cloud)
```

> **Catatan:** Folder `data/` tidak disertakan (dataset publik, terlalu besar). Lihat bagian [Melatih Model](#-melatih-model).

---

## ⚙️ Instalasi

### Prasyarat
- Python 3.9 – 3.12
- pip

### Langkah
```bash
# Clone repo
git clone https://github.com/USERNAME/license-plate-expiry.git
cd license-plate-expiry

# (Opsional) buat virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Menjalankan Aplikasi

```bash
streamlit run app/streamlit_app.py
```

Buka browser ke `https://lisence-plate-detector-2z9sxpciywtobpapsjufwm.streamlit.app/`, lalu upload foto plat atau kendaraan.

---

## 🧠 Melatih Model

Model sudah disertakan (`models/ocr_svm.joblib`), jadi **tidak wajib** melatih ulang. Tapi jika ingin melatih dengan dataset sendiri:

### Dataset
Letakkan foto plat ter-crop dengan **nama file = nomor plat** (contoh: `H1234AB.jpg`):
```
data/PlateTrainingDataset/
├── H1234AB.jpg
├── AD5678CD.jpg
└── ...
```

Dan/atau dataset karakter individual:
```
data/raw/DatasetCharacter/
├── A/  ├── B/  ├── 0/  ├── 1/  ...
```

### Training
```bash
python scripts/train_ocr_from_cropped_plates.py \
    --plates-dir data/PlateTrainingDataset \
    --char-dir   data/raw/DatasetCharacter \
    --augment-factor 5
```

Model baru akan tersimpan di `models/ocr_svm.joblib`.

---

## 📊 Evaluasi

### Evaluasi pada foto plat (PlateTrainingDataset)
```bash
python scripts/evaluate_ocr.py \
    --plates-dir data/PlateTrainingDataset \
    --mode plates \
    --out-dir outputs
```

### Evaluasi pada karakter (DatasetCharacter)
```bash
python scripts/evaluate_dataset_char.py \
    --char-dir data/raw/DatasetCharacter \
    --out-dir  outputs/eval_dataset1 \
    --split-eval
```

### Output evaluasi
- `evaluation_report.txt` — accuracy, precision, recall, F1, confusion matrix (ASCII)
- `confusion_matrix.png` — heatmap confusion matrix
- `evaluation_metrics.json` — semua metrik dalam JSON

---

## 📈 Hasil

| Dataset | Karakter | Accuracy | Weighted F1 |
|---|---|---|---|
| **DatasetCharacter** (karakter bersih) | 10.562 | **97.57%** | 0.976 |
| **PlateTrainingDataset** (plat asli) | 422 | **89.49%** | 0.893 |
| **Archive Dataset** (COCO, plat jalanan) | 7.990 | **58.36%** | 0.576 |

**Temuan utama:** Model HOG+SVM sangat akurat pada karakter bersih (97.57%), namun akurasi menurun pada foto plat dunia nyata akibat noise (keausan, kemiringan, pencahayaan). Hal ini menunjukkan bahwa keterbatasan sistem ada pada tahap **segmentasi dan kualitas citra**, bukan pada kemampuan klasifikasi SVM.

---

## 🛠️ Teknologi

| Komponen | Library |
|---|---|
| Image processing | OpenCV (`opencv-python-headless`) |
| Feature extraction | scikit-image (HOG) |
| Classifier | scikit-learn (SVM) |
| Web app | Streamlit |
| Model serialization | joblib |

---

## 👥 Tim

| Nama | NIM |
|---|---|
| Limanto Winata | 2802512530 |
| Dhammika Kumara | 2802512865 |
| I Komang Satrya Artha Suryawan | 2802512133 |
| Nyoman Rajendra Gautama Ewari | 2802512096 |
| Kenneth Howen | 2802504346 |

---
