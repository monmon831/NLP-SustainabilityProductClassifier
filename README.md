# 🌱 Sustainability Product Review Classification API

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

API untuk mengklasifikasikan review produk e-commerce berdasarkan aspek keberlanjutan (sustainability) sesuai dengan **SDG 12: Responsible Consumption and Production**.

## 🎯 Tentang Projek

Projek ini mengembangkan sistem klasifikasi otomatis untuk menganalisis review produk dari perspektif keberlanjutan. Sistem ini membantu:

- **E-commerce platforms** untuk monitoring kualitas produk dan kemasan
- **Konsumen** untuk membuat keputusan pembelian yang lebih berkelanjutan
- **Produsen** untuk mendapat feedback tentang sustainability produk mereka
- **Peneliti** untuk analisis sentimen dan tren sustainability

Dataset yang digunakan: **PRDECT-ID Dataset** (Product Review Dataset Indonesia)

## 🏷️ Kategori Klasifikasi

Model mengklasifikasikan review ke dalam 4 kategori:

| Kategori | Deskripsi | Contoh Review |
|----------|-----------|---------------|
| **Kemasan Boros** | Review yang menyoroti penggunaan kemasan berlebihan | "Terlalu banyak plastik dan bubble wrap" |
| **Produk Tidak Tahan Lama** | Review tentang produk yang cepat rusak/tidak awet | "Baru 2 minggu sudah rusak, kualitas buruk" |
| **Produk Awet & Berkualitas** | Review positif tentang kualitas dan daya tahan | "Sudah 6 bulan masih bagus, worth it!" |
| **Netral** | Review yang tidak terkait aspek sustainability | "Pengiriman cepat, terima kasih" |

## ✨ Fitur

- ✅ **Single Prediction**: Klasifikasi satu review
- ✅ **Batch Prediction**: Klasifikasi hingga 100 review sekaligus
- ✅ **Confidence Score**: Skor kepercayaan untuk setiap prediksi
- ✅ **Probability Distribution**: Probabilitas untuk semua kategori
- ✅ **RESTful API**: Mudah diintegrasikan dengan aplikasi lain
- ✅ **CORS Enabled**: Support untuk frontend dari domain berbeda
- ✅ **Health Check**: Monitoring status API dan model
- ✅ **Error Handling**: Response error yang informatif

## 🛠️ Teknologi

- **Backend Framework**: Flask 2.3.3
- **Machine Learning**: scikit-learn 1.3.0
- **Text Processing**: NLTK, Regex
- **Model**: Multinomial Naive Bayes
- **Feature Extraction**: TF-IDF Vectorizer
- **Data Handling**: imbalanced-learn (SMOTE)

## 📦 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/username/text_classifier.git
cd text_classifier
```

### 2. Buat Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Pastikan Model Files Ada

Pastikan file-file berikut ada di root directory:
- `sustainability_model_model.pkl`
- `sustainability_model_vectorizer.pkl`
- `sustainability_model_preprocessing.pkl`

## 🚀 Cara Menggunakan

### Menjalankan Server
```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

### Testing dengan cURL

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Produk bagus dan tahan lama, recommended!"}'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      "Kemasan terlalu banyak plastik",
      "Barang cepat rusak",
      "Pengiriman cepat"
    ]
  }'
```

### Testing dengan Python
```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={'review': 'Produk sangat awet dan berkualitas'}
)
print(response.json())

# Batch prediction
response = requests.post(
    'http://localhost:5000/predict/batch',
    json={
        'reviews': [
            'Kemasan boros banget',
            'Produk tidak tahan lama',
            'Pengiriman cepat'
        ]
    }
)
print(response.json())
```

## 🔌 API Endpoints

### 1. Home Endpoint
```
GET /
```
Cek status API

**Response:**
```json
{
  "message": "Sustainability Classification API is running!",
  "status": "success",
  "timestamp": "2024-01-15T10:30:00",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

### 2. Health Check
```
GET /health
```
Monitor kesehatan API

### 3. Single Prediction
```
POST /predict
Content-Type: application/json

{
  "review": "Review text here"
}
```

**Response:**
```json
{
  "prediction": "produk_awet_berkualitas",
  "prediction_label": "Produk Awet & Berkualitas",
  "confidence": 0.8547,
  "all_probabilities": {
    "kemasan_boros": 0.0234,
    "produk_tidak_tahan_lama": 0.0512,
    "produk_awet_berkualitas": 0.8547,
    "netral": 0.0707
  },
  "processed_text": "produk sangat awet berkualitas",
  "original_text": "Produk sangat awet dan berkualitas",
  "text_length": 34,
  "status": "success",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 4. Batch Prediction
```
POST /predict/batch
Content-Type: application/json

{
  "reviews": ["Review 1", "Review 2", "Review 3"]
}
```

**Limitations:**
- Maximum 100 reviews per request
- Maximum 5000 characters per review

### 5. Model Information
```
GET /model/info
```
Informasi tentang model yang digunakan

## 📊 Model Performance

Berdasarkan evaluasi dengan test set:

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 88% |
| **Macro F1-Score** | 0.88 |
| **Weighted F1-Score** | 0.88 |

**Per-Class Performance:**

| Kategori | Precision | Recall | F1-Score | Accuracy |
|----------|-----------|--------|----------|----------|
| Kemasan Boros | 0.88 | 0.93 | 0.90 | 92.72% |
| Netral | 0.87 | 0.79 | 0.82 | 78.54% |
| Produk Awet & Berkualitas | 0.86 | 0.89 | 0.87 | 88.58% |
| Produk Tidak Tahan Lama | 0.90 | 0.95 | 0.92 | 94.69% |

## 📁 Struktur Projek
```
text_classifier/
│
├── app.py                              # Flask API application
├── product_review_sentiment_nlp.ipynb  # Notebook training & eksperimen
├── model_verification.py               # Script verifikasi model
├── index.html                          # Frontend demo (optional)
├── requirements.txt                    # Python dependencies
│
├── sustainability_model_model.pkl           # Trained model
├── sustainability_model_vectorizer.pkl      # TF-IDF vectorizer
├── sustainability_model_preprocessing.pkl   # Preprocessing config
│
└── __pycache__/                        # Python cache
    └── __init__.py
```

## 🌐 Deployment

### Option 1: Local Development
```bash
python app.py
```

### Option 2: Production dengan Gunicorn (Linux/Mac)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 3: Production dengan Waitress (Windows)
```bash
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Option 4: Docker (Coming Soon)
```dockerfile
# Dockerfile example
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Environment Variables (Recommended)
```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
export MODEL_PATH=/path/to/model
```

## 🧪 Testing

### Run Model Verification
```bash
python model_verification.py
```

### Manual Testing Checklist
- [ ] `/` endpoint returns success
- [ ] `/health` shows model loaded
- [ ] `/predict` works with valid input
- [ ] `/predict` handles empty input correctly
- [ ] `/predict/batch` processes multiple reviews
- [ ] `/model/info` returns model details

## 📈 Future Improvements

- [ ] Add more sustainability categories
- [ ] Implement aspect-based sentiment analysis
- [ ] Add multilingual support
- [ ] Create web interface
- [ ] Add database for logging predictions
- [ ] Implement caching for common queries
- [ ] Add rate limiting
- [ ] Create Docker container
- [ ] Add unit tests and CI/CD

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. Fork repository ini
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 Lisensi

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Author

**Alisha Monifa**
- GitHub: [@author](https://github.com/monmon831)
- Email: alishamonifa3@gmail.com

## 🙏 Acknowledgments

- Dataset: PRDECT-ID (Product Review Dataset Indonesia)
- SDG 12: Responsible Consumption and Production
- scikit-learn community
- Flask framework


---

⭐ **Jangan lupa beri star jika projek ini membantu!** ⭐
