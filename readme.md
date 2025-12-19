# 🖼️ Image Similarity  
> *“Find how close two images really are — in pixels and perception.”*  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Framework-lightgrey?style=flat-square&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat-square&logo=opencv)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)
![Build Passing](https://img.shields.io/badge/Build-Passing-success?style=flat-square)

---

## 🌟 Overview
**Image Similarity** is an intelligent web-based application that compares images and quantifies their visual similarity.  
Built with **Flask + Python + OpenCV + deep feature extraction**, it allows users to upload one or multiple images, compute similarity scores, and visualize results in an interactive interface.

---

## 🧠 Tech Stack
| Layer | Technology |
|:------|:------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Python (Flask Framework) |
| **Image Processing** | OpenCV, NumPy, Scikit-Image |
| **Feature Extraction** | Pretrained CNN models (ResNet / MobileNet etc.) |
| **Database (optional)** | SQLite or file storage |
| **Deployment** | Flask server / Docker container |

---

## ⚙️ Installation & Setup
```bash
git clone https://github.com/sushantgarde/image_similarity.git
cd image_similarity
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open browser → [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧭 Folder Structure
```
image_similarity/
├── data/
│    ├── embeddings
│    ├── images
│    ├── model
├── src/
│   ├── feature_extractor.py
│   ├── similarity_search.py
│   ├── utils.py
│   └── preprocessing.py
├── static/
│    ├── css
│    │   ├── style.css
│    ├── js
│    │   ├── script.js
├── templates/
│   ├── index.html
│   ├── error.html
│   ├── result.html
├── config.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 🧪 Usage
- Upload images → Compare → View similarity %
- Batch compare folder of images
- Visual report with thumbnails & scores

---

## 🧮 How It Works
1. **Preprocessing** → Resize & normalize images  
2. **Feature Extraction** → CNN embeddings  
3. **Similarity Computation** → Cosine/Euclidean metrics  
4. **Result Visualization** → Web UI display

---

## 🧱 Docker Deployment
```bash
docker build -t image_similarity .
docker run -p 5000:5000 image_similarity
```

---

## 🚀 Future Enhancements
✅ Cluster similar images (K-Means / DBSCAN)  
✅ Integrate FAISS for fast search  
✅ REST API support  
✅ GPU acceleration  

---

## 👥 Contributors

| Contributors |
|---------------|
| **sushantgarde** |
| **gauravkale-8011** |
| **Vedant2004X** |

> *Want to join? Open a PR or connect on GitHub!*

---

## 📜 License
MIT License

---

## 👤 Author
**Sushant Dattatray Garde**  
B.Tech | Developer | Innovator  
🔗 [GitHub](https://github.com/sushantgarde)

---

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=footer&text=Made%20with%20💙%20by%20Sushant%20Garde&fontSize=18)
