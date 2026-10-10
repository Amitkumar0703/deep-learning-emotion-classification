# 🚀 Deep Learning Emotion Classification System

A production-ready Deep Learning Emotion Classification System built using RoBERTa, FastAPI, and a modern frontend interface.

The system predicts human emotions from text input and classifies them into six categories:

- 😊 Joy
- 😢 Sadness
- ❤️ Love
- 😡 Anger
- 😨 Fear
- 😲 Surprise

---

# 🌐 Live Demo

### Web Application
https://your-render-url.onrender.com

### API Documentation
https://your-render-url.onrender.com/docs

---

# 📸 Application Screenshots

## Main Interface

![UI](screenshots/joy_prediction.jpg)

---

## Joy Prediction

![Joy](screenshots/joy_prediction.jpg)

---

## Sadness Prediction

![Sadness](screenshots/sadness_prediction.jpg)

---

## Love Prediction

![Love](screenshots/love_prediction.jpg)

---

## Fear Prediction

![Fear](screenshots/fear_prediction.jpg)

---

## Surprise Prediction

![Surprise](screenshots/surprise_prediction.jpg)

---

# 🏗 System Architecture

## Overall Architecture

![Architecture](docs/System%20architeture.jpg)

---

## Model Workflow

![Workflow](docs/Model%20Workflow.drawio.png)

---

## Deployment Architecture

![Deployment](docs/Deployment%20Architecture.drawio.png)

---

# 📂 Project Structure

```bash
DEEP_LEARNING_EMOTION_CLASSIFICATION_PROJECT
│
├── best_emotion_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
│
├── docs/
│   ├── Deployment Architecture.drawio.png
│   ├── Model Workflow.drawio.png
│   └── System_architecture.jpg
│
├── notebooks/
│
├── screenshots/
│   ├── api_docs.jpg
│   ├── api_response.jpg
│   ├── joy_prediction.jpg
│   ├── sadness_prediction.jpg
│   ├── fear_prediction.jpg
│   ├── love_prediction.jpg
│   ├── surprise_prediction.jpg
│   └── predict_docs.jpg
│
├── static/
│   └── index.html
│
├── main.py
├── requirements.txt
├── runtime.txt
├── render.yaml
├── MODEL_CARD.md
├── LICENSE
└── README.md
```

---

# 🧠 Model Information

| Property | Value |
|-----------|---------|
| Model | RoBERTa |
| Framework | Transformers |
| Backend | FastAPI |
| Frontend | HTML/CSS/JavaScript |
| Deployment | Render |
| Language | Python |
| Task | Emotion Classification |

---

# 🎯 Supported Emotions

| Emotion | Description |
|----------|-------------|
| Joy | Happiness, excitement |
| Sadness | Grief, disappointment |
| Love | Affection, care |
| Anger | Frustration, rage |
| Fear | Anxiety, worry |
| Surprise | Shock, amazement |

---

# ⚡ API Endpoint

## Predict Emotion

```http
POST /predict
```

Request

```json
{
    "text": "I am very happy today"
}
```

Response

```json
{
    "predicted_emotion": "joy",
    "confidence": 0.98,
    "all_probabilities": {
        "joy": 0.98,
        "sadness": 0.01,
        "anger": 0.00,
        "fear": 0.00,
        "love": 0.01,
        "surprise": 0.00
    }
}
```

---

# 📖 API Documentation

FastAPI automatically generates Swagger documentation.

### Swagger UI

![Swagger](screenshots/api_docs.jpg)


# 📖 Model Response

FastAPI model response

### Response UI

![Swagger](screenshots/api_response.jpg)



Access:

```bash
/docs
```

---

# 🚀 Local Installation

Clone repository

```bash
git clone https://github.com/your-username/deep-learning-emotion-classification.git
```

Move into project

```bash
cd deep-learning-emotion-classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
uvicorn main:app --reload
```

Open

```bash
http://127.0.0.1:8000
```

---

# ☁️ Deployment

This project is deployed using Render.

Build Command

```bash
pip install -r requirements.txt
```

Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

# 📊 Future Improvements

- Multi-language emotion detection
- Emotion trends dashboard
- User history tracking
- Explainable AI (XAI)
- Real-time analytics
- Speech Emotion Recognition

---

# 👨‍💻 Author

Amit Kumar

B.Sc Computer Science & Data Analytics

AI / ML Engineer Aspirant

LinkedIn: Add Your LinkedIn Link

GitHub: Add Your GitHub Link

---

# ⭐ If you found this project useful

Please give this repository a star.