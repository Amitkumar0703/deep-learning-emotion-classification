
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from fastapi.staticfiles import StaticFiles
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# from fastapi.responses import FileResponse
# from pydantic import BaseModel, Field
# from keras.models import load_model
# import numpy as np
# import pickle
# import re




# """
# 1. We are going to make some constants like:
# A. Model Path (BiGRU)
# B. Tokenizer Path
# C. Max Sequence Length
# D. Emotion Labels
# E. Emotion emojis
# """
# #A. Model Path (BiGRU)
# model_path = "Artifacts/BiGRU_Model.keras"

# #B. Tokenizer Path
# tokenizer_path = "Artifacts/tokenizer.pkl"

# #C. Max Sequence Length
# max_sequence_length = 50

# #D. Emotion Labels
# emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# #E. Emotion emojis
# EMOTION_EMOJIS = {
#     "sadness": "😢",
#     "joy": "😄",
#     "love": "❤️",
#     "anger": "😠",
#     "fear": "😨",
#     "surprise": "😲",
# }



# """
# 2. Preprocess the upcoming text
# Cleans raw text so it matches the format used while training.
# A. Convert the text to lowercase. -done
# B. Remove apostrophes (e.g can't -> cant). -done
# C. Remove Special Characters and Punctuation. -done
# D. Remove extra spaces -done
# """

# def preprocess_text(text: str)->str:
#     text = text.lower()
#     text = re.sub(r"'","",text)
#     text = re.sub(r"[^a-z0-9\s]"," ", text)
#     text = re.sub(r"\s+", " ",text).strip()
#     return text


# """
# 3. Request and Response Schemas
# A. Text Input -> Input schema the text sent by user. -done
# B. Prediciton Response -> Output schema the emotion to predict. -done
# C. Health Response (Server health check)
# """

# class TextInput(BaseModel):
#     text : str = Field(
#         ...,
#         min_length=1,
#         max_length=2000,
#         description="The sentence to analyze",
#         json_schema_extra={"example": "I feel so happy and excited"}
#         )

# class PredictionResponse(BaseModel):
#     text: str
#     predicted_emotion: str
#     confidence : float
#     all_probabilites: dict[str, float]

# class HealthResponse(BaseModel):
#     status: str
#     model_loaded: bool

# """
# 4. Model Loading and LifeSpan Management
# Load the model and toknizer once the server starts up.
# """
# dl_model = {} #{1. BiGRU, 2. Tokenizer}-> True , {} -> False

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print('Loading the model and tokenizer...')
#     dl_model["BiGRU"] = load_model(model_path)                      #BiGRU Model
#     with open(tokenizer_path, 'rb') as file:
#         dl_model["Tokenizer"] = pickle.load(file)
#     print('Model are loaded successfully...')   

#     yield #Pause, model is laoded and server is running and at this point model wait karega for request

#     dl_model.clear() #Ek baar server band ho gaya uske baad model ko memory se hata do.
               

# """
# 5. Mount the static files to the FastAPI app
# A. Enable CORS (Cross-Origin Resource Sharing) to allow requests from different origins.
# """
# app = FastAPI(
#     lifespan=lifespan
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.mount('/static', StaticFiles(directory="static"), name="static")



# """
# 6. API Endpoints.
# A. Server UI at homepage ('/')
# B. Health Check Endpoint ('/health')
# C. Predict Emotion Endpoint ('/predict')
# """

# #A. Server UI at homepage ('/')
# @app.get('/', include_in_schema=False)
# def server_ui():
#     return FileResponse('static/index.html')

# #B. Health Check Endpoint ('/health')
# @app.get('/health', response_model=HealthResponse)
# def health_check():
#     return HealthResponse(status="Server is running", model_loaded=bool(dl_model))

# #C. Predict Emotion Endpoint ('/predict')
# @app.post('/predict', response_model=PredictionResponse)
# def predict_emotion(text_input: TextInput):
#     """
#     1. Cleans the input sentences.      
#     2. Convert the words into numeric using tokenizer.
#     3. Pad the sequences to ensure uniform length.
#     4. Run prediction using the BiGRU model.
#     5. Return the top emotion and full probability breakdown.
#     """

#     BiGRU_model     = dl_model.get("BiGRU")
#     tokenizer_model = dl_model.get("Tokenizer")

#     if BiGRU_model is None or tokenizer_model is None:
#         raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

#     #1. 
#     cleaned_text = preprocess_text(text_input.text)

#     #2. and 3. 
#     tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
#     padded_sequence = pad_sequences(
#         tokenized_text,
#         maxlen=max_sequence_length,
#         padding="post",
#         truncating="post"
#     )

#     probabilites     = BiGRU_model.predict(padded_sequence)[0]

#     top_emotion_index = int(np.argmax(probabilites)) # 4
#     all_probabilites =  {
#         label: float(prob) for prob, label in zip(probabilites, emotion_labels)
          
#     }

#     return PredictionResponse(
#         text = text_input.text,
#         predicted_emotion = emotion_labels[top_emotion_index],
#         confidence = float(probabilites[top_emotion_index]), 
#         all_probabilites = all_probabilites
#     )






from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import torch.nn.functional as F


# ============================================================
# 1. CONSTANTS
# ============================================================

"""
A. Model Path (Fine-tuned RoBERTa)
B. Maximum Token Length
C. Emotion Labels
D. Emotion Emojis
"""

# A. RoBERTa model + tokenizer folder
MODEL_PATH = "best_emotion_model"

# B. Maximum token length
MAX_SEQUENCE_LENGTH = 64

# C. Emotion labels
EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

# Mapping used during model training
ID_TO_LABEL = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

LABEL_TO_ID = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5
}

# D. Emojis
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# ============================================================
# 2. REQUEST AND RESPONSE SCHEMAS
# ============================================================

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "I feel so happy and excited today!"
        }
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ============================================================
# 3. MODEL STORAGE
# ============================================================

"""
The model and tokenizer are loaded only once when FastAPI starts.

Old:
    BiGRU_Model.keras
    tokenizer.pkl

New:
    Hugging Face RoBERTa model
    Hugging Face RoBERTa tokenizer
"""

dl_model = {}


# ============================================================
# 4. APPLICATION LIFESPAN
# ============================================================
import os

print("=" * 60)
print("Current Directory:", os.getcwd())
print("Model Path:", MODEL_PATH)
print("Path Exists:", os.path.exists(MODEL_PATH))

if os.path.exists(MODEL_PATH):
    print("Files inside model folder:")
    print(os.listdir(MODEL_PATH))
print("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Loading RoBERTa emotion model and tokenizer...")

    # Automatically use GPU if available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    # Load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    # Ensure correct labels are attached to the model
    model.config.id2label = ID_TO_LABEL
    model.config.label2id = LABEL_TO_ID

    # Move model to GPU/CPU
    model.to(device)

    # Important for inference
    model.eval()

    # Store everything
    dl_model["model"] = model
    dl_model["tokenizer"] = tokenizer
    dl_model["device"] = device

    print("RoBERTa model loaded successfully.")
    print(f"Running on: {device}")
    print("=" * 60)

    yield

    # Remove from memory when server stops
    dl_model.clear()

    # Optional GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Model unloaded successfully.")


# ============================================================
# 5. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Emotion Classification API",
    description=(
        "Emotion classification using a fine-tuned "
        "RoBERTa transformer model."
    ),
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# 6. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    # Development:
    allow_origins=["*"],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 7. STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# 8. HOME PAGE
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def server_ui():

    return FileResponse(
        "static/index.html"
    )


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    model_loaded = (
        "model" in dl_model and
        "tokenizer" in dl_model
    )

    device = str(
        dl_model.get(
            "device",
            "not-loaded"
        )
    )

    return HealthResponse(
        status="Server is running",
        model_loaded=model_loaded,
        device=device
    )


# ============================================================
# 10. PREDICT EMOTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_emotion(
    text_input: TextInput
):

    """
    Prediction Pipeline:

    1. Receive raw text.
    2. Tokenize text using RoBERTa tokenizer.
    3. Convert tokens to PyTorch tensors.
    4. Run inference through fine-tuned RoBERTa.
    5. Convert logits to probabilities using Softmax.
    6. Select emotion with highest probability.
    7. Return predicted emotion, confidence and all scores.

    IMPORTANT:
    We DO NOT manually lowercase/remove punctuation because
    RoBERTa tokenizer handles the raw text itself.
    """

    model = dl_model.get("model")
    tokenizer = dl_model.get("tokenizer")
    device = dl_model.get("device")

    # Make sure model is loaded
    if (
        model is None or
        tokenizer is None or
        device is None
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded yet. "
                "Please try again later."
            )
        )

    # Remove only unnecessary leading/trailing spaces
    text = text_input.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    try:

        # ----------------------------------------------------
        # 1. TOKENIZATION
        # ----------------------------------------------------

        inputs = tokenizer(
            text,

            # Return PyTorch tensors
            return_tensors="pt",

            # Truncate sentences longer than max length
            truncation=True,

            max_length=MAX_SEQUENCE_LENGTH,

            # Padding is useful if batching later
            padding=True
        )

        # Move tensors to same device as model
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # 2. MODEL INFERENCE
        # ----------------------------------------------------

        # Disables gradient calculations
        # -> faster inference
        # -> less memory usage
        with torch.inference_mode():

            outputs = model(**inputs)

            logits = outputs.logits


        # ----------------------------------------------------
        # 3. LOGITS -> PROBABILITIES
        # ----------------------------------------------------

        probabilities = F.softmax(
            logits,
            dim=-1
        )[0]


        # ----------------------------------------------------
        # 4. GET TOP EMOTION
        # ----------------------------------------------------

        top_emotion_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        predicted_emotion = ID_TO_LABEL[
            top_emotion_index
        ]

        confidence = float(
            probabilities[
                top_emotion_index
            ].item()
        )


        # ----------------------------------------------------
        # 5. ALL EMOTION PROBABILITIES
        # ----------------------------------------------------

        all_probabilities = {
            ID_TO_LABEL[index]:
                float(probability.item())

            for index, probability
            in enumerate(probabilities)
        }


        # ----------------------------------------------------
        # 6. RESPONSE
        # ----------------------------------------------------

        return PredictionResponse(
            text=text_input.text,

            predicted_emotion=predicted_emotion,

            emoji=EMOTION_EMOJIS[
                predicted_emotion
            ],

            confidence=confidence,

            all_probabilities=all_probabilities
        )


    except Exception as error:

        print(
            f"Prediction Error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to predict emotion."
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)