# import os
# import pickle
# import requests

# MODEL_PATH = "model/prebuild_model.pkl"
# MODEL_URL = "https://news-sentiment-290586778476.s3.amazonaws.com/prebuild_model.pkl"

# os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# if not os.path.exists(MODEL_PATH):
#     print("🔽 Downloading model from S3...")
#     r = requests.get(MODEL_URL)
#     with open(MODEL_PATH, "wb") as f:
#         f.write(r.content)

# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

# def predict_tone(text):
#     if not isinstance(text, str):
#         return "neutral"
#     return model.predict([text])[0]["label"]

import requests
import pickle
from io import BytesIO

MODEL_URL = "https://news-sentiment-290586778476.s3.ap-southeast-1.amazonaws.com/prebuild_model.pkl"

print("🔽 Downloading model from S3...")
response = requests.get(MODEL_URL)

if response.status_code != 200:
    print("❌ Response status:", response.status_code)
    print("❌ First 200 bytes:", response.text[:200])
    raise RuntimeError("❌ Failed to download model from S3")

# Only now do we try to unpickle
model = pickle.load(BytesIO(response.content))
