from huggingface_hub import login
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from translate import Translator
import torch
from gtts import gTTS
import os
from dotenv import load_dotenv

load_dotenv()


# Token do Hugging Face
token = os.getenv("HUGGING_FACE_TOKEN")
login(token=token)

# URL da imagem
url = "https://cdn.pixabay.com/photo/2023/08/05/08/15/ship-8170663_1280.jpg"

# Carrega a imagem da URL
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Carrega o processador e o modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Preprocessa a imagem
inputs = processor(image, return_tensors="pt")

# Gera a descrição em inglês
with torch.no_grad():
    out = model.generate(**inputs)

description_en = processor.decode(out[0], skip_special_tokens=True)

# Traduzindo para o português
translator = Translator(to_lang="pt")
description_pt = translator.translate(description_en)
print("Descrição em Português:", description_pt)

# Converte a descrição para áudio
tts = gTTS(text=description_pt, lang='pt')
tts.save("description_pt.mp3")

