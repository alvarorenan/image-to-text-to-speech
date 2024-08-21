import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
from gtts import gTTS
import torch
from io import BytesIO
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Token do Hugging Face
token = os.getenv("HUGGING_FACE_TOKEN")

def login_hugging_face(token):
    try:
        login(token=token)
        print("Login realizado com sucesso no Hugging Face!")
    except Exception as e:
        print(f"Erro ao realizar login no Hugging Face: {e}")
        return False
    return True

def load_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except requests.RequestException as e:
        print(f"Erro ao carregar a imagem: {e}")
        return None

def process_blip(image):
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    inputs_blip = blip_processor(image, return_tensors="pt")
    
    with torch.no_grad():
        out_blip = blip_model.generate(**inputs_blip)
    description_en = blip_processor.decode(out_blip[0], skip_special_tokens=True)
    return description_en

def translate_text(text):
    try:
        translator = GoogleTranslator(source='en', target='pt')
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        print(f"Erro ao traduzir o texto: {e}")
        return None

def convert_to_audio(text, filename):
    try:
        tts = gTTS(text=text, lang='pt')
        tts.save(filename)
    except Exception as e:
        print(f"Erro ao salvar o áudio: {e}")

def main():
    if not login_hugging_face(token):
        return
    
    url = "https://farm6.staticflickr.com/5519/9382494910_b34268b6e4_z.jpg"
    
    image = load_image(url)
    if image is None:
        return
    
    description_en = process_blip(image)
    description_pt = translate_text(description_en)
    if description_pt is None:
        return
    print("Descrição em Português:", description_pt)
    
    audio_file = "description_pt.mp3"
    convert_to_audio(description_pt, audio_file)

if __name__ == "__main__":
    main()