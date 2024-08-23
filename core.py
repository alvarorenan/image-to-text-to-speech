import os
# Importa o módulo 'os' para acessar variáveis de ambiente e interagir com o sistema operacional, como leitura de arquivos .env.

import requests
# Importa a biblioteca 'requests', que é usada para fazer requisições HTTP, como baixar imagens da web.

from PIL import Image
# Importa a classe 'Image' do módulo 'PIL' (Python Imaging Library), que permite abrir, manipular e salvar diferentes formatos de imagem.

from transformers import BlipProcessor, BlipForConditionalGeneration
# Importa o 'BlipProcessor' e o 'BlipForConditionalGeneration' da biblioteca 'transformers'.
# 'BlipProcessor' é responsável por processar a imagem para ser usada pelo modelo.
# 'BlipForConditionalGeneration' é o modelo que gera descrições textuais baseadas em imagens.

from deep_translator import GoogleTranslator
# Importa 'GoogleTranslator' da biblioteca 'deep_translator', que é usada para traduzir texto de um idioma para outro, nesse caso, do inglês para o português.

from gtts import gTTS
# Importa 'gTTS' (Google Text-to-Speech), que converte texto em fala e permite salvar o áudio gerado em um arquivo.

import torch
# Importa a biblioteca 'torch', utilizada principalmente para operações de tensor e computação com GPU, comumente usada em machine learning e deep learning.

from io import BytesIO
# Importa 'BytesIO' da biblioteca 'io', que é uma classe usada para manipular dados binários em memória como se fossem um arquivo, útil para abrir imagens diretamente de bytes recebidos em uma requisição.

from huggingface_hub import login
# Importa a função 'login' da biblioteca 'huggingface_hub', que permite fazer login na plataforma Hugging Face para acessar modelos e outros recursos.

from dotenv import load_dotenv
# Importa a função 'load_dotenv' do módulo 'dotenv', que carrega variáveis de ambiente de um arquivo .env para o ambiente de execução do Python.


# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Token do Hugging Face é carregado das variáveis de ambiente
token = os.getenv("HUGGING_FACE_TOKEN")

def login_hugging_face(token):
    """
    Realiza o login no Hugging Face usando o token fornecido.
    Se o login for bem-sucedido, imprime uma mensagem de sucesso.
    Caso contrário, imprime a mensagem de erro e retorna False.
    """
    try:
        login(token=token)
        print("Login realizado com sucesso no Hugging Face!")
    except Exception as e:
        print(f"Erro ao realizar login no Hugging Face: {e}")
        return False
    return True

def load_image(url):
    """
    Faz o download de uma imagem da URL fornecida e a carrega como um objeto PIL Image.
    Em caso de erro na requisição HTTP, imprime a mensagem de erro e retorna None.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
        image = Image.open(BytesIO(response.content))  # Abre a imagem a partir dos bytes recebidos
        return image
    except requests.RequestException as e:
        print(f"Erro ao carregar a imagem: {e}")
        return None

def process_blip(image):
    """
    Usa o modelo BLIP para gerar uma descrição textual da imagem.
    Carrega o processador e o modelo BLIP, processa a imagem e gera uma descrição em inglês.
    """
    # Carrega o processador e o modelo BLIP pré-treinados
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    # Processa a imagem para gerar tensores de entrada para o modelo
    inputs_blip = blip_processor(image, return_tensors="pt")
    
    # Gera a descrição da imagem usando parâmetros de configuração
    out_blip = blip_model.generate(
        **inputs_blip,
        max_length=100,  # Ajuste o comprimento máximo conforme necessário
        min_length=20,   # Ajuste o comprimento mínimo conforme necessário
        num_beams=5,    # Use beam search para melhorar a qualidade
        length_penalty=1 # Penalize comprimentos mais curtos para gerar textos mais longos
    )
    
    # Decodifica a descrição gerada e remove tokens especiais
    description_en = blip_processor.decode(out_blip[0], skip_special_tokens=True)
    return description_en

def translate_text(text):
    """
    Traduz o texto fornecido do inglês para o português usando GoogleTranslator.
    Em caso de erro, imprime a mensagem de erro e retorna None.
    """
    try:
        translator = GoogleTranslator(source='en', target='pt')  # Define os idiomas de origem e destino
        translated_text = translator.translate(text)  # Realiza a tradução
        return translated_text
    except Exception as e:
        print(f"Erro ao traduzir o texto: {e}")
        return None

def convert_to_audio(text, filename):
    """
    Converte o texto fornecido em áudio e salva como um arquivo MP3 usando gTTS.
    Em caso de erro, imprime a mensagem de erro.
    """
    try:
        tts = gTTS(text=text, lang='pt')  # Converte o texto para fala em português
        tts.save(filename)  # Salva o áudio em um arquivo com o nome especificado
    except Exception as e:
        print(f"Erro ao salvar o áudio: {e}")

def main():
    """
    Função principal que orquestra o fluxo do programa.
    Realiza login no Hugging Face, baixa uma imagem, gera uma descrição,
    traduz a descrição para o português e converte a descrição em áudio.
    """
    # Realiza login no Hugging Face; se falhar, interrompe a execução
    if not login_hugging_face(token):
        return
    
    # URL da imagem a ser processada
    url = "https://cdn.pixabay.com/photo/2024/08/01/08/17/dahlia-8936439_1280.jpg"
    
    # Carrega a imagem a partir da URL; se falhar, interrompe a execução
    image = load_image(url)
    if image is None:
        return
    
    # Gera a descrição da imagem em inglês usando o modelo BLIP
    description_en = process_blip(image)
    
    # Traduz a descrição para o português
    description_pt = translate_text(description_en)
    if description_pt is None:
        return
    print("Descrição em Português:", description_pt)  # Imprime a descrição traduzida
    
    # Converte a descrição traduzida em áudio e salva como arquivo MP3
    audio_file = "description_pt.mp3"
    convert_to_audio(description_pt, audio_file)

# Executa a função principal ao rodar o script
if __name__ == "__main__":
    main()
