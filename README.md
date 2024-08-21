# Projeto da matéria IMD0265 - TÓPICOS ESPECIAIS EM BIOINFORMÁTICA

## Descrição

Esse script utiliza o modelo `BLIP` para gerar uma descrição de uma imagem. A descrição é gerada em inglês e traduzida para o português. Por fim, a descrição é convertida para áudio. O script utiliza a biblioteca Hugging Face Transformers para carregar o modelo `BLIP`, a biblioteca `PIL` para carregar a imagem, a biblioteca requests para fazer a requisição da imagem, a biblioteca `translate` para traduzir a descrição e a biblioteca `gTTS` para converter a descrição para áudio. O script também utiliza a biblioteca `python-dotenv` para carregar o token do Hugging Face a partir de um arquivo `.env`.

Você pode rodar o script com o comando:

```bash
python core.py
```

Lembrando que é necessário colocar a url da imagem que você deseja descrever na variável `url` no arquivo `core.py`. Também é necessário colocar o token do Hugging Face em um arquivo `.env` na raiz do projeto. O arquivo `example.env` mostra um exemplo de como o arquivo `.env` deve ser.