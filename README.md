# Detecção de placa

## Sobre

A aplicação detecção de placa é uma versão simplificada da aplicação [Cérbero](https://github.com/RToramaru/cerbero-mercosul), essa versão tem como propósito demosntrar como o Cérbero detecta a placa veícular.

Nessa versão simplificada, a detecção é realizada unicamente pela imagem, e após detectada a placa, a mesma é exibida na tela juntamente com a imagem do veículo, mostrando a região a qual foi retirada, podendo ser aplicado apenas em uma placa por vez.

## O projeto

O projeto foi desenvolvido utilizando de 4 etapas. A seguir está as etapas e suas caracterísitcas.

1. Detecção da região da placa.
    * Para a detecção é utlizado um modelo ONNX obtido pelo treiamento da rede convolucional [YOLOv7](https://github.com/WongKinYiu/yolov7) com os dados de veículos com placas do Mercosul de [Silvano, et al. 2020](https://data.mendeley.com/datasets/nx9xbs4rgx). Ao concluir a etapa de extração da placa, essa região é isolada e passada para as etapas seguintes.
2. Obter caracteres das placas.
    * Nessa etapa é aplicada um limiar na imagem para obter apenas os contornos na imagem, esses contornos são filtrados e agrupados somente os caractreres em uma nova imagem.
3. Reconhecimento dos caracteres.
    * A imagem dos caracteres é processada pelo algoritmo [EasyOCR
](https://github.com/JaidedAI/EasyOCR), que retorna uma string com o texto na imagem.
4. Interface.
    * Para interface da aplicação foi utilizado o micro-framework [Flask](https://flask.palletsprojects.com/en/2.2.x/).

## Clone do projeto

**Importante**
Para utilizar o repositório é necessário ter:
* **Python 3** : utilizado o Python 3.10.8

    * Download [Versão 3.10.8 64bits](https://www.python.org/ftp/python/3.10.8/python-3.10.8-amd64.exe)
    
    * Download [última versão](https://www.python.org/downloads/)

```
git clone https://github.com/RToramaru/detectcao-placa.git

pip install -r requirements.txt

```

## Executando

Para executar o projeto execute o comando:

```
python app.py
```
e acesse o endereço ``localhost:5000/``



## Demostração

https://user-images.githubusercontent.com/42619833/200982914-967d67b8-bc04-48cd-820a-5ecf52243e03.mp4



``@author Rafael Almeida``
