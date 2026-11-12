# Supermercado · Caixa Autônomo

Protótipo de ponto de venda que usa a câmera para reconhecer embalagens e montar o carrinho. A interface em português mostra o vídeo, o progresso da leitura, o seletor de câmera e o total em reais.

## Abrir no Windows

1. Baixe `CaixaVisaoComputacional-Windows-Python.zip` na seção **Releases** deste repositório.
2. Extraia o ZIP por inteiro para uma pasta local e abra `CaixaVisaoComputacional.exe`. Mantenha `_internal`, `modelo` e `ocr_keywords.json` junto do executável.
3. Selecione a câmera e apresente um produto de cada vez. Use **Remover** para excluir uma linha e **Nova compra** para ver o total da compra atual e esvaziar o carrinho. `Esc` fecha a janela.

O pacote inclui o ambiente do aplicativo. O computador precisa de Windows de 64 bits e acesso a uma câmera. Na execução por código-fonte, instale as dependências abaixo.

O ZIP Python atual inclui os modelos de OCR. Na release de 3 de novembro, faltava um arquivo de configuração do RapidOCR.

## Delimitar a leitura e aguardar a retirada da mão

Os executáveis Python e Rust aceitam as duas opções, que podem ser combinadas no PowerShell.

```powershell
.\CaixaVisaoComputacional.exe --zone --hand-block
```

- `--zone` pausa o YOLO até você arrastar um retângulo sobre o vídeo. Ao soltar, o contorno some e o YOLO passa a analisar somente aquele recorte. Use **Redefinir área de detecção** para selecionar outra região. Ao trocar de câmera ou reiniciar, selecione a área novamente.
- `--hand-block` usa o MediaPipe no quadro inteiro da câmera. Enquanto houver uma mão detectada ou faltar uma leitura recente, o caixa pausa o lançamento, inclusive quando o OCR confirma o YOLO. Ao retirar a mão, a leitura do item recomeça. A opção funciona com o fechamento por gesto desativado.

## Classificador de produtos opcional

```powershell
.\CaixaVisaoComputacional.exe --classifier
.\CaixaVisaoComputacional.exe --classifier "C:\modelos\meu_modelo.h5"
```

Em ambos os executáveis, `--classifier` ativa o modelo Keras do projeto já convertido para ONNX. Ele classifica a caixa válida de maior confiança do YOLO. A classe de fundo `Nada`, nomes fora do catálogo e resultados abaixo do limiar de confiança não pontuam. A opção pode ser combinada com `--zone` e `--hand-block`.

Para abrir outro arquivo `.h5`, coloque um `labels.txt` ao lado dele, com um índice e um nome por linha, como `0 Miojo`. Use um modelo Keras completo com entrada RGB NHWC de dimensões fixas, normalização de pixels de −1 a 1 e saída de probabilidades. A conversão usa Python com `keras`, `h5py`, `torch` e `onnx`. Se necessário, defina `CLASSIFIER_PYTHON` com o caminho do Python que tem essas dependências. Para converter uma vez e usar o resultado diretamente, execute `python Programa/export_classifier.py modelo.h5 modelo.onnx` e aponte `--classifier` para o arquivo ONNX.

## Como a leitura funciona

- A captura e a inferência rodam em tarefas separadas. O modo **Vídeo fluido** exibe o quadro mais recente com a última detecção. Desmarque para exibir cada quadro junto de sua própria inferência.
- O YOLO26 e o classificador somam até 30 pontos cada pela confiança. O OCR soma até 35 pontos pela fração de palavras-chave únicas da classe reconhecidas no rótulo, com desconto por correspondência aproximada. O desempate pela posição vale até 5 pontos.
- A confiança mínima começa em 60% e pode ser ajustada com `−` e `+`. Quando o YOLO e uma leitura atual do OCR confirmam o mesmo produto, ele entra no carrinho assim que o OCR responde. Em caso de divergência, empate ou OCR indisponível, a classe precisa permanecer como candidata por 1,5 s. Depois de adicionar o produto, o caixa espera 2,5 s. A mesma classe só volta a ser elegível depois de sair da cena pelo intervalo configurado.
- Depois de lançar um item, a barra sobe até 100% em cerca de 120 ms e fica cheia por 0,4 s. Esse efeito visual não altera o tempo de confirmação nem o intervalo contra duplicidade.
- O gesto é opcional e começa desativado. O MediaPipe Hand Landmarker extrai 21 pontos em três dimensões de cada mão. O programa normaliza os pontos pelo pulso e pelo tamanho da palma, ordena esquerda e direita e envia 126 valores ao classificador MLP treinado. Com as duas mãos e confiança de pelo menos 80% por 1,5 s, o gesto aciona o mesmo fechamento de **Nova compra**.
- `--debug` abre o painel de pontuação e OCR. Preços e parâmetros ficam em `config.py`. Para sobrescrever parâmetros em uma instalação, copie `Programa/config_override.example.json` para `config_override.json` ao lado do executável e reinicie.

O catálogo, as classes do detector e as palavras de OCR devem permanecer alinhados ao atualizar os modelos. O arquivo `Programa/best_openvino_model/metadata.yaml` identifica o modelo OpenVINO de doces guardado no projeto, enquanto o caixa Python abre `modelo/best.onnx` por padrão.

## Desenvolver e empacotar a versão Python

```powershell
cd Programa
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt pyinstaller
.\.venv\Scripts\python.exe main.py
```

Para empacotar, execute `.\.venv\Scripts\python.exe -m PyInstaller CaixaVisaoComputacional.spec` a partir de `Programa`. Em `dist\CaixaVisaoComputacional`, coloque `ocr_keywords.json` e uma pasta `modelo` com `best.onnx`, `gesture_model.joblib`, `hand_landmarker.task`, `keras_model.onnx` e `labels.txt`. Distribua a pasta inteira em um ZIP. Valide a inicialização, câmera, leitura e gesto no pacote antes de substituir uma release.

O [guia técnico Python](Programa/README.md) descreve a exportação ONNX e a ferramenta `train_gesture_model.py` para coletar exemplos e treinar a MLP. Consulte o [histórico de versões](CHANGELOG.md) para ver o conteúdo de cada release.
