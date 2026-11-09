# Supermercado · Caixa Autônomo

Protótipo de ponto de venda que usa a câmera para reconhecer embalagens e montar o carrinho. A interface em português mostra o vídeo, o progresso da leitura, o seletor de câmera e o total em reais.

## Abrir no Windows

1. Baixe `CaixaVisaoComputacional-Windows-Python.zip` na seção **Releases** deste repositório.
2. Extraia o ZIP por inteiro para uma pasta local e abra `CaixaVisaoComputacional.exe`. Mantenha `_internal`, `modelo` e `ocr_keywords.json` junto do executável.
3. Selecione a câmera e apresente um produto de cada vez. Use **Remover** para excluir uma linha e **Nova compra** para ver o total da compra atual e esvaziar o carrinho. `Esc` fecha a janela.

O pacote inclui o ambiente do aplicativo. O computador precisa de Windows de 64 bits e acesso a uma câmera. Na execução por código-fonte, instale as dependências abaixo.

O ZIP Python de 3 de novembro está sem o arquivo de configuração do RapidOCR. A leitura de rótulos por OCR fica indisponível nesse pacote.

## Como a leitura funciona

- A captura e a inferência rodam em tarefas separadas. O modo **Vídeo fluido** exibe o quadro mais recente com a última detecção. Desmarque para exibir cada quadro junto de sua própria inferência.
- O YOLO26 analisa o quadro usando `Programa/modelo/best.onnx`. O aplicativo considera classes do catálogo em `Programa/config.py`. A pontuação soma até 40 pontos pela confiança do YOLO e até 5 pelo desempate pela posição no quadro. Quando o OCR está disponível e ativado, palavras do rótulo somam até 60 pontos, segundo `ocr_keywords.json`.
- A confiança mínima começa em 60% e pode ser ajustada com `−` e `+`. Uma classe precisa permanecer como candidata por 1,5 s. Depois de adicionar o produto, o caixa espera 2,5 s. A mesma classe só volta a ser elegível depois de sair da cena pelo intervalo configurado.
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

Para empacotar, execute `.\.venv\Scripts\python.exe -m PyInstaller CaixaVisaoComputacional.spec` a partir de `Programa`. Em `dist\CaixaVisaoComputacional`, coloque `ocr_keywords.json` e uma pasta `modelo` com `best.onnx`, `gesture_model.joblib` e `hand_landmarker.task`. Distribua a pasta inteira em um ZIP. Valide a inicialização, câmera, leitura e gesto no pacote antes de substituir uma release.

O [guia técnico Python](Programa/README.md) descreve a exportação ONNX e a ferramenta `train_gesture_model.py` para coletar exemplos e treinar a MLP. Consulte o [histórico de versões](CHANGELOG.md) para ver o conteúdo de cada release.
