# Supermercado · Caixa Autônomo

Protótipo de ponto de venda que usa a câmera para reconhecer embalagens e montar o carrinho. A interface em português mostra o vídeo, o progresso da leitura, o seletor de câmera e o total em reais.

## Versão Rust para Windows x64

Esta branch contém o caixa em Rust. Baixe `CaixaVisaoComputacional-Windows-Rust-x64.zip` na release `rust-v0.2.0`, extraia todos os arquivos para uma pasta local e abra `CaixaVisaoComputacional.exe`. O pacote inclui os modelos e as DLLs necessários. A instalação exige Windows x64 com Media Foundation, uma câmera disponível e um driver gráfico com suporte a OpenGL. O caixa abre em tela cheia, sem moldura.

O fluxo de operação é o mesmo da versão Python. Selecione ou atualize a câmera, ajuste a confiança com **−** e **+**, e apresente um produto. O carrinho mostra quantidade, preço e subtotal. **Remover** exclui a linha e **Nova compra** exibe o total e limpa o carrinho. Marque **Finalizar compra com gesto (2 mãos)** para usar o gesto treinado. Use `Esc` para sair.

- A interface usa a disposição da versão Python, com o vídeo à esquerda, os cartões à direita, a tabela do carrinho e as barras de progresso. O Windows Media Foundation captura os quadros em uma tarefa separada. O modo **Vídeo fluido** alterna entre o quadro mais recente e o quadro usado pelo detector.
- O YOLO26 usa ONNX Runtime ou OpenVINO em CPU. A entrada é ajustada ao tamanho fixo do modelo com preenchimento e as coordenadas das caixas são projetadas de volta para o quadro original.
- O OCR usa modelos PP-OCRv4 de detecção e reconhecimento distribuídos pelo RapidOCR. A preparação da imagem, a localização dos trechos de texto, a decodificação e a pontuação são implementadas em Rust. A leitura usa o recorte do YOLO por padrão. O resultado pode variar com relação ao motor RapidOCR em Python, especialmente em texto invertido.
- O MediaPipe Tasks é chamado pela interface C. A normalização dos 21 pontos de cada mão é igual à da versão Python. O Rust executa os pesos exportados da MLP e usa o mesmo limiar e tempo de confirmação do gesto.

Rode o executável com `--debug` para ver as pontuações do YOLO, do desempate pelo centro e do OCR. Para alterar valores suportados sem compilar, copie `config_override.example.json` para `config_override.json` ao lado do executável. As chaves aceitas estão em [rust/defaults.json](rust/defaults.json). O catálogo é a chave `PRODUCTS` e as palavras de OCR ficam em `ocr_keywords.json`.

### Selecionar o OpenVINO

Para usar o OpenVINO em CPU com o `best.onnx` incluído, adicione `"MODEL_BACKEND": "openvino"` ao `config_override.json` e reinicie. O catálogo e as regras de leitura continuam os mesmos.

Para usar um modelo exportado pelo Ultralytics em OpenVINO IR, coloque juntos o `.xml`, o `.bin` com o mesmo nome-base e o `metadata.yaml`. Aponte `MODEL_PATH` para o `.xml` ou para a pasta que contém um único `.xml`. Com `"MODEL_BACKEND": "auto"`, o programa escolhe o OpenVINO para IR e ONNX Runtime para ONNX. A opção `"onnxruntime"` seleciona explicitamente ONNX Runtime.

O modelo em `Programa/best_openvino_model` é de doces. Para usá-lo no caixa, defina os nomes e preços dessas classes em `PRODUCTS` no mesmo arquivo de configuração. O aplicativo espera um YOLO26 end-to-end com entrada FP32 quadrada fixa em NCHW e saída FP32 `[1, N, 6]`.

### Compilar e empacotar

Use Rust com alvo `x86_64-pc-windows-msvc`, Visual Studio Build Tools com C++ e o Windows SDK. Rode a partir da raiz do repositório.

```powershell
cargo build --release --locked
cargo test --release --locked
python -m venv .release-work/venv
$py = '.release-work/venv/Scripts/python.exe'
& $py -m pip install --no-deps 'mediapipe==1.0.1' 'onnxruntime==1.27.0' 'openvino==2026.2.1'
& $py -m pip download --no-deps 'rapidocr-onnxruntime==1.4.4' -d .release-work
```

O ambiente Python acima serve para obter as bibliotecas nativas, avisos e modelos. O executável Rust roda sem Python. Baixe a licença do RapidOCR como `.release-work/RapidOCR-LICENSE.txt`. Use o caminho do redistribuível x64 do Visual C++ instalado com o Visual Studio em `$crt`. A pasta deve conter `msvcp140.dll`, `msvcp140_1.dll`, `vcruntime140.dll` e `vcruntime140_1.dll`. [Licença do RapidOCR](https://github.com/RapidAI/RapidOCR/blob/main/LICENSE)

```powershell
$site = '.release-work/venv/Lib/site-packages'
./rust/package.ps1 `
  -MediaPipeDll "$site/mediapipe/tasks/c/libmediapipe.dll" `
  -MediaPipeNotices "$site/mediapipe-1.0.1.dist-info/licenses" `
  -OnnxRuntimeDir "$site/onnxruntime" `
  -OpenVinoDir "$site/openvino" `
  -OpenVinoLicense "$site/openvino-2026.2.1.dist-info/licenses/LICENSE" `
  -VcRuntimeDir $crt `
  -OcrWheel '.release-work/rapidocr_onnxruntime-1.4.4-py3-none-any.whl' `
  -OcrLicense '.release-work/RapidOCR-LICENSE.txt'
```

O script monta a distribuição em `.release-work`, gera o ZIP e escreve o SHA-256 ao lado. A exportação `rust/gesture_mlp.json` já está versionada. Depois de treinar uma nova MLP com a ferramenta Python, execute `python rust/tools/export_gesture.py Programa/modelo/gesture_model.joblib rust/gesture_mlp.json` num ambiente com joblib e scikit-learn compatíveis com o modelo.

Os modos de diagnóstico `--check-fullscreen`, `--check-models`, `--check-camera`, `--check-image caminho.png` e `--check-ocr caminho.png` testam a cobertura do monitor, os modelos, a captura ou uma imagem conhecida. Eles devolvem um código de saída e escrevem em stdout ou stderr, capturáveis via redirecionamento do processo no PowerShell.

## Abrir a versão Python no Windows

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
