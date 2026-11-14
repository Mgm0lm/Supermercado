# Supermercado

O Supermercado é um caixa autônomo para Windows que usa a câmera para reconhecer embalagens e lançar produtos no carrinho. O **YOLO26** localiza o produto e sugere uma classe. O **OCR** lê o rótulo, normalmente dentro da caixa detectada, e compara o texto com as palavras-chave do catálogo. O **classificador Keras**, ativado com `--classifier`, analisa o recorte da caixa válida de maior confiança do YOLO e dá um segundo palpite sobre o produto.

Selecione a câmera e apresente um produto de cada vez. Os botões **−** e **+** ajustam a confiança mínima. Use **Remover** para excluir uma linha do carrinho e **Nova compra** para mostrar o total e iniciar outro atendimento. Com **Finalizar compra com gesto (2 mãos)** ativado, o MediaPipe extrai 21 pontos de cada mão e um classificador de gestos aciona o fechamento. Pressione **Esc** para sair.

## Python ou Rust

| | Python (`main`) | Rust (`Rust`) |
|---|---|---|
| Interface e câmera | PySide6 e OpenCV | egui/eframe e Windows Media Foundation |
| YOLO26 | Ultralytics, com o modelo ONNX incluído por padrão | ONNX Runtime ou OpenVINO em CPU |
| OCR | RapidOCR | Modelos PP-OCRv4 executados via ONNX Runtime, com processamento em Rust |
| Gesto | MediaPipe e classificador scikit-learn | MediaPipe pela interface C e pesos exportados para JSON |
| Distribuição | PyInstaller com o ambiente Python incluído | Executável nativo com modelos e DLLs |

As duas versões abrem em tela cheia e usam o mesmo fluxo de caixa. O pacote Rust roda sem instalar Python para o uso normal. Em ambos os pacotes, abrir um classificador `.h5` externo exige um ambiente Python para a conversão. O modelo padrão já vem convertido.

Para usar uma versão pronta no **Windows x64**, baixe o ZIP correspondente, extraia a pasta inteira e execute `CaixaVisaoComputacional.exe`. A versão Rust também precisa de um driver gráfico com suporte a OpenGL. [Downloads das versões Python e Rust](https://github.com/Mgm0lm/Supermercado/releases)

## Argumentos

| Argumento | Efeito |
|---|---|
| `--classifier` | Ativa o classificador Keras do projeto, distribuído em ONNX. |
| `--classifier caminho.h5` ou `--classifier caminho.onnx` | Usa outro classificador. Coloque um `labels.txt` ao lado do modelo. |
| `--zone` | Pausa o YOLO até você arrastar um retângulo sobre o vídeo. O contorno some ao soltar o mouse. Use **Redefinir área de detecção** para desenhar outro recorte. Se trocar de câmera ou reiniciar, selecione a área novamente. |
| `--hand-block` | Suspende o lançamento se o MediaPipe detectar uma mão em qualquer parte do quadro ou se faltar uma leitura recente. A leitura do produto recomeça ao retirar a mão. Funciona com o fechamento por gesto desativado. |
| `--debug` | Mostra a pontuação por classe e o texto lido pelo OCR. |

No PowerShell, dentro da pasta extraída, você pode combinar as opções:

```powershell
.\CaixaVisaoComputacional.exe --classifier --zone --hand-block --debug
```

## Sistema de pontuação

| Fonte | Máximo | Regra |
|---|---:|---|
| YOLO26 | 30 | Multiplica 30 pela confiança da melhor caixa da classe. Só entram caixas que atingem o limiar de confiança. |
| Classificador Keras | 30 | Multiplica 30 pela confiança da classe prevista. O resultado precisa atingir o limiar. A classe `Nada` e nomes fora do catálogo são descartados. |
| OCR | 35 | Divide a soma dos acertos pelo total de palavras-chave únicas da classe em `ocr_keywords.json` e multiplica por 35. A repetição de uma palavra não soma pontos. Um acerto aproximado vale a sua semelhança, desde que passe pelo filtro de OCR. |
| Posição | 5 | Quando duas ou mais classes estão a até 8 pontos percentuais da melhor confiança do YOLO, a caixa mais central ganha um bônus proporcional à proximidade do centro. |
| **Total** | **100** | |

Dois acertos exatos entre quatro palavras esperadas, por exemplo, valem **17,5 pontos** de OCR. Para o lançamento por pontuação, vence a classe elegível com o maior total desde que atinja `30 × confiança mínima`. Com o limiar inicial de 60%, o mínimo é 18 pontos.

Quando uma leitura recente e sem empate do OCR confirma o mesmo produto detectado pelo YOLO, o item entra no carrinho imediatamente. Nos demais casos, a classe vencedora precisa permanecer candidata por **1,5 s**. Após cada lançamento, o caixa aguarda **2,5 s** para ler outro item e só aceita novamente a mesma classe após ela sair da cena pelo intervalo configurado. A barra chega a 100% em cerca de 120 ms e fica cheia por **0,4 s** como efeito visual.

Para ajustar preços, palavras-chave ou tempos, copie `config_override.example.json` para `config_override.json` ao lado do executável ou de `main.py` e reinicie. Ajuste o catálogo na chave `PRODUCTS` e as palavras do rótulo em `ocr_keywords.json`. [Configuração Python](Programa/config.py) · [Valores padrão do Rust](https://github.com/Mgm0lm/Supermercado/blob/Rust/rust/defaults.json)

## Python — preparar, testar e empacotar

Use o **Python 3.12 x64 a partir da versão 3.12.1** e o PowerShell. Os comandos usam diretamente o Python do ambiente virtual, dispensando ativação. [Ambientes virtuais no Python](https://docs.python.org/3.12/library/venv.html)

```powershell
git clone --branch main https://github.com/Mgm0lm/Supermercado.git Supermercado-Python
Set-Location Supermercado-Python\Programa
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt pyinstaller
& $py main.py --help
& $py -m unittest discover -s tests -v
& $py main.py --classifier --debug
```

Os testes automatizados cobrem pontuação, confirmação por OCR e bloqueio por mão. Depois de sair do caixa, rode o PyInstaller na mesma pasta. Use um PowerShell normal do Windows para que o empacotador encontre as DLLs do ambiente Python e do sistema.

```powershell
& $py -m PyInstaller --noconfirm CaixaVisaoComputacional.spec
$dist = '.\dist\CaixaVisaoComputacional'
New-Item -ItemType Directory -Force -Path "$dist\modelo" | Out-Null
foreach ($file in 'best.onnx','gesture_model.joblib','hand_landmarker.task','keras_model.onnx','labels.txt') {
    Copy-Item -LiteralPath ".\modelo\$file" -Destination "$dist\modelo"
}
Copy-Item -LiteralPath '.\ocr_keywords.json','.\config_override.example.json' -Destination $dist
& "$dist\CaixaVisaoComputacional.exe" --classifier --zone --hand-block --debug
```

Confira a abertura e a câmera no executável. Se ocorrer um erro, consulte `app.log` ao lado dele. Para distribuir, compacte a pasta `dist\CaixaVisaoComputacional` inteira, incluindo `_internal` e `modelo`. [Guia técnico Python](Programa/README.md)

## Rust — compilar, empacotar e testar

Instale o Rust pelo **rustup** para o alvo `x86_64-pc-windows-msvc`. No Visual Studio Build Tools, instale **Desenvolvimento para desktop com C++** com o compilador MSVC x64, Windows SDK e o redistribuível Visual C++ x64. Use também Python 3.12 x64 para baixar os componentes de distribuição em um ambiente virtual separado. [Alvo Windows MSVC no Rust](https://doc.rust-lang.org/rustc/platform-support/windows-msvc.html) · [Componentes do Visual Studio Build Tools](https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools)

```powershell
git clone --branch Rust https://github.com/Mgm0lm/Supermercado.git Supermercado-Rust
Set-Location Supermercado-Rust
rustup toolchain install stable-x86_64-pc-windows-msvc
cargo +stable-x86_64-pc-windows-msvc test --release --locked
cargo +stable-x86_64-pc-windows-msvc build --release --locked

py -3.12 -m venv .release-work\venv
$py = (Resolve-Path .\.release-work\venv\Scripts\python.exe).Path
& $py -m pip install --upgrade pip
& $py -m pip install --no-deps 'mediapipe==1.0.1' 'onnxruntime==1.27.0' 'openvino==2026.2.1'
& $py -m pip download --no-deps 'rapidocr-onnxruntime==1.4.4' -d .release-work
```

Salve a licença do RapidOCR em `.release-work\RapidOCR-LICENSE.txt`. Defina `$crt` com a pasta do redistribuível x64 do Visual C++ que contém `msvcp140.dll`, `msvcp140_1.dll`, `vcruntime140.dll` e `vcruntime140_1.dll`. Ela fica sob `VC\Redist\MSVC\<versão>\x64` na instalação do Visual Studio Build Tools. [Licença do RapidOCR](https://github.com/RapidAI/RapidOCR/blob/main/LICENSE)

```powershell
$crt = 'C:\caminho\para\a\pasta\do\redistribuivel\x64'
$site = '.release-work/venv/Lib/site-packages'
.\rust\package.ps1 `
  -MediaPipeDll "$site/mediapipe/tasks/c/libmediapipe.dll" `
  -MediaPipeNotices "$site/mediapipe-1.0.1.dist-info/licenses" `
  -OnnxRuntimeDir "$site/onnxruntime" `
  -OpenVinoDir "$site/openvino" `
  -OpenVinoLicense "$site/openvino-2026.2.1.dist-info/licenses/LICENSE" `
  -VcRuntimeDir $crt `
  -OcrWheel '.release-work/rapidocr_onnxruntime-1.4.4-py3-none-any.whl' `
  -OcrLicense '.release-work/RapidOCR-LICENSE.txt'
```

O script reúne o executável de `target\release`, os modelos e as DLLs em `.release-work\CaixaVisaoComputacional-Rust` e gera o ZIP com seu SHA-256. Execute os diagnósticos contra o pacote montado:

```powershell
$exe = '.\.release-work\CaixaVisaoComputacional-Rust\CaixaVisaoComputacional.exe'
foreach ($check in '--check-models','--check-camera','--check-fullscreen') {
    $run = Start-Process -FilePath $exe -ArgumentList $check -Wait -PassThru `
        -RedirectStandardOutput '.release-work\diagnostico.out' `
        -RedirectStandardError '.release-work\diagnostico.err'
    Get-Content '.release-work\diagnostico.out','.release-work\diagnostico.err'
    if ($run.ExitCode -ne 0) { throw "Falhou: $check (código $($run.ExitCode))" }
}
& $exe --classifier --zone --hand-block --debug
```

| Argumento Rust | Diagnóstico |
|---|---|
| `--check-models` | Executa YOLO, MediaPipe e o classificador de gestos em um quadro sintético. |
| `--check-camera` | Captura 10 quadros pela primeira câmera. |
| `--check-fullscreen` | Verifica se o caixa ocupa a tela inteira. |
| `--check-image caminho.png` | Informa detecções e mãos em uma imagem. Com `--classifier`, também classifica a imagem inteira para diagnóstico. |
| `--check-ocr caminho.png` | Lê o texto de uma imagem. |
| `--check-mlp caminho.json` | Testa o classificador de gestos com uma lista de amostras de 126 valores cada. |
| `--assets pasta` | Indica a pasta que contém os modelos, configurações e DLLs do pacote. |

Para usar **OpenVINO em CPU** com o `best.onnx` no Rust, defina `"MODEL_BACKEND": "openvino"` em `config_override.json`. Para um modelo OpenVINO IR, mantenha juntos o `.xml`, o `.bin` correspondente e o `metadata.yaml`. Aponte `MODEL_PATH` para o `.xml` ou para a pasta que contém um único `.xml`. O modo `auto` usa OpenVINO para IR e ONNX Runtime para ONNX.

## Teste manual nas duas versões

1. Abra o executável com `--classifier --zone --hand-block --debug`. Delimite uma área sobre o vídeo e confira se o contorno some ao soltar o mouse.
2. Apresente um produto dentro da área. Veja as parcelas da pontuação no painel de depuração. Com a mão no quadro, o item aguarda. Ao retirar a mão, verifique o lançamento e o breve intervalo da barra em 100%.
3. Deixe o mesmo produto diante da câmera e confira se ele entra somente uma vez. Retire-o antes de repetir a leitura.
4. Confira o total com **Nova compra**. Depois ative o fechamento por gesto e teste com as duas mãos.

## Usar outro classificador Keras

Instale o conversor em um ambiente Python. Você pode reaproveitar o ambiente de desenvolvimento da versão Python ou criar um só para o conversor. Rode os comandos a partir da raiz de um clone do projeto:

```powershell
py -3.12 -m venv .release-work\keras-venv
$kerasPy = (Resolve-Path .\.release-work\keras-venv\Scripts\python.exe).Path
& $kerasPy -m pip install keras h5py torch onnx
$env:CLASSIFIER_PYTHON = $kerasPy
```

O modelo deve aceitar uma imagem RGB em formato NHWC com dimensões fixas, pixels normalizados entre −1 e 1 e saída de probabilidades por classe. Crie um `labels.txt` com um índice e nome por linha, começando em zero. Exemplo de linha `0 Miojo`.

No mesmo PowerShell, execute o caixa com `--classifier "C:\modelos\meu_modelo.h5"` para converter ao carregar. Para converter uma vez, use o comando abaixo. Depois passe o caminho do ONNX em `--classifier`, com o mesmo `labels.txt` ao lado.

```powershell
& $kerasPy .\Programa\export_classifier.py 'C:\modelos\meu_modelo.h5' 'C:\modelos\meu_modelo.onnx'
```

O [histórico de versões](CHANGELOG.md) registra as mudanças de cada release.

## Notas

* Esse projeto só foi possível devido a contribuição dos meus professores e colegas de classe.

* O código utilizado se baseia/foi criado por ferramentas de Inteligência Artificial.
