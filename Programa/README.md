# Caixa Autônomo com Visão Computacional

Protótipo de PDV (ponto de venda) que detecta produtos por visão
computacional (YOLO) em vez de código de barras.

## Estrutura

```
config.py                 -> preços, classes, thresholds, parâmetros de desempenho (edite aqui)
camera_manager.py         -> enumeração de câmeras + thread de captura de vídeo
detector.py                -> thread de inferência YOLO (.pt ou .onnx)
scoring.py                 -> sistema de pontos (YOLO + centro + OCR) que decide o produto
ocr.py                     -> OCR do rótulo (RapidOCR/onnxruntime) em thread própria
cart.py                    -> lógica do carrinho
ui_styles.py                -> visual (QSS)
sound.py                   -> gera o bip de "produto adicionado"
gesture_features.py        -> extração de landmarks de mão (compartilhado app <-> treino)
mediapipe_utils.py          -> camada sobre a API atual do MediaPipe (HandLandmarker)
gesture.py                  -> thread opcional: gesto de finalizar compra (MediaPipe + MLP)
train_gesture_model.py     -> script standalone pra coletar amostras e treinar o gesto
main.py                    -> aplicação principal / interface
modelo/best.onnx           -> coloque aqui o modelo YOLO exportado
modelo/gesture_model.joblib -> gerado por train_gesture_model.py (opcional)
```

## 1. Instalação

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

## 2. Exportar o modelo treinado para ONNX

No Colab (ou localmente), depois do treino:

```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx", opset=12, simplify=True, imgsz=640)
```

Isso gera `best.onnx`. Coloque esse arquivo em `modelo/best.onnx`
(ou ajuste `MODEL_PATH` em `config.py`).

> Se preferir testar antes com o `.pt` original, só apontar `MODEL_PATH`
> para o `.pt` — o `detector.py` usa a classe `YOLO` da ultralytics, que
> aceita os dois formatos sem mudar nada no código.

## 3. Ajustar preços e classes

Edite o dicionário `PRODUCTS` em `config.py`. Duas coisas para revisar
antes da apresentação:

- **"Coca 600ml" e "Milho"** estão com preço `0.00` (placeholder) — não
  foi informado o valor.
- **"3 Corações" e "Feijão Vermelho"**: no relatório de treino e no log
  mais recente, essas classes apareceram duplicadas (mesmo produto que
  "Café" e "Feijão", respectivamente), com poucas imagens/métricas
  ruins. O código já mapeia as duas para o mesmo produto/preço para não
  quebrar o carrinho, mas o ideal é **corrigir isso no dataset e
  retreinar**.

## 3.1 Gesto para finalizar a compra (opcional)

Pra manter a interação 100% pela câmera (sem mouse/teclado), o app
suporta finalizar a compra levantando as duas mãos num gesto treinado
por você, usando o **MediaPipe HandLandmarker** (API atual do
MediaPipe - `mp.solutions` foi removida nas versões mais novas, então
usamos essa) pra rastrear os landmarks e uma **MLP (rede neural) do
scikit-learn** (nada de TensorFlow aqui) pra reconhecer o gesto. É
opcional e desligado por padrão, ativado por um botão na interface
("Finalizar compra com gesto (2 mãos)"), do mesmo jeito que o de
"vídeo fluido".

**Por que treinar com exemplos "negativos" também?** Um classificador
não aprende "isso é o gesto X" olhando só pra exemplos do gesto X - ele
aprende a diferença entre duas coisas. Sem exemplos do que NÃO é o
gesto (mãos paradas, segurando produto, gesticulando à toa), o modelo
nunca vê contraste nenhum e tende a ficar com confiança alta pra
qualquer coisa que apareça na câmera, disparando à toa durante o uso
normal do caixa.

**Passo 1 - Treinar o gesto:**

Na primeira execução, dois modelos do MediaPipe são baixados
automaticamente e salvos em `modelo/` (precisa de internet só nessa
primeira vez): `hand_landmarker.task` (mãos, usado também pelo app em
produção) e `face_landmarker.task` (rosto, usado **só** por esse
script, pra detectar o piscar - não precisa ir no `.exe` final, é só
ferramenta de coleta de dados). Lembre de incluir `hand_landmarker.task`
junto com `best.onnx` e `gesture_model.joblib` quando copiar a pasta
`modelo/` pro `.exe` (seção 6) - o `face_landmarker.task` pode ficar
de fora.

```bash
python train_gesture_model.py
```

A janela abre em **tela cheia** por padrão (alterne com `F`), com os
controles:

| Tecla | Ação |
|---|---|
| `P` | captura uma amostra **positiva** e define o "modo do piscar" como positivo |
| `O` | captura uma amostra **negativa** e define o "modo do piscar" como negativo |
| Piscar | captura uma amostra no **modo atual** (o que foi definido pela última vez com `P` ou `O`) - útil quando as duas mãos estão ocupadas: aperte `P`/`O` uma vez pra escolher o modo, solte o teclado, segure a pose e vá piscando |
| `S` | salva as amostras capturadas em `modelo/gesture_dataset.csv` |
| `T` | salva e treina o modelo, gerando `modelo/gesture_model.joblib` |
| `F` | alterna tela cheia |
| `ESC` | sai |

A detecção de piscar usa o **Eye Aspect Ratio (EAR)** — a mesma
técnica clássica de detectores de sonolência — calculado a partir da
malha facial do MediaPipe. O valor de EAR aparece na tela em tempo
real, útil pra calibrar `BLINK_EAR_THRESHOLD` em `config.py` se o
piscar não estiver sendo detectado direito com a sua iluminação/rosto
(o padrão é `0.21` - abaixe um pouco se estiver disparando sozinho
com o olho aberto, aumente se não estiver detectando o piscar).

Só é possível capturar quando **exatamente 2 mãos** estão visíveis
(mesma regra usada em produção) - inclusive pelo piscar: se você
piscar sem as duas mãos prontas, nada é capturado. Recomendado: bem
mais amostras negativas que positivas (ex: 100 positivas, 300+
negativas variadas - mãos paradas, segurando produto, acenando,
etc.), pra o modelo não disparar à toa. Pode rodar o script várias
vezes pra ir acumulando amostras no mesmo `gesture_dataset.csv` antes
de treinar.

**Passo 2 - Usar no app:** com `modelo/gesture_model.joblib` gerado,
basta marcar o checkbox "Finalizar compra com gesto" na interface.
Segurando o gesto por `GESTURE_HOLD_SECONDS` (1.5s por padrão), a
compra é finalizada: mostra o total, limpa o carrinho e volta pro
próximo cliente. Um pequeno indicador de progresso aparece abaixo do
checkbox, e os pontos das mãos detectadas aparecem no vídeo (verde
quando o gesto está sendo reconhecido).

**Nota de desempenho:** essa é mais uma thread de inferência rodando
em paralelo à detecção de produtos - em hardware fraco, ative só na
hora de treinar a apresentação de fato, e deixe desligado no resto do
tempo se nao for usar.

## 3.2 Trocar configurações sem recompilar o .exe

Qualquer variável de `config.py` pode ser sobrescrita **depois de
gerado o `.exe`**, sem precisar rodar o PyInstaller de novo:

1. Copie `config_override.example.json` para `config_override.json`,
   deixando-o do lado do `.exe` (mesma pasta).
2. Edite os valores que quiser mudar, por exemplo:
   ```json
   { "CONF_THRESHOLD_DEFAULT": 0.5, "ENABLE_GESTURE_CHECKOUT_DEFAULT": true }
   ```
3. Abra o programa de novo — o `app.log` mostra quais overrides foram
   aplicados.

## 3.3 Sistema de pontos (YOLO + OCR) e modo `--debug`

Pra não depender só do YOLO, cada classe recebe pontos de duas fontes:

| Fonte | Pontos | Regra |
|---|---|---|
| YOLO | até 40 | `40 x confiança` da melhor caixa da classe |
| Desempate | até 5 | se classes acima da confiança mínima estão a até 8 pontos percentuais da melhor, a mais perto do centro da imagem ganha `5 x (1 - distância/distância_máxima)` |
| OCR | até 60 | lê o rótulo e procura palavras de `OCR_KEYWORDS` (2 palavras = 60, 1 = 30; tolera erros de leitura e acentos) |

A classe com maior total é a candidata, desde que
`total >= 40 x precisão mínima`. Quando uma leitura atual do OCR
confirma o produto detectado pelo YOLO, ele entra no carrinho assim que
o OCR responde. Nos demais casos, a candidata precisa ficar 1.5s. Os botões `+`/`-`
continuam valendo e, com o OCR desligado, o comportamento é idêntico
ao de antes. O OCR pode arbitrar entre classes parecidas (ex: o YOLO
oscila entre milho e molho de tomate, mas o rótulo diz "predilecta").
Se só o OCR reconhecer uma classe (sem caixa do YOLO pra ela), ela
também pode ser aceita.

- **Oculto por padrão.** `python main.py --debug` (ou o `.exe` com
  `--debug`) mostra um painel com YOLO / centro / OCR / total de cada
  classe, o texto lido e o tempo do OCR - útil pra ajustar
  `OCR_KEYWORDS` (palavras específicas de cada produto, sem repetir
  palavras que apareçam em outros rótulos; "Milho" só tem uma palavra,
  adicione a marca da sua lata).
- **Desempenho:** o OCR roda em thread própria, só sobre a caixa do
  YOLO (`OCR_USE_BOX_CROP`), no máximo 1 leitura a cada
  `OCR_INTERVAL_SECONDS` e só quando há algo na cena. Em PC muito fraco,
  aumente o intervalo ou use `ENABLE_OCR = False` (via
  `config_override.json`, sem recompilar).
- Os modelos do OCR já vêm dentro do pacote `rapidocr` (funciona
  offline, sem PyTorch).

## 4. Rodar

```bash
python main.py
```

- `ESC` fecha o programa.
- `python main.py --debug` mostra o painel com a pontuação de cada classe (seção 3.3).
- "Nova compra" limpa o carrinho na hora (sem confirmação) e mostra um
  aviso que some sozinho em `TOAST_SECONDS` (3.5s, em `config.py`).
- Botões `+`/`-` ajustam a confiança mínima em tempo real.
- Combo de câmera troca entre os dispositivos detectados (o OpenCV não
  informa de forma confiável se é câmera USB ou integrada — os nomes
  são genéricos "Câmera 0", "Câmera 1"... teste antes do dia da
  apresentação para saber qual índice é qual).
- O lançamento é imediato após a concordância entre YOLO e OCR. Quando
  a confirmação por OCR falta ou é ambígua, o caixa exige **1.5s
  contínuos** de detecção. Nos dois casos há um **cooldown de 2.5s**
  depois do lançamento (os tempos são ajustáveis em `config.py`).

## 5. Sobre desempenho em hardware fraco

A opção mais importante para isso já está implementada:
`DESYNC_BBOXES` em `config.py` (com checkbox equivalente na interface).

- **Ligado (padrão):** a câmera é exibida sempre a ~30fps de forma
  fluida; as caixas do YOLO são desenhadas por cima usando a última
  detecção disponível, mesmo que ela tenha sido calculada em um frame
  anterior. Ou seja, o vídeo nunca trava, só a posição da caixa pode
  ficar levemente atrasada — imperceptível quando o produto fica
  parado na frente da câmera (que é o caso de uso real aqui).
- **Desligado:** a imagem só é redesenhada quando uma nova detecção
  termina, então a taxa de exibição cai para a taxa de inferência
  (mais "travado" em hardware fraco, porém 100% sincronizado).

Outras opções, se mesmo assim o desempenho não for suficiente na
máquina da apresentação:

1. **Reduzir `imgsz`** em `config.py` (`INFERENCE_IMGSZ`), ex: de 640
   para 480 ou 416 — perde um pouco de precisão, ganha velocidade.
2. **Trocar `onnxruntime` puro no lugar da ultralytics**: hoje o
   `detector.py` usa `YOLO(caminho_onnx)` da própria ultralytics, que
   por baixo dos panos ainda depende do PyTorch instalado (mesmo
   rodando um `.onnx`). Isso deixa a instalação/exe mais pesados e a
   inicialização mais lenta. Se sobrar tempo, migrar a inferência para
   `onnxruntime` puro (pré-processamento/NMS manuais, sem PyTorch)
   reduz bastante o tamanho e o tempo de start — vale a pena se o
   notebook da apresentação for realmente muito fraco.
3. Testar `format="onnx", half=True` na exportação (precisão
   reduzida/FP16) — funciona melhor em GPU, em CPU o ganho é menor.

## 6. Gerar o .exe (PyInstaller)

Se usar a série Python 3.12, empacote com Python 3.12.1 ou posterior
para o SciPy carregar corretamente no executável.

```bash
pip install pyinstaller

pyinstaller --noconfirm --windowed --name CaixaVisaoComputacional main.py
```

Se for usar o **gesto de finalizar compra** (MediaPipe), adicione
`--collect-all mediapipe`. O OCR (`rapidocr`) também precisa de
`--collect-all rapidocr`, pra levar os modelos e o `config.yaml` do pacote:

```bash
pyinstaller --noconfirm --windowed --name CaixaVisaoComputacional ^
  --collect-all rapidocr --collect-all mediapipe main.py
```

(no Linux/Mac troque `^` por `\` na quebra de linha)

`--collect-all` ajuda porque bibliotecas como o MediaPipe fazem
bastante import dinâmico/carregam arquivos de dados internos que o
PyInstaller não detecta sozinho por análise estática do código — sem
isso, o `.exe` pode até compilar sem erro, mas falhar silenciosamente
(ou com `ModuleNotFoundError`/erro ao carregar o modelo) só quando
aquele código é executado em tempo real. Se não for usar o gesto, pode
deixar de fora.

Depois de gerar, **copie a pasta `modelo/` manualmente** para dentro de
`dist/CaixaVisaoComputacional/` (do lado do `.exe`):

```
dist/CaixaVisaoComputacional/
    CaixaVisaoComputacional.exe
    modelo/
        best.onnx
    ...
```

`config.py` procura o modelo na pasta do próprio executável quando
rodando como `.exe` (`BASE_DIR = os.path.dirname(sys.executable)`), e
na pasta do script quando rodando via `python main.py`. Ou seja: **sim,
dá pra trocar o `best.onnx` depois de gerado o `.exe`** — basta
substituir o arquivo dentro de `dist/CaixaVisaoComputacional/modelo/`
e abrir o programa de novo, sem precisar recompilar nada.

Recomendações:
- Prefira **não** usar `--onefile` — o modo `--onedir` (padrão acima)
  abre bem mais rápido, importante numa demonstração ao vivo, e é o
  que permite trocar o modelo sem recompilar.
- Teste o `.exe` gerado na própria máquina que vai ser usada na
  apresentação, com antecedência — PyTorch/ultralytics empacotados
  costumam ficar grandes (vários GB) e a primeira abertura pode
  demorar.
- Copie a pasta `dist/CaixaVisaoComputacional` inteira para o outro
  computador (não só o `.exe`).

### Erro `IndexError: tuple index out of range` em `dis.py`

Se o `pyinstaller` falhar com um traceback terminando em
`dis.py ... IndexError: tuple index out of range`, isso é um bug
conhecido do próprio módulo `dis` na versão **exata 3.10.0** do
Python (corrigido em patches seguintes) — não tem relação com o
código deste projeto. Solução: atualize o Python do venv para
3.10.11+ (ou 3.11/3.12), recrie o venv, reinstale as dependências e
o PyInstaller (`pip install -U pyinstaller`), e gere o `.exe` de novo.

## 7. Ideias para melhorar a precisão do modelo depois

Olhando o relatório do seu treino (mAP50-95 geral de 0.728, best na
epoch 43 e early stopping na 93), alguns pontos:

- **"3 Corações"**: corrigir/mesclar a classe duplicada (ver seção 3)
  deve, sozinho, já melhorar a média geral, já que ela está puxando o
  mAP pra baixo com apenas 1 imagem/instância.
- **Feijão** (mAP50-95 = 0.797, mas P menor: 0.848) e **Coca 600ml**
  (0.729) foram as classes mais fracas entre as "boas" — vale
  fotografar mais exemplos delas, em ângulos/fundos que ainda não
  usou, incluindo mais casos com objetos parcialmente ocluídos (já que
  no cenário real o cliente pode segurar o produto de qualquer jeito
  na frente da câmera).
- Bom já ter fotografado individual + em grupos de 3 — para reforçar,
  vale incluir também fotos com **mais de 3 produtos ao mesmo tempo**
  e com **oclusão parcial** (mão segurando o produto, produto
  parcialmente fora do quadro), que é o cenário real de uso no caixa.
- `patience=50` interrompeu o treino relativamente cedo (best na
  epoch 43). Com dataset maior/mais balanceado, vale tentar
  `patience=100` ou mais.
- Testar `yolo26s` (em vez de `yolo26n`) só para treino/validação, para
  ver se a precisão sobe — depois exportar a versão `n` (mais leve)
  para a demonstração se a diferença não compensar o custo de
  inferência.
