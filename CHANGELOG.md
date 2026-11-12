# Histórico de versões

## rust-v0.2.3 e python-2026.11.12

- `--classifier` ativa o classificador Keras do projeto ou carrega um modelo H5 indicado pelo usuário.
- Pontuação de até 30 pelo YOLO, 30 pelo classificador, 35 pelo OCR proporcional às palavras-chave e 5 pela posição.
- A barra sobe rapidamente a 100% depois do lançamento e permanece cheia por 0,4 s.

## rust-v0.2.2 e python-2026.11.11

- `--zone` delimita a área de leitura do YOLO com um retângulo que some após a seleção.
- `--hand-block` impede o lançamento de produtos enquanto o MediaPipe detectar uma mão, inclusive na confirmação imediata por OCR.

## rust-v0.2.1

- Lança o item no carrinho assim que uma leitura atual do OCR confirma o produto reconhecido pelo YOLO.
- Mantém a espera de detecção em caso de divergência, empate entre produtos ou falta de confirmação. O intervalo contra duplicidade vale para ambos os caminhos.

## rust-v0.2.0

- Abertura sem moldura, com a área do caixa ocupando o monitor e `Esc` para sair.
- Interface ajustada pela referência Python, com Segoe UI, título proporcional à tela, cartões, carrinho com cabeçalho e grade e os indicadores originais de leitura.
- OpenVINO em CPU para o modelo ONNX e para exportações YOLO26 em OpenVINO IR. O runtime acompanha o ZIP de Windows x64.

## rust-v0.1.0

- Port do aplicativo de caixa para Rust em Windows x64, com vídeo à esquerda, controles e carrinho à direita, abertura em tela cheia e os mesmos atalhos e ações de compra.
- Inferência YOLO26 pelo ONNX Runtime, captura pelo Media Foundation, OCR local com modelos PP-OCRv4 e rastreamento de mãos via MediaPipe Tasks.
- Exportação da MLP treinada e inferência em Rust com os mesmos 126 valores de entrada. O gesto e o botão **Nova compra** compartilham o fechamento do carrinho.
- ZIP portátil, verificação SHA-256, instruções de compilação e modos de diagnóstico para câmera e modelos.

## python-2026.11.09

- Lança o item no carrinho assim que YOLO e uma leitura atual e sem empate do OCR confirmam o mesmo produto.
- Mantém o tempo de detecção nos outros casos e o intervalo contra duplicidade após cada lançamento.
- Inclui os modelos e arquivos de configuração do RapidOCR no pacote PyInstaller.

## python-2026.11.03

Primeira publicação no GitHub. O identificador usa a data do ZIP do PyInstaller existente no projeto.

- Reconhecimento de embalagens pelo YOLO26, com ajustes de confiança, pontuação por centralidade e OCR opcional no código-fonte.
- Exibição de câmera com caixas de detecção, progresso de leitura, prevenção de lançamentos duplicados, carrinho e fechamento com total.
- Fechamento opcional por gesto de duas mãos com MediaPipe e uma MLP treinada, além da ferramenta de coleta e treinamento.
- Documentação de instalação, funcionamento, configuração e empacotamento. O ZIP publicado conserva o executável e os pesos da versão de 3 de novembro. As cópias das amostras de treino do gesto foram retiradas do ZIP de distribuição.
