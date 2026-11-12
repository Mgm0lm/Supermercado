# Histórico de versões

## python-2026.11.12

- `--classifier` ativa o classificador Keras do projeto ou carrega um modelo H5 indicado pelo usuário.
- Pontuação de até 30 pelo YOLO, 30 pelo classificador, 35 pelo OCR proporcional às palavras-chave e 5 pela posição.
- A barra sobe rapidamente a 100% depois do lançamento e permanece cheia por 0,4 s.

## python-2026.11.11

- `--zone` delimita a área de leitura do YOLO com um retângulo que some após a seleção.
- `--hand-block` impede o lançamento de produtos enquanto o MediaPipe detectar uma mão, inclusive na confirmação imediata por OCR.

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
