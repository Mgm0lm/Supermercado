# Histórico de versões

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
