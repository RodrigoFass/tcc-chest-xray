---
title: Raio X de tórax (TCC)
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: DenseNet-121 (NIH ChestX-ray14) com Grad-CAM no navegador
---

# Detecção de doenças pulmonares em raio X de tórax

**Protótipo acadêmico. Não usar para diagnóstico.**

Interface de demonstração do Trabalho de Conclusão de Curso de Rodrigo Fassarella: uma DenseNet-121
treinada no NIH ChestX-ray14 devolve, para cada uma das 14 doenças do conjunto, uma probabilidade
estimada (recalibrada na validação) com o limiar de decisão, e mostra o mapa de calor Grad-CAM da
doença escolhida. O modelo roda no navegador de quem acessa, com ONNX Runtime Web: a imagem não é
enviada a nenhum servidor.

Código, experimentos e monografia: <https://github.com/RodrigoFass/tcc-chest-xray>.
