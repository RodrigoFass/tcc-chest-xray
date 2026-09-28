# 1 INTRODUÇÃO

A radiografia de tórax é o exame de imagem mais realizado no mundo [CONFERIR: citar fonte. Candidata,
a verificar antes de usar: RAOOF, S. et al. Interpretation of plain chest roentgenogram. **Chest**,
v. 141, n. 2, p. 545-558, 2012]. Ela é barata, rápida, amplamente
disponível e expõe o paciente a uma dose baixa de radiação, o que a torna o primeiro exame na
investigação de queixas respiratórias, no acompanhamento de pacientes internados e na triagem de
doenças como pneumonia, derrame pleural e tuberculose. A interpretação dessas imagens, porém, está
longe de ser simples: estruturas anatômicas se sobrepõem numa projeção bidimensional, achados
diferentes produzem aspectos parecidos, e a leitura depende da experiência de quem laudou. A
variabilidade entre radiologistas na interpretação de radiografias de tórax é bem documentada
[CONFERIR: citar fonte. Candidata, a verificar antes de usar: HOPSTAKEN, R. M. et al. Inter-observer
variation in the interpretation of chest radiographs for pneumonia in community-acquired lower
respiratory tract infections. **Clinical Radiology**, v. 59, n. 8, p. 743-752, 2004], e em
muitas regiões simplesmente não há radiologistas suficientes para laudar todos os exames em tempo
hábil [CONFERIR: se quiser um número sobre a distribuição de radiologistas no Brasil, citar uma fonte
como a Demografia Médica no Brasil (CFM/USP)].

Nesse contexto, sistemas computacionais de apoio ao diagnóstico podem ajudar a priorizar exames com
maior chance de alteração, oferecer uma segunda opinião e reduzir o tempo até o laudo. Na última
década, o aprendizado profundo (*deep learning*), em especial as redes neurais convolucionais
(*Convolutional Neural Networks*, CNN), transformou a análise automática de imagens médicas (LITJENS
et al., 2017). Dois fatores tornaram isso possível para a radiografia de tórax: a publicação de
grandes conjuntos de dados rotulados, como o NIH ChestX-ray14, com 112.120 imagens (WANG et al., 2017),
e o CheXpert, com mais de 224 mil (IRVIN et al., 2019); e a técnica de transferência de aprendizado,
que permite partir de uma rede já treinada em milhões de imagens naturais (o ImageNet) e ajustá-la ao
domínio médico. O trabalho mais conhecido dessa linha, o CheXNet (RAJPURKAR et al., 2017), usou uma
DenseNet-121 (HUANG et al., 2017) treinada no ChestX-ray14 e relatou, para a detecção de pneumonia,
desempenho comparável ao de radiologistas.

Resultados como esse, no entanto, precisam ser lidos com cuidado. Os rótulos do ChestX-ray14 foram
extraídos automaticamente dos laudos e contêm erros (OAKDEN-RAYNER, 2020); o desempenho medido varia
com a forma de dividir os dados em treino e teste (BALTRUSCHAT et al., 2019); e modelos treinados num
hospital costumam perder desempenho quando aplicados a imagens de outro (ZECH et al., 2018). Além
disso, uma rede neural que devolve apenas um número é pouco útil na prática clínica: o profissional
precisa saber em que região da imagem o modelo se baseou, para julgar se a predição faz sentido. Por
isso, métodos de interpretabilidade, como o Grad-CAM (SELVARAJU et al., 2017), tornaram-se parte
esperada desse tipo de sistema.

Este trabalho reproduz e estende a abordagem do CheXNet com atenção a esses pontos: divisão dos dados
por paciente, intervalos de confiança para todas as métricas, comparação estatística entre variações
do modelo, análise por subgrupos de pacientes, calibração das saídas, mapas de calor avaliados contra
marcações de radiologistas e validação em um conjunto de dados de outra instituição.

## 1.1 Objetivo geral

Desenvolver um sistema capaz de identificar doenças pulmonares em radiografias de tórax por meio de
técnicas de inteligência artificial, que receba uma imagem e forneça, para cada doença, um valor que
indique a chance de ela estar presente, acompanhado de um mapa de calor que mostre as regiões da imagem
que mais influenciaram a decisão. O foco está em três doenças: pneumonia, atelectasia e efusão pleural
(derrame pleural).

## 1.2 Objetivos específicos

a) treinar uma rede DenseNet-121, pré-treinada no ImageNet, para classificar as 14 doenças rotuladas no
   NIH ChestX-ray14, com divisão dos dados por paciente em treino, validação e teste;

b) avaliar o desempenho do modelo com a área sob a curva ROC (AUC) e a área sob a curva
   precisão-revocação (AUPRC), com intervalos de confiança de 95%, além de sensibilidade,
   especificidade e valor preditivo num limiar definido na validação;

c) comparar os resultados com os reportados na literatura, em especial por Wang et al. (2017) e pelo
   CheXNet (RAJPURKAR et al., 2017), discutindo as diferenças de protocolo que tornam a comparação
   aproximada;

d) medir, por meio de experimentos controlados, o efeito da ponderação de classes na função de perda, do
   aumento de dados (*data augmentation*) e da transferência de aprendizado;

e) verificar se o desempenho se mantém em subgrupos de pacientes (sexo, faixa etária e incidência do
   exame) e se os valores produzidos pelo modelo podem ser lidos como probabilidades (calibração);

f) gerar mapas de calor com o Grad-CAM e compará-los com as marcações feitas por radiologistas;

g) avaliar o modelo, sem novo treino, no conjunto de validação do CheXpert, de outra instituição;

h) disponibilizar o modelo numa interface web de demonstração.

## 1.3 Justificativa

A motivação principal é o impacto social. O diagnóstico por imagem do tórax exige especialização, está
sujeito à variabilidade entre profissionais e esbarra na falta de radiologistas em muitas regiões. Um
sistema automático de apoio, que sinalize exames com maior chance de alteração e mostre onde está a
alteração suspeita, pode agilizar laudos e ampliar o acesso a um diagnóstico de qualidade.

Há também uma justificativa científica. A reprodução independente de resultados publicados é parte
essencial do método científico, e no aprendizado profundo aplicado à medicina ela é particularmente
necessária: detalhes como a divisão dos dados, o critério de escolha do modelo e a forma de reportar a
incerteza mudam os números de maneira relevante (BALTRUSCHAT et al., 2019). Um trabalho que reproduz o
CheXNet com um protocolo documentado, código aberto e avaliação estatística explícita contribui para
separar o que é resultado robusto do que é artefato de protocolo, e essa linha de pesquisa ainda é
pouco explorada no Brasil [CONFERIR: se possível, apoiar com uma referência brasileira].

## 1.4 Escopo e limitações

O sistema é uma prova de conceito acadêmica. Ele **não é um dispositivo médico**, não foi validado
clinicamente e não deve ser usado para decidir sobre pacientes. O modelo avalia apenas as 14 doenças
rotuladas no ChestX-ray14, em radiografias frontais de adultos; não detecta outras condições (como
tuberculose ou fraturas) e não usa informações clínicas do paciente. Os rótulos de treino vêm de laudos
processados automaticamente e contêm erros, o que limita o desempenho que qualquer modelo treinado
nessa base pode atingir e medir.

## 1.5 Organização do trabalho

O Capítulo 2 apresenta os conceitos necessários (radiografia de tórax, redes neurais convolucionais,
DenseNet, transferência de aprendizado, métricas de avaliação, calibração e Grad-CAM) e os trabalhos
relacionados. O Capítulo 3 descreve os dados, o pré-processamento, a divisão, o modelo, o treino, os
experimentos e os métodos de avaliação. O Capítulo 4 apresenta os resultados, que o Capítulo 5 discute,
com as limitações do trabalho. O Capítulo 6 traz as conclusões e sugestões de trabalhos futuros.
