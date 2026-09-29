# RESUMO

A radiografia de tórax é o exame de imagem mais realizado no mundo, mas sua interpretação exige
especialização e está sujeita à variabilidade entre profissionais. Este trabalho desenvolveu um sistema
de apoio à identificação de doenças pulmonares em radiografias de tórax, com foco em pneumonia,
atelectasia e efusão pleural. Uma rede neural convolucional DenseNet-121, pré-treinada no ImageNet, foi
ajustada nas 14 doenças do conjunto público NIH ChestX-ray14 (112.120 imagens de 30.805 pacientes), com
divisão dos dados por paciente e escolha do modelo apenas na validação. No conjunto de teste (16.822
imagens), o modelo obteve área sob a curva ROC (AUC) média de 0,841 (IC95% 0,833–0,848), na mesma faixa
do CheXNet (0,841), com AUC de 0,886 para efusão pleural, 0,816 para atelectasia e 0,751 para pneumonia.
A área sob a curva precisão-revocação da pneumonia foi de 0,044 (3,9 vezes a prevalência de 1,15%) e, no
limiar escolhido na validação, só 2,4% dos exames apontados pelo modelo tinham o rótulo de pneumonia: o
modelo ordena os exames melhor que o acaso, mas a grande maioria dos que ele aponta não tem a doença. A
ponderação de classes na função de perda não melhorou a AUC e piorou a calibração; o modelo sem
ponderação saiu do treino bem calibrado. Sem aumento de dados ou sem transferência de aprendizado, a AUC
média caiu 0,010 e 0,020, respectivamente. Nos exames em incidência anteroposterior (AP), típicos de
pacientes mais graves, a AUC foi menor que nos posteroanteriores na efusão pleural (0,844 contra 0,908)
e na atelectasia (0,789 contra 0,828); na pneumonia, não houve diferença clara entre as incidências.
Os mapas de calor Grad-CAM apontaram para regiões plausíveis, mas o ponto de máximo do mapa caiu dentro
da marcação do radiologista em só 24% das imagens das três doenças. [PREENCHER: uma frase sobre o
CheXpert.] O modelo
foi disponibilizado numa interface web de demonstração, que mostra o valor de cada doença, o limiar de
decisão e um mapa de calor Grad-CAM. [CONFERIR: manter esta frase só depois de publicar a interface.]

**Palavras-chave:** aprendizado profundo; radiografia de tórax; redes neurais convolucionais; DenseNet;
Grad-CAM.

[CONFERIR: limite de palavras do resumo no modelo da UVV; a NBR 6028 recomenda de 150 a 500 palavras
para trabalhos acadêmicos.]

# ABSTRACT

Chest radiography is the most frequently performed imaging exam in the world, but its interpretation
requires expertise and varies between readers. This work developed a system to support the
identification of lung diseases on chest X-rays, focusing on pneumonia, atelectasis and pleural effusion.
A DenseNet-121 convolutional neural network, pretrained on ImageNet, was fine-tuned on the 14 diseases of
the public NIH ChestX-ray14 dataset (112,120 images from 30,805 patients), with a patient-level data
split and model selection on the validation set only. On the test set (16,822 images), the model reached
a mean area under the ROC curve (AUC) of 0.841 (95% CI 0.833–0.848), in the same range as CheXNet
(0.841), with AUCs of 0.886 for pleural effusion, 0.816 for atelectasis and 0.751 for pneumonia. The
area under the precision-recall curve for pneumonia was 0.044 (3.9 times the 1.15% prevalence), and at
the threshold chosen on the validation set only 2.4% of the exams flagged by the model carried the
pneumonia label: the model ranks exams better than chance, but the vast majority of those it flags do not
have the disease. Class weighting in the loss function did not improve AUC and worsened calibration; the
unweighted model was well calibrated after training. Without data augmentation or without transfer
learning, mean AUC dropped by 0.010 and 0.020, respectively. On anteroposterior (AP) exams, typical of more
severely ill patients, AUC was lower than on posteroanterior exams for pleural effusion (0.844 vs. 0.908)
and atelectasis (0.789 vs. 0.828); for pneumonia, there was no clear difference between views. Grad-CAM heatmaps
pointed to plausible regions, but the heatmap peak fell inside the radiologist's box in only 24% of the
images of the three diseases. [TO FILL: one sentence on CheXpert.] The model was made available in a
web demo that shows each disease's score, the decision threshold and a Grad-CAM heatmap. [TO CHECK: keep
this sentence only after the demo is published.]

**Keywords:** deep learning; chest radiography; convolutional neural networks; DenseNet; Grad-CAM.
