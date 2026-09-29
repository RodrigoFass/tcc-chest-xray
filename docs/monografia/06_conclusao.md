# 6 CONCLUSÃO

Este trabalho desenvolveu um sistema de apoio à identificação de doenças pulmonares em radiografias de
tórax, baseado numa DenseNet-121 pré-treinada no ImageNet e ajustada no NIH ChestX-ray14, com foco em
pneumonia, atelectasia e efusão pleural. O sistema recebe uma radiografia e devolve, para cada uma das
14 doenças do conjunto, um valor recalibrado com o limiar de decisão escolhido na validação, uma frase de
resumo e um mapa de calor Grad-CAM, numa interface web de demonstração.

Os objetivos específicos foram atingidos da seguinte forma:

a) o modelo foi treinado com divisão dos dados por paciente, e a configuração final foi escolhida apenas
   com a validação;

b) no teste, a AUC média nas 14 classes foi de 0,841 (IC95% 0,833–0,848); nas doenças estudadas, 0,886
   para efusão pleural, 0,816 para atelectasia e 0,751 para pneumonia, com AUPRC de 0,543, 0,361 e
   0,044, respectivamente;

c) os resultados estão na mesma faixa dos do CheXNet (AUC média de 0,8414) e acima dos de Wang et al.
   (2017b) em todas as classes, com a ressalva de que a comparação é aproximada, por causa das diferenças
   de protocolo; na divisão oficial do NIH (E5), a AUC média foi de 0,811 (0,804–0,816), num conjunto de
   teste mais difícil, que concentra pacientes com muitos exames de acompanhamento;

d) a ponderação de classes não melhorou a AUC e piorou a calibração; sem aumento de dados, a AUC média
   caiu 0,010 e o sobreajuste começou logo depois da terceira época; sem transferência de aprendizado, caiu 0,020,
   com mais que o dobro do tempo de treino; e a repetição da configuração final com três sementes deu
   desvio-padrão de 0,003 na AUC média e de 0,008 na pneumonia, o que torna frágeis as diferenças
   pequenas entre experimentos, em especial na pneumonia;

e) o modelo final é bem calibrado já sem ajuste, e o *Platt scaling* corrige a faixa de escores mais
   altos; na efusão pleural e na atelectasia, o desempenho é pior nos exames AP, típicos de pacientes
   mais graves (AUC de 0,844 contra 0,908 nos PA e de 0,789 contra 0,828), o que é a principal diferença
   entre subgrupos encontrada; na pneumonia, as estimativas das duas incidências (PA 0,722; AP 0,746) têm
   intervalos que se sobrepõem e ficaram ambas abaixo da AUC geral;

f) os mapas de calor Grad-CAM apontam para regiões plausíveis nos acertos, mas o pico do mapa caiu
   dentro da caixa do radiologista em só 15 de 62 imagens das três doenças (24%), acima do centro da
   imagem na atelectasia e na efusão, mas não na pneumonia;

g) no CheXpert, de outro hospital e com rótulos de radiologistas, a AUC média das seis classes em comum
   foi de 0,823 (IC95% 0,777–0,876), contra 0,865 no teste do NIH; a atelectasia (0,810) e a efusão
   pleural (0,849) ficaram na mesma faixa, a cardiomegalia caiu e a consolidação subiu, e a pneumonia não
   pôde ser avaliada, por falta de rótulos de radiologistas nesse conjunto;

h) a interface foi publicada como uma página web que roda no navegador (<https://huggingface.co/spaces/rotriguin/tcc-raio-x>) e reproduz os
   escores e os mapas de calor da avaliação.

A hipótese do trabalho, apresentada no Capítulo 1, de que uma DenseNet-121 com transferência de
aprendizado atinge no ChestX-ray14 desempenho competitivo com a literatura, foi sustentada pelos
resultados, no sentido definido ali: o modelo ficou na mesma faixa do CheXNet, numa comparação que é
aproximada porque os conjuntos de teste são diferentes. Mais importante que o número final, porém, é o
que a avaliação detalhada revelou: a AUC, sozinha, esconde que a pneumonia é detectada com valor
preditivo positivo muito baixo; na efusão e na atelectasia, o desempenho cai nos exames AP, que costumam
ser de pacientes internados e mais graves, e os exames AP recebem escores mais altos mesmo sem doença
rotulada; os mapas de calor indicam regiões amplas, sem localizar os achados com precisão; e os valores do
modelo só podem ser lidos como probabilidades depois de verificada a calibração, e apenas para a
população em que ela foi verificada. Esses são os pontos que separam um bom resultado em um conjunto
público de uma ferramenta que possa, um dia, ser útil na prática.

## 6.1 Trabalhos futuros

- **Rótulos melhores:** reavaliar o conjunto de teste com rótulos revistos por radiologistas, ou treinar
  com conjuntos de rótulos mais confiáveis, para medir o desempenho real e não a concordância com o
  rotulador automático. Para a pneumonia, que ficou sem avaliação externa, um caminho direto são as
  anotações de radiologistas feitas para o desafio de pneumonia da RSNA sobre imagens do próprio
  ChestX-ray14 (SHIH et al., 2019).
- **Mais dados e mais hospitais:** treinar com o CheXpert e outros conjuntos públicos (como o MIMIC-CXR),
  com as estratégias de tratamento de rótulos incertos propostas por Irvin et al. (2019), e validar em
  dados de hospitais brasileiros.
- **Resolução maior:** avaliar entradas de 512 × 512 pixels ou mais, que podem ajudar em achados pequenos.
- **Atalhos:** investigar de forma direta o uso da incidência e de dispositivos pelo modelo, por exemplo
  treinando só com exames PA ou mascarando regiões fora dos pulmões.
- **Arquiteturas mais recentes:** comparar com redes mais recentes e com modelos pré-treinados em grandes
  volumes de radiografias, em vez do ImageNet.
- **Avaliação com usuários:** medir se o sistema, com os mapas de calor, ajuda de fato radiologistas ou
  médicos generalistas, num estudo com leitores.
