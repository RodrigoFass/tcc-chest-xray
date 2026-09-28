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
   (2017) em todas as classes, com a ressalva de que a comparação é aproximada, por causa das diferenças
   de protocolo;

d) a ponderação de classes não melhorou a AUC e piorou a calibração; [PREENCHER: efeito do aumento de
   dados (E3) e da transferência de aprendizado (E4), e a variação entre sementes];

e) o modelo final é bem calibrado já sem ajuste, e o *Platt scaling* corrige a faixa de escores mais
   altos; o desempenho é pior nos exames AP, típicos de pacientes mais graves, o que é a principal
   diferença entre subgrupos encontrada;

f) [PREENCHER: resultado do Grad-CAM e do *pointing game*];

g) [PREENCHER: resultado no CheXpert];

h) a interface foi publicada [PREENCHER: link] e reproduz os escores da avaliação.

A hipótese do trabalho, de que uma DenseNet-121 com transferência de aprendizado atinge no ChestX-ray14
desempenho competitivo com a literatura, foi confirmada. Mais importante que o número final, porém, é o
que a avaliação detalhada revelou: a AUC, sozinha, esconde que a pneumonia é detectada com valor
preditivo positivo muito baixo; o desempenho cai no grupo de pacientes mais graves; e os valores do
modelo só podem ser lidos como probabilidades depois de verificada a calibração, e apenas para a
população em que ela foi verificada. Esses são os pontos que separam um bom resultado em um conjunto
público de uma ferramenta que possa, um dia, ser útil na prática.

## 6.1 Trabalhos futuros

- **Rótulos melhores:** reavaliar o conjunto de teste com rótulos revistos por radiologistas, ou treinar
  com conjuntos de rótulos mais confiáveis, para medir o desempenho real e não a concordância com o
  rotulador automático.
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
