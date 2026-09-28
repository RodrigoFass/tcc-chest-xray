# 5 DISCUSSÃO

## 5.1 O modelo está na faixa da literatura

O objetivo central do trabalho era mostrar que uma DenseNet-121 pré-treinada no ImageNet e ajustada no
ChestX-ray14 atinge desempenho competitivo com a literatura. O resultado confirma isso: a AUC média do
modelo final, 0,841 (IC95% 0,833–0,848), é igual à do CheXNet (0,8414), e o valor do CheXNet está dentro
do intervalo de confiança deste trabalho em 11 das 14 classes, incluindo a atelectasia e a pneumonia. Em
relação à primeira referência publicada para o conjunto (WANG et al., 2017), o modelo é melhor em todas
as classes, com diferenças de 0,05 a 0,17 na AUC.

A comparação, porém, precisa ser feita com cuidado. Os dois trabalhos usaram conjuntos de teste
diferentes, sorteados de formas diferentes, e Baltruschat et al. (2019) mostraram que só a divisão dos
dados já muda os resultados no ChestX-ray14. O protocolo também difere: este trabalho usou taxa de
aprendizado inicial dez vezes menor ($10^{-4}$ contra $10^{-3}$), lotes de 32 imagens em vez de 16, mais
transformações de aumento de dados e escolha do modelo pela AUC de validação, e não pela perda. Por isso,
a leitura correta não é que um modelo seja melhor que o outro, mas que a abordagem, reproduzida de forma
independente e com um protocolo documentado, chega **à mesma faixa de desempenho** relatada pelo CheXNet.
Esse é um resultado relevante por si só, porque indica que o desempenho do CheXNet não dependia de
detalhes específicos da divisão ou do treino dos autores.

As três classes em que os resultados divergem merecem um comentário. A efusão pleural ficou acima do
CheXNet (0,886 contra 0,864). Uma hipótese, que este trabalho não testou diretamente, é a decisão de não
recortar as bordas da imagem na avaliação, preservando os seios costofrênicos, onde a efusão aparece
primeiro (Seção 3.3); outra é simplesmente a diferença entre os conjuntos de teste. A infiltração e o
enfisema ficaram abaixo. A infiltração é a classe de rótulo mais inespecífico do conjunto e a de menor
AUC em todos os trabalhos; diferenças pequenas de protocolo mudam seu resultado com facilidade.

## 5.2 Pneumonia: boa ordenação, pouca utilidade isolada

A pneumonia teve a menor AUC entre as três doenças estudadas (0,751), como em todos os trabalhos com esse
conjunto. Três fatores ajudam a explicar. Primeiro, o diagnóstico de pneumonia é clínico-radiológico: a
imagem de uma pneumonia pode ser idêntica à de uma atelectasia, de um edema ou de uma hemorragia, e o
laudo que originou o rótulo muitas vezes dependeu de informações que não estão na imagem. Segundo, os
rótulos do ChestX-ray14 foram extraídos automaticamente dos laudos e contêm erros, e Oakden-Rayner
(2020) identificou justamente a pneumonia entre as classes de rótulo menos confiável; se parte dos
rótulos está errada, nenhum modelo consegue uma AUC alta **medida contra esses rótulos**, mesmo que
acerte a doença. Terceiro, a pneumonia é rara (1,2% do teste, 194 casos), o que torna a estimativa
incerta (IC95% de 0,714 a 0,787) e o problema, desbalanceado.

A AUPRC de 0,044 e o VPP de 2,4% no limiar de Youden mostram a consequência prática. Mesmo ordenando os
exames bem melhor que o acaso (AUPRC 3,8 vezes a prevalência), a maioria dos exames marcados pelo modelo
não tem pneumonia. Nesse cenário, o modelo não deveria ser usado para **confirmar** pneumonia; o VPN de
99,5% sugere, no máximo, um papel de triagem, ajudando a identificar exames com baixa chance da doença,
e ainda assim com a ressalva de que ele deixou de detectar 31% dos casos nesse limiar. Esse é o tipo de
limitação que a AUC, sozinha, esconde, e é o motivo de este trabalho reportar a AUPRC ao lado da
prevalência.

## 5.3 A ponderação de classes não ajudou

A intuição de que dar mais peso aos casos positivos melhora as classes raras não se confirmou. O E2 foi
significativamente pior que o E1 na atelectasia e na efusão, sem ganho significativo na pneumonia, e
praticamente empatado na média. Isso é coerente com o que a teoria prevê (Seção 2.6): a AUC depende
apenas da ordem dos escores, e multiplicar o peso dos positivos na perda muda sobretudo a escala dos
escores, empurrando-os para cima. O efeito colateral apareceu na calibração: os escores brutos do E2
ficaram muito acima da frequência real (Brier da pneumonia dez vezes maior que o do E1). O *Platt
scaling* corrigiu a escala dos dois modelos, mas não a ordenação, e o E1 continuou melhor. O resultado é
consistente com a escolha do CheXNet, que também treinou as 14 classes sem ponderação.

Um resultado prático dessa análise é que o E1 já sai do treino quase calibrado, e a recalibração só
ajusta a faixa de escores mais altos. Isso permite que a interface mostre um valor interpretável como
probabilidade [PREENCHER: se a decisão da Seção 4.5 for essa], com a ressalva de que essa probabilidade
vale para a população e a prevalência do NIH.

## 5.4 O modelo pode estar usando a incidência como atalho?

As três doenças estudadas são bem mais frequentes nos exames AP (cerca de metade dos casos) do que no
conjunto como um todo (40%), porque exames AP são, em geral, de pacientes internados e mais graves. Um
modelo pode aprender a reconhecer que o exame é AP (pela posição do paciente, pela ampliação do coração
ou pela presença de cabos e dispositivos) e usar isso como indício da doença, o que produziria uma AUC
alta sem que o modelo reconhecesse a doença em si. Se esse atalho explicasse boa parte do desempenho, a
AUC **dentro** de cada incidência seria bem menor que a geral.

Os resultados indicam que isso não acontece de forma dominante na efusão e na atelectasia: dentro dos
exames PA, a AUC foi até maior que a geral (0,908 e 0,828, contra 0,886 e 0,816). O que se observa é que
o modelo tem **mais dificuldade nos exames AP** (0,844 e 0,789). Isso tem explicação clínica e técnica:
no paciente deitado, o derrame pleural não se acumula nos seios costofrênicos e produz um véu difuso, mais
difícil de ver; e exames portáteis têm pior qualidade de imagem, mais sobreposição de dispositivos e
pacientes com várias doenças ao mesmo tempo. Ou seja, o modelo funciona pior justamente no grupo de
pacientes mais graves, o que é importante para qualquer uso prático.

Na pneumonia, o padrão é diferente: a AUC dentro dos exames PA (0,722) e dentro dos AP (0,746) ficaram
ambas abaixo da geral (0,751). Quando isso acontece, parte da discriminação medida no conjunto todo vem
da própria diferença de prevalência entre os grupos, o que é compatível com algum uso da incidência como
atalho. Os intervalos, porém, são largos (87 e 107 casos), e o dado não permite conclusão. Os mapas de
calor [PREENCHER: se os falsos positivos de pneumonia forem, em sua maioria, exames AP com dispositivos,
isso reforça a hipótese] e a validação externa (Seção 5.6) ajudam a avaliar essa questão.

Nas análises por sexo e idade, a diferença mais clara foi na atelectasia, com AUC menor nas mulheres e
nos pacientes com mais de 60 anos. Este trabalho não investigou a causa; possibilidades incluem
diferenças na prevalência de outras doenças associadas nesses grupos e na qualidade dos rótulos.
Diferenças de desempenho entre grupos de pacientes são uma questão de equidade e devem ser avaliadas
antes de qualquer uso clínico.

## 5.5 O que os mapas de calor mostram

[PREENCHER depois do Grad-CAM. Pontos a discutir:
- se os mapas se concentram nas regiões anatomicamente esperadas (bases e seios costofrênicos na efusão;
  opacidades segmentares ou lineares na atelectasia; o pulmão, e não o mediastino ou as bordas, na
  pneumonia);
- o *pointing game* comparado ao centro da imagem: se a taxa de acerto for parecida com a do centro,
  os mapas não localizam melhor que o acaso, mesmo que pareçam convincentes;
- padrões nos falsos positivos (cabos, eletrodos, drenos, marcações de texto na imagem) que indiquem
  atalhos, na linha de Zech et al. (2018) e Oakden-Rayner (2020);
- a limitação do Grad-CAM: o mapa tem resolução de 7 × 7 posições ampliada para 224 × 224, e por isso
  indica regiões amplas, não lesões pequenas; e um mapa correto não prova que o modelo raciocinou como
  um radiologista.]

## 5.6 Generalização para outro hospital

[PREENCHER depois do CheXpert. Pontos a discutir:
- a variação da AUC do NIH para o CheXpert, por classe, com os intervalos;
- que a diferença mistura dois efeitos: a mudança de hospital e de equipamento (que tende a reduzir a
  AUC; ZECH et al., 2018) e a mudança da qualidade dos rótulos (radiologistas em consenso no CheXpert,
  laudos processados automaticamente no NIH, o que pode aumentar a AUC medida);
- que os limiares e a calibração ajustados no NIH não valem automaticamente no CheXpert, onde a
  prevalência é outra;
- as classes com poucos casos, sobre as quais não se conclui nada.]

## 5.7 Limitações

a) **Rótulos ruidosos.** Os rótulos de treino e de teste do ChestX-ray14 vêm de laudos processados
   automaticamente e contêm erros (OAKDEN-RAYNER, 2020). As métricas medem a concordância com esses
   rótulos, não com a verdade clínica, e o erro dos rótulos limita o desempenho mensurável, sobretudo na
   pneumonia.

b) **Comparação aproximada com a literatura.** A divisão dos dados e os hiperparâmetros diferem dos do
   CheXNet (Seção 5.1). O experimento E5, na divisão oficial, permite comparar com os trabalhos que a
   usam, mas não com o CheXNet.

c) **Um único hospital no treino.** O modelo foi treinado com imagens de uma única instituição, e modelos
   de radiografia de tórax costumam perder desempenho em outras (ZECH et al., 2018). A validação no
   CheXpert avalia isso parcialmente, num conjunto pequeno.

d) **Possíveis atalhos.** A incidência AP, dispositivos visíveis na imagem e marcações de texto podem
   estar correlacionados com as doenças. A análise por incidência e os mapas de calor investigam isso,
   mas não eliminam a possibilidade.

e) **Resolução e informação.** As imagens foram reduzidas para 224 × 224 pixels, o que pode apagar
   achados pequenos (como nódulos), e o modelo não usa informações clínicas nem exames anteriores do
   paciente, que um radiologista usaria.

f) **Escopo das doenças.** O modelo só conhece 14 doenças. Um exame sem nenhum achado acima do limiar
   não é um exame normal: pode ter uma doença que o modelo não avalia ou um caso que ele não detectou.

g) **Orçamento de treino.** Cada configuração foi treinada uma vez, exceto a final (três sementes), e os
   hiperparâmetros não foram otimizados por busca sistemática. O E4 mede o custo de não usar
   transferência de aprendizado com o mesmo número de épocas, não o limite de uma rede treinada do zero.

h) **Uso clínico.** O sistema é uma prova de conceito. Não passou por validação clínica, avaliação de
   impacto no fluxo de trabalho ou processo regulatório, e não deve ser usado para decisões sobre
   pacientes.

## 5.8 Cuidados tomados para a validade dos resultados

Algumas decisões do protocolo foram tomadas para que os números do Capítulo 4 sejam confiáveis, e vale
registrá-las porque são frequentemente omitidas em trabalhos da área:

- a divisão por paciente impede que o modelo reconheça o paciente em vez da doença;
- o conjunto de teste foi usado uma única vez, depois de a configuração final ser escolhida e registrada
  com base apenas na validação; limiares e recalibração também foram ajustados só na validação;
- todos os resultados têm intervalo de confiança, calculado por paciente, e as comparações entre
  experimentos usam bootstrap pareado;
- a AUPRC e a prevalência acompanham a AUC, para não esconder o problema das classes raras;
- a configuração final foi repetida com três sementes, para separar efeito da configuração de variação
  aleatória;
- as imagens da galeria de mapas de calor foram escolhidas por uma regra fixa, e a localização foi
  medida contra marcações de radiologistas e comparada com uma referência trivial;
- o código tem testes automáticos, e cada experimento registra a configuração, as versões das bibliotecas
  e o *commit* usado.
