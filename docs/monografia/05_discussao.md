# 5 DISCUSSÃO

## 5.1 O modelo está na faixa da literatura

A hipótese do trabalho (Capítulo 1) era que uma DenseNet-121 pré-treinada no ImageNet e ajustada no
ChestX-ray14 atinge desempenho competitivo com a literatura. O resultado sustenta essa hipótese: a AUC média do
modelo final, 0,841 (IC95% 0,833–0,848), é igual à do CheXNet (0,8414), e o valor do CheXNet está dentro
do intervalo de confiança deste trabalho em 11 das 14 classes, incluindo a atelectasia e a pneumonia. Em
relação à primeira referência publicada para o conjunto (WANG et al., 2017b), o modelo é melhor em todas
as classes, com diferenças de 0,05 a 0,17 na AUC.

A comparação, porém, precisa ser feita com cuidado. Os dois trabalhos usaram conjuntos de teste
diferentes, sorteados de formas diferentes, e Baltruschat et al. (2019) mostraram que só a divisão dos
dados já muda os resultados no ChestX-ray14. O protocolo também difere: este trabalho usou taxa de
aprendizado inicial dez vezes menor ($10^{-4}$ contra $10^{-3}$), lotes de 32 imagens em vez de 16, mais
transformações de aumento de dados e escolha do modelo pela AUC de validação, e não pela perda. Por isso,
a leitura correta não é que um modelo seja melhor que o outro, mas que a abordagem, reproduzida de forma
independente e com um protocolo documentado, chega **à mesma faixa de desempenho** relatada pelo CheXNet.
Esse é um resultado relevante por si só, porque sugere que o desempenho do CheXNet não dependia apenas
de detalhes específicos da divisão ou do treino dos autores, ainda que uma única reprodução não baste
para afirmar isso.

As três classes em que os resultados divergem merecem um comentário. A efusão pleural ficou acima do
CheXNet (0,886 contra 0,864). Este trabalho não testou a causa. A explicação mais simples é a diferença
entre os conjuntos de teste: o valor do CheXNet vem de outra amostra, e só a divisão dos dados já muda os
resultados (BALTRUSCHAT et al., 2019). As diferenças de treino listadas acima, como o aumento de dados
mais amplo, também podem contribuir. A decisão de não recortar as bordas da imagem (Seção 3.3) preserva os
seios costofrênicos, onde a efusão aparece primeiro, em relação ao recorte central de 256 para 224 pixels
usado em reimplementações comuns; ela não distingue este trabalho do artigo original, porque o CheXNet
também reduziu a imagem inteira para 224 × 224, sem recorte. A infiltração e o
enfisema ficaram abaixo. A infiltração é a classe de rótulo mais inespecífico do conjunto e a de menor
AUC nos trabalhos citados; diferenças pequenas de protocolo mudam seu resultado com facilidade.

## 5.2 Pneumonia: ordenação moderada, pouca utilidade isolada

A pneumonia teve a menor AUC entre as três doenças estudadas (0,751), como nos trabalhos citados
(WANG et al., 2017b; RAJPURKAR et al., 2017). Três fatores ajudam a explicar. Primeiro, o diagnóstico de pneumonia é clínico-radiológico: a
imagem de uma pneumonia pode ser idêntica à de uma atelectasia, de um edema ou de uma hemorragia, e o
laudo que originou o rótulo muitas vezes dependeu de informações que não estão na imagem. Segundo, os
rótulos do ChestX-ray14 foram extraídos automaticamente dos laudos e contêm erros. Na revisão visual de
Oakden-Rayner (2020), só 60% das imagens rotuladas como pneumonia mostravam o achado, um dos valores
mais baixos entre as classes. Se parte dos
rótulos está errada, nenhum modelo consegue uma AUC alta **medida contra esses rótulos**, mesmo que
acerte a doença. Terceiro, a pneumonia tem poucos exemplos positivos: só 985 imagens de treino têm o
rótulo, contra 8.158 de atelectasia e 9.175 de efusão. Para um padrão já ambíguo e com rótulos ruidosos,
isso pode somar-se aos dois primeiros fatores. A raridade sozinha, porém, não explica a AUC menor: a
hérnia (172 imagens de treino) chegou a 0,939 e o edema (1.632) a 0,896, e a ponderação de classes do E2
não mudou a pneumonia de forma significativa (Seção 5.3). A raridade afeta sobretudo a precisão da
estimativa: com 194 casos no teste, o IC95% vai de 0,714 a 0,787, mas mesmo o limite superior fica
abaixo da AUC da atelectasia (0,816) e da efusão (0,886).

A AUPRC de 0,044 e o VPP de 2,4% no limiar de Youden mostram a consequência prática. Mesmo ordenando os
exames melhor que o acaso (AUPRC 3,9 vezes a prevalência), a grande maioria dos exames marcados pelo
modelo não tem pneumonia. Nesse cenário, o modelo não deveria ser usado para **confirmar** pneumonia, e
também não serve bem para **descartá-la**: o VPN de 99,5% reflete sobretudo a prevalência (sem modelo
algum, 98,8% dos exames não têm a doença), e ele deixou de detectar 31% dos casos nesse limiar (razão de
verossimilhança negativa de cerca de 0,46). Um papel de triagem exigiria um limiar com sensibilidade bem
maior, ao custo de ainda mais alarmes falsos. Esse é o tipo de
limitação que a AUC, sozinha, esconde, e é o motivo de este trabalho reportar a AUPRC ao lado da
prevalência.

## 5.3 Ponderação de classes, aumento de dados e transferência de aprendizado

A intuição de que dar mais peso aos casos positivos melhora as classes raras não se confirmou. O E2 foi
significativamente pior que o E1 na atelectasia e na efusão, sem ganho significativo na pneumonia, e
praticamente empatado na média. Isso é coerente com o que a teoria prevê (Seção 2.6): a AUC depende
apenas da ordem dos escores, e multiplicar o peso dos positivos na perda muda sobretudo a escala dos
escores, empurrando-os para cima. O efeito colateral apareceu na calibração: os escores brutos do E2
ficaram muito acima da frequência real (Brier da pneumonia dez vezes maior que o do E1). O *Platt
scaling* corrigiu a escala dos dois modelos; como é uma transformação crescente, ele não muda a
ordenação, e o E1 continuou melhor na atelectasia e na efusão. O resultado é
consistente com a escolha do CheXNet, que também treinou as 14 classes sem ponderação.

Um resultado prático dessa análise é que o E1 já sai do treino quase calibrado, e a recalibração só
ajusta a faixa de escores mais altos. Isso permitiu que a interface mostre o valor como "probabilidade
estimada", com a ressalva de que essa probabilidade vale para a população e a prevalência do NIH.

Os outros dois experimentos confirmaram o que se esperava. Sem aumento de dados (E3), o modelo
sobreajustou logo depois da terceira época e perdeu 0,010 de AUC média; sem transferência de aprendizado
(E4), perdeu 0,020, mesmo usando todo o orçamento de 30 épocas e mais que o dobro do tempo de treino. As
duas técnicas contribuem para o resultado, e a transferência é a que mais pesa. O E4 mostra, porém, que o
ChestX-ray14 é grande o bastante para uma rede treinada do zero chegar a 0,821, bem acima da primeira
referência publicada (0,738). Parte da vantagem da transferência está em acelerar a convergência, e um
treino mais longo, com outro cronograma de taxa de aprendizado, poderia reduzir a diferença (Seção 5.7,
item g).

A repetição com três sementes dá a escala dessas diferenças. Só a troca da semente mudou a AUC média em
até 0,005 e a da pneumonia em até 0,015, uma variação que os intervalos por bootstrap não capturam,
porque consideram a amostra de teste, mas não o treino (Seção 4.6.3). Diante disso, as conclusões mais
seguras são as das diferenças grandes e consistentes: a perda do E4 na média, na atelectasia e na efusão
e a do E2 na atelectasia. As diferenças na pneumonia, positivas nos três experimentos, estão dentro da
variação entre sementes e não permitem dizer que alguma configuração favoreça essa classe. Uma
comparação mais rigorosa treinaria cada configuração com várias sementes, o que não coube no orçamento
deste trabalho.

## 5.4 O modelo pode estar usando a incidência como atalho?

Os exames AP são 40% do conjunto, mas concentram cerca de metade dos casos das três doenças estudadas
(50% a 56%, Tabela 1). Assim, a prevalência dessas doenças nos exames AP é 1,5 a 1,9 vez a dos exames PA
(15,0% contra 9,8% na efusão, 13,0% contra 8,5% na atelectasia e 1,8% contra 0,9% na pneumonia), porque
exames AP são, em geral, de pacientes internados e mais graves. Um
modelo pode aprender a reconhecer que o exame é AP (pela posição do paciente, pela ampliação do coração
ou pela presença de cabos e dispositivos) e usar isso como indício da doença, o que produziria uma AUC
alta sem que o modelo reconhecesse a doença em si. Se esse atalho explicasse boa parte do desempenho, a
AUC **dentro** de cada incidência seria bem menor que a geral.

Os resultados indicam que isso não acontece de forma dominante na efusão e na atelectasia: dentro dos
exames PA, a AUC foi até maior que a geral (0,908 e 0,828, contra 0,886 e 0,816). O que se observa é que
o modelo tem **mais dificuldade nos exames AP** (0,844 e 0,789). Isso tem explicação clínica e técnica:
no paciente deitado, o derrame pleural não se acumula nos seios costofrênicos e produz um véu difuso, mais
difícil de ver; e exames portáteis têm pior qualidade de imagem, mais sobreposição de dispositivos e
pacientes com várias doenças ao mesmo tempo. Ou seja, na efusão e na atelectasia o modelo funciona pior
justamente nos exames AP, que costumam ser de pacientes mais graves, o que é importante para qualquer uso
prático.

Na pneumonia, o padrão é diferente: a AUC dentro dos exames PA (0,722) e dentro dos AP (0,746) ficaram
ambas abaixo da geral (0,751). Quando isso acontece, parte da discriminação medida no conjunto todo vem
da própria diferença de prevalência entre os grupos, o que é compatível com algum uso da incidência como
atalho. Os intervalos, porém, são largos (87 e 107 casos), e o dado não permite conclusão.

Um indício mais forte aparece nos erros, e não só na pneumonia. No limiar da validação, o modelo marcou
como pneumonia 46% dos exames AP sem pneumonia, contra 23% dos PA. Mesmo entre os exames sem nenhuma das
14 doenças, marcou 31% dos AP e 16% dos PA; o mesmo acontece na efusão (24% contra 9% nesses exames) e
na atelectasia (30% contra 15%). Ou seja, com o mesmo limiar, um exame AP recebe escores mais altos que
um PA mesmo sem doença rotulada, e a discriminação **dentro** de cada incidência, medida pela AUC, não
mostra isso. O dado é compatível com o uso da incidência como atalho, mas também com rótulos incompletos
nos exames AP, de pacientes mais graves, em que achados podem não ter sido registrados como diagnóstico
no laudo. Os mapas de calor não resolvem a questão: nos falsos positivos de pneumonia da galeria, dois
dos três exames são AP portáteis, mas o mapa fica sobre os pulmões, e não sobre a marcação de texto ou
os dispositivos (Seção 4.8), o que não exclui o atalho, porque a incidência muda o aspecto de todo o
tórax. A validação externa (Seção 5.6) ajuda a avaliar essa questão, e, na prática, um uso real
precisaria de limiares diferentes para cada incidência ou de um modelo treinado para não depender dela.

Nas análises por sexo e idade, a diferença mais clara foi na atelectasia, com AUC menor nas mulheres e
nos pacientes com mais de 60 anos. Este trabalho não investigou a causa; possibilidades incluem
diferenças na prevalência de outras doenças associadas nesses grupos e na qualidade dos rótulos.
Diferenças de desempenho entre grupos de pacientes são uma questão de equidade e devem ser avaliadas
antes de qualquer uso clínico.

## 5.5 O que os mapas de calor mostram

Os mapas de calor apontam, em geral, para regiões plausíveis: nos verdadeiros positivos de efusão, o
hemitórax opacificado e as bases; nos de pneumonia, as opacidades pulmonares (Seção 4.8). O *pointing
game*, porém, mostra os limites dessa impressão visual. O pico do mapa caiu dentro da caixa do
radiologista em cerca de um quarto das imagens das três doenças estudadas (15 de 62). Isso é melhor que
o centro da imagem na atelectasia e na efusão, mas não na pneumonia, em que o acerto (3 de 20) não se
distingue da referência trivial. Na cardiomegalia, o centro da imagem acerta todas as imagens, o que
mostra como uma localização "correta" pode vir só da posição típica do achado.

Dois fatores limitam a leitura dos mapas. Primeiro, o mapa tem resolução de 7 × 7 posições, ampliada
para 224 × 224, e por isso indica regiões amplas, não lesões pequenas; o *pointing game* exige que um
único ponto caia numa caixa às vezes pequena, o que é mais exigente do que a impressão visual. Segundo,
um mapa no lugar certo não prova que o modelo raciocinou como um radiologista: ele mostra onde estão as
características que mais pesaram, não por que pesaram. Nos falsos positivos da galeria, os mapas ficaram
sobre os pulmões, e não sobre marcações de texto, eletrodos ou cateteres, sem sinal claro de atalhos
desse tipo (ZECH et al., 2018; OAKDEN-RAYNER, 2020). A galeria, no entanto, tem só três exemplos por
grupo, e a análise por incidência (Seção 5.4) sugere uma influência difusa da incidência nos escores,
que um mapa de calor não mostraria. Nos falsos negativos, com escores próximos de zero, o mapa não tem
significado, e isso deveria ser dito a quem usa a interface. Por fim, a escolha da camada-alvo pouco
importou: as duas candidatas geraram mapas quase iguais e taxas de acerto sem diferença significativa.

## 5.6 Generalização para outro hospital

No CheXpert, de outro hospital, o modelo manteve a atelectasia e ficou próximo na efusão pleural e no
edema. Na média das seis classes, a AUC foi 0,041 menor que no teste do NIH, uma queda dentro da
incerteza de um conjunto de 202 imagens. Não apareceu, portanto, uma perda grande e generalizada de
desempenho ao mudar de hospital, como a relatada por Zech et al. (2018) para a pneumonia. Essa comparação,
porém, mistura dois efeitos de sinais opostos.

O primeiro efeito é a **mudança de hospital e de equipamento**, que tende a reduzir a AUC. O segundo é a
**mudança da qualidade dos rótulos**: no CheXpert, voto da maioria de três radiologistas; no NIH,
rótulos processados automaticamente a partir dos laudos. Rótulos melhores podem aumentar a AUC medida,
porque o modelo deixa de ser "punido" por acertar casos rotulados errado.

A consolidação é compatível com o segundo efeito. No NIH, ela é uma das classes de aspecto mais
inespecífico, com sobreposição à infiltração e à pneumonia, e teve AUC de só 0,783. No CheXpert,
com rótulos de radiologistas, a AUC foi de 0,911. A melhora sugere que parte do desempenho baixo medido
no NIH vinha dos rótulos, e não do modelo. A cardiomegalia foi no sentido oposto: caiu de 0,920 para
0,836. Uma explicação possível, não testada aqui, é a diferença no critério de "coração aumentado"
entre os laudos do NIH e os radiologistas do CheXpert.

Dois cuidados limitam essas conclusões. Primeiro, os limiares e a calibração ajustados no NIH não valem
automaticamente no CheXpert, onde a prevalência é bem maior (32% de efusão nas imagens frontais, contra
12,6% no teste do NIH): a "probabilidade estimada" da interface vale para a população do NIH, como ela
mesma avisa. Segundo, o pneumotórax, com 7 casos, e a pneumonia, sem rótulos de radiologistas neste
conjunto, ficam sem avaliação externa. Para a pneumonia, isso quer dizer que o desempenho fora do NIH
continua desconhecido.

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
   CheXpert avalia isso parcialmente, num conjunto pequeno e sem a pneumonia.

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
- o conjunto de teste só foi usado depois de a configuração final ser escolhida e registrada com base
  apenas na validação, e todos os experimentos foram avaliados nele com o mesmo protocolo; a escolha do
  modelo, os limiares e a recalibração saíram só da validação, e o teste serviu apenas para decisões de
  apresentação que não alteram as métricas de classificação (a camada do Grad-CAM, escolhida pelo
  *pointing game*, e o nome do escore na interface);
- as AUCs e AUPRCs têm intervalo de confiança de 95% calculado por bootstrap por paciente, e as
  comparações entre experimentos usam bootstrap pareado (as métricas no limiar e o escore de Brier são
  apresentados como estimativas pontuais);
- a AUPRC e a prevalência acompanham a AUC, para não esconder o problema das classes raras;
- a configuração final foi repetida com três sementes, para separar efeito da configuração de variação
  aleatória;
- as imagens da galeria de mapas de calor foram escolhidas por uma regra fixa, e a localização foi
  medida contra marcações de radiologistas e comparada com uma referência trivial;
- o desenho da validação externa (fonte dos rótulos, classes e configuração principal) foi registrado
  antes de o modelo ser aplicado ao CheXpert;
- o código tem testes automáticos, e cada experimento registra a configuração, as versões das principais
  bibliotecas e o *commit* usado.
