# 2 FUNDAMENTAÇÃO TEÓRICA

Este capítulo apresenta os conceitos em que o trabalho se apoia: o exame e as doenças estudadas
(Seção 2.1), as redes neurais e as redes convolucionais (Seções 2.2 e 2.3), a arquitetura DenseNet
(Seção 2.4), a transferência de aprendizado (Seção 2.5), a formulação multirrótulo (Seção 2.6), as
métricas e os métodos estatísticos de avaliação (Seções 2.7 a 2.9), a interpretabilidade por mapas de
calor (Seção 2.10) e, por fim, os trabalhos relacionados (Seção 2.11).

## 2.1 A radiografia de tórax e as doenças estudadas

A radiografia de tórax é uma imagem de projeção: um feixe de raios X atravessa o paciente e a
intensidade que chega ao detector depende de quanto cada tecido atenuou a radiação. Estruturas densas,
como ossos e líquido, atenuam mais e aparecem claras; o ar dos pulmões atenua pouco e aparece escuro.
Como todo o volume do tórax é projetado num único plano, estruturas se sobrepõem, e uma mesma aparência
pode ter causas diferentes.

Duas incidências frontais são comuns. Na **posteroanterior (PA)**, o paciente fica em pé, com o tórax
encostado no detector e o feixe entrando pelas costas; é a incidência padrão para pacientes que
conseguem ficar de pé. Na **anteroposterior (AP)**, o feixe entra pela frente, geralmente com o
paciente sentado ou deitado no leito, com um aparelho portátil. Exames AP são típicos de pacientes
internados e mais graves, e a própria técnica muda a imagem (por exemplo, o coração parece maior).
Essa diferença é importante neste trabalho, porque um modelo pode aprender a reconhecer "exame de
paciente acamado" em vez da doença em si (ver Seção 3.8.4 e Capítulo 5).

As três doenças em que o trabalho se concentra são:

- **Pneumonia:** infecção do parênquima pulmonar. Na radiografia, aparece como uma opacidade
  (consolidação ou infiltrado) numa região do pulmão. O diagnóstico, porém, é clínico-radiológico: a
  mesma opacidade pode ser atelectasia, edema ou hemorragia, e a distinção depende de sintomas e exames
  que a imagem não mostra. Por isso, a pneumonia é notoriamente difícil de rotular apenas pela imagem.
- **Atelectasia:** colapso, parcial ou total, de uma parte do pulmão, que perde ar. Aparece como uma
  opacidade, muitas vezes linear ou em faixa, com sinais de perda de volume (deslocamento de fissuras,
  do diafragma ou do mediastino na direção da área colapsada). É comum em pacientes acamados e no
  pós-operatório.
- **Efusão pleural (derrame pleural):** acúmulo de líquido no espaço entre as pleuras. Na radiografia
  em pé, o líquido se acumula nas partes mais baixas e apaga primeiro os seios costofrênicos, os ângulos
  agudos formados entre o diafragma e a parede torácica, nos cantos inferiores da imagem; derrames
  maiores formam uma opacidade com borda superior curva (menisco). No paciente deitado, o líquido se
  espalha pela parte de trás do tórax e produz um véu difuso, mais difícil de ver.

A localização típica da efusão nos cantos inferiores da imagem motivou uma decisão de
pré-processamento: não recortar as bordas da imagem na avaliação (Seção 3.3).

## 2.2 Aprendizado de máquina e redes neurais

No **aprendizado supervisionado**, um modelo aprende uma função que associa entradas (aqui, imagens) a
saídas desejadas (os rótulos), a partir de exemplos rotulados. O modelo tem parâmetros ajustáveis, e o
treino consiste em escolher os parâmetros que minimizam uma **função de perda**, que mede o quanto as
saídas do modelo se afastam dos rótulos nos exemplos de treino (GOODFELLOW; BENGIO; COURVILLE, 2016).

Uma **rede neural artificial** é composta de camadas de unidades (neurônios). Cada unidade calcula uma
soma ponderada das suas entradas, soma um viés e aplica uma função de ativação não linear. A ativação
mais usada em redes profundas é a ReLU (*Rectified Linear Unit*), $\mathrm{ReLU}(z) = \max(0, z)$, que
é simples de calcular e reduz o problema do desaparecimento do gradiente. Empilhando camadas, a rede
consegue representar funções muito complexas.

Os parâmetros são ajustados por **descida do gradiente**: calcula-se o gradiente da perda em relação a
cada parâmetro, pelo algoritmo de retropropagação (*backpropagation*), e dá-se um pequeno passo na
direção oposta. Na prática, o gradiente é estimado em pequenos lotes (*mini-batches*) de exemplos, a
cada passo. O tamanho do passo é a **taxa de aprendizado**. O otimizador **Adam** (KINGMA; BA, 2015)
adapta o passo de cada parâmetro a partir de médias móveis do gradiente e do seu quadrado, e é o
otimizador padrão em boa parte dos trabalhos de visão computacional, incluindo o CheXNet. Uma estratégia
comum é reduzir a taxa de aprendizado quando a métrica de validação para de melhorar (*reduce on
plateau*), o que permite passos grandes no início e ajustes finos no final.

O risco central do treino é o **sobreajuste** (*overfitting*): o modelo decora os exemplos de treino,
incluindo seu ruído, e perde desempenho em dados novos. Para detectar e controlar esse risco, os dados
são divididos em três conjuntos disjuntos. O **treino** ajusta os parâmetros; a **validação** orienta
as escolhas feitas durante o desenvolvimento (quando parar, qual configuração usar); e o **teste** é
usado uma única vez, no final, para estimar o desempenho em dados nunca vistos. Usar o teste para
qualquer escolha contamina a estimativa, que passa a ser otimista. A **parada antecipada** (*early
stopping*) interrompe o treino quando a métrica de validação deixa de melhorar por um número de épocas
(a paciência) e guarda o modelo da melhor época.

## 2.3 Redes neurais convolucionais

Numa imagem, pixels vizinhos são fortemente relacionados, e o mesmo padrão (uma borda, uma textura) pode
aparecer em qualquer posição. As **redes neurais convolucionais** exploram essas propriedades. Em vez de
ligar cada unidade a todos os pixels, uma camada convolucional aplica pequenos filtros (por exemplo,
3 × 3) que deslizam pela imagem, com os mesmos pesos em todas as posições. Cada filtro produz um **mapa
de características**, que indica onde o padrão que ele detecta aparece. Camadas de *pooling* reduzem a
resolução dos mapas, e a **normalização em lote** (*batch normalization*) padroniza as ativações, o que
estabiliza e acelera o treino.

Empilhadas, as camadas convolucionais formam uma hierarquia: as primeiras detectam bordas e texturas; as
intermediárias, combinações delas; e as últimas, padrões de alto nível. No final, uma camada de
*pooling* global resume cada mapa de características num único número, e uma camada totalmente
conectada (linear) combina esses números para produzir a saída de cada classe. Essa estrutura final
(*pooling* global seguido de uma camada linear) é o que permite os mapas de calor da Seção 2.10.

## 2.4 A arquitetura DenseNet

Redes mais profundas representam funções mais complexas, mas são mais difíceis de treinar, porque o
gradiente enfraquece ao atravessar muitas camadas. As **redes densamente conectadas** (DenseNet),
propostas por Huang et al. (2017), atacam esse problema com um padrão de conexão simples: dentro de um
**bloco denso**, cada camada recebe como entrada a concatenação dos mapas de características de **todas**
as camadas anteriores do bloco, e passa os seus próprios mapas para todas as seguintes. Numa camada
$\ell$, a saída é

$$x_\ell = H_\ell([x_0, x_1, \ldots, x_{\ell-1}]),$$

em que $[\cdot]$ é a concatenação e $H_\ell$, na forma original, é a sequência normalização em lote,
ReLU e convolução 3 × 3. Na DenseNet-121, que usa a variante DenseNet-BC, cada $H_\ell$ começa com um
gargalo: normalização em lote, ReLU e uma convolução 1 × 1 que produz 128 mapas ($4k$); depois vêm
normalização em lote, ReLU e a convolução 3 × 3, que produz os $k = 32$ mapas novos.

Essa conectividade tem três efeitos. O gradiente chega mais diretamente às camadas iniciais, o que
facilita o treino de redes profundas; as características de cada camada são reaproveitadas pelas
seguintes, em vez de reaprendidas; e, por isso, cada camada pode ser estreita, produzindo poucos mapas
novos (a taxa de crescimento, 32 na DenseNet-121), o que resulta em redes com relativamente poucos
parâmetros. Entre os blocos densos, **camadas de transição** (convolução 1 × 1 e *pooling* médio)
reduzem a resolução e reduzem o número de mapas pela metade.

A **DenseNet-121** tem quatro blocos densos, com 6, 12, 24 e 16 camadas, e cerca de 8 milhões de
parâmetros (HUANG et al., 2017). Cada camada densa tem duas convoluções; somadas à convolução inicial, às
três camadas de transição e à camada linear final, elas dão as 121 camadas do nome
(1 + 2 × 58 + 3 + 1). Numa entrada de 224 × 224 pixels, o último bloco produz 1.024 mapas de
7 × 7, que passam por uma normalização em lote, uma ReLU e o *pooling* global, resultando num vetor de
1.024 números. Foi a arquitetura usada pelo CheXNet (RAJPURKAR et al., 2017) e é a escolhida neste
trabalho.

## 2.5 Transferência de aprendizado

Treinar uma rede profunda do zero exige muitos dados rotulados. A **transferência de aprendizado**
(PAN; YANG, 2010) reaproveita uma rede já treinada numa tarefa com muitos dados como ponto de partida
para outra tarefa. Em visão computacional, o ponto de partida mais comum é o ImageNet (DENG et al.,
2009), mais precisamente o subconjunto de mil categorias usado na competição ILSVRC, com cerca de 1,3
milhão de fotografias de treino (RUSSAKOVSKY et al., 2015). As camadas iniciais de uma rede treinada
nele aprendem detectores genéricos de bordas, texturas e formas, úteis também em imagens médicas.

No **ajuste fino** (*fine-tuning*), a última camada da rede pré-treinada é substituída por uma nova,
com uma saída por classe da nova tarefa, e todos os pesos continuam sendo ajustados, agora com os dados
da nova tarefa e uma taxa de aprendizado pequena. Mesmo com imagens tão diferentes de fotografias quanto
radiografias em tons de cinza, essa inicialização costuma acelerar a convergência e melhorar o
resultado, sobretudo quando os dados médicos são poucos (LITJENS et al., 2017). O experimento E4 deste
trabalho mede esse ganho no ChestX-ray14 (Seção 3.7).

## 2.6 Classificação multirrótulo e desbalanceamento

Uma radiografia pode ter várias doenças ao mesmo tempo: no ChestX-ray14, 18,5% das imagens têm duas ou
mais (Seção 3.2). O problema é, portanto, **multirrótulo**: para cada uma das $C$ classes, decide-se
independentemente se ela está presente. A rede produz $C$ números reais (os *logits* $z_c$), e cada um
passa por uma função **sigmoide**, $\sigma(z) = 1/(1 + e^{-z})$, que o leva ao intervalo $(0, 1)$.
Diferentemente da função *softmax* usada em problemas de classe única, as sigmoides não competem entre
si: a presença de uma doença não reduz o valor das outras.

A perda correspondente é a **entropia cruzada binária** (*Binary Cross-Entropy*, BCE), somada ou
promediada sobre as classes:

$$\mathcal{L} = -\frac{1}{C} \sum_{c=1}^{C} \left[ w_c\, y_c \log \sigma(z_c) + (1 - y_c) \log\big(1 - \sigma(z_c)\big) \right],$$

em que $y_c \in \{0, 1\}$ é o rótulo. Com $w_c = 1$, é a BCE padrão. Quando uma classe é rara, os
exemplos negativos dominam a perda, e uma estratégia comum é aumentar o peso dos positivos, com
$w_c$ igual à razão entre negativos e positivos da classe no treino (o `pos_weight` do PyTorch). O
efeito dessa ponderação é mais sutil do que parece: como a AUC depende apenas da ordem dos escores
(Seção 2.7), a ponderação costuma mudar pouco a AUC; o que ela muda é a escala dos escores, que ficam
mais altos e deixam de corresponder a probabilidades (Seção 2.9). O experimento E2 testa essa
estratégia.

O **aumento de dados** (*data augmentation*) é outra forma de lidar com poucos dados e reduzir o
sobreajuste: a cada época, cada imagem de treino é apresentada com pequenas transformações aleatórias
(recorte, espelhamento, rotação, brilho), de modo que a rede nunca veja exatamente a mesma imagem duas
vezes e aprenda características menos dependentes desses detalhes. O experimento E3 mede seu efeito.

## 2.7 Métricas de avaliação

### 2.7.1 Matriz de confusão e métricas num limiar

Para transformar o escore contínuo de uma classe numa decisão (presente ou ausente), escolhe-se um
**limiar**: escores iguais ou acima dele contam como positivos. Comparando as decisões com os rótulos,
obtêm-se os verdadeiros positivos (VP), falsos positivos (FP), falsos negativos (FN) e verdadeiros
negativos (VN), e deles as métricas usuais:

- **sensibilidade** (revocação, *recall*) = VP / (VP + FN): fração dos doentes que o modelo detecta;
- **especificidade** = VN / (VN + FP): fração dos não doentes corretamente descartados;
- **valor preditivo positivo** (VPP, precisão) = VP / (VP + FP): fração dos alarmes que são doença;
- **valor preditivo negativo** (VPN) = VN / (VN + FN);
- **acurácia** = (VP + VN) / total;
- **F1** = média harmônica de precisão e sensibilidade.

A acurácia é enganosa em classes raras. Com 1,2% de pneumonia no teste, um modelo que nunca diz
"pneumonia" acerta 98,8% das imagens. O VPP também depende fortemente da prevalência: mesmo com boa
sensibilidade e especificidade, se a doença é rara, a maioria dos alarmes será falsa.

Neste trabalho, o limiar de cada classe é escolhido na validação pelo **índice de Youden**
(YOUDEN, 1950), $J = \text{sensibilidade} + \text{especificidade} - 1$, que é máximo no ponto da curva
ROC mais distante da diagonal. O limiar é então aplicado, sem alteração, ao teste.

### 2.7.2 Curva ROC e AUC

A **curva ROC** (*Receiver Operating Characteristic*) mostra a sensibilidade em função da taxa de falsos
positivos (1 − especificidade) para todos os limiares possíveis. A **área sob a curva ROC** (AUC) resume
a curva num número entre 0 e 1: 0,5 corresponde a um classificador aleatório e 1 a um classificador
perfeito. A AUC tem uma interpretação direta: é a probabilidade de que, sorteando um exame com a doença e
um sem, o modelo dê escore maior ao que tem a doença (HANLEY; MCNEIL, 1982). Ela é equivalente à
estatística U de Mann-Whitney normalizada e, por isso, depende apenas da **ordem** dos escores, não dos
seus valores. É a métrica principal da literatura do ChestX-ray14, o que permite comparar resultados.

### 2.7.3 Curva precisão-revocação e AUPRC

A AUC não é afetada pela prevalência, o que é uma vantagem para comparar conjuntos, mas pode esconder o
problema das classes raras: com milhares de negativos, uma pequena taxa de falsos positivos corresponde
a muitos falsos alarmes. A **curva precisão-revocação** mostra o VPP em função da sensibilidade, e a
área sob ela (AUPRC, calculada como a precisão média, *average precision*) reflete diretamente essa
dificuldade (SAITO; REHMSMEIER, 2015). Diferentemente da AUC, a AUPRC de um classificador aleatório é
igual à prevalência da classe, e por isso ela deve sempre ser lida ao lado da prevalência.

## 2.8 Incerteza das estimativas

Toda métrica calculada num conjunto de teste finito é uma estimativa, e reportá-la sem intervalo de
confiança impede saber se uma diferença é real. O **bootstrap** (EFRON; TIBSHIRANI, 1993) estima essa
incerteza reamostrando o conjunto de teste com reposição muitas vezes, recalculando a métrica em cada
amostra e tomando os percentis 2,5% e 97,5% da distribuição obtida como intervalo de confiança de 95%.

A unidade de reamostragem precisa respeitar a estrutura dos dados. Imagens do mesmo paciente são
correlacionadas (a mesma anatomia, muitas vezes a mesma doença em exames seguidos); reamostrar imagens
como se fossem independentes subestima a variabilidade e produz intervalos estreitos demais. Por isso,
este trabalho reamostra **pacientes**: cada amostra sorteia pacientes com reposição e inclui todas as
imagens de cada paciente sorteado.

Para comparar dois modelos avaliados no mesmo conjunto, usa-se o **bootstrap pareado**: em cada amostra,
os dois modelos são avaliados nas mesmas imagens, e calcula-se a diferença das métricas. O intervalo de
confiança da diferença leva em conta que os dois modelos erram, em parte, nos mesmos exames. Se o
intervalo contém zero, a diferença não é estatisticamente significativa ao nível de 5%.

Para proporções calculadas com poucos casos, como a taxa de acerto dos mapas de calor (Seção 2.10), o
intervalo de **Wilson** (WILSON, 1927) é preferível ao intervalo normal, porque continua dentro de
[0, 1] e tem cobertura adequada mesmo com dezenas de casos.

## 2.9 Calibração

Um modelo é **calibrado** quando seus escores podem ser lidos como probabilidades: entre os exames que
recebem escore próximo de 0,3, cerca de 30% têm a doença. Discriminação e calibração são propriedades
diferentes. Um modelo pode ordenar muito bem os exames (AUC alta) e, ainda assim, dar escores
sistematicamente altos ou baixos demais; é o que acontece quando a perda é ponderada (Seção 2.6).

A calibração é avaliada pelo **diagrama de confiabilidade**, que agrupa os exames por faixas de escore e
compara o escore médio de cada faixa com a fração observada de positivos (pontos sobre a diagonal
indicam boa calibração), e pelo **escore de Brier** (BRIER, 1950), a média do erro quadrático entre o
escore e o rótulo (0 ou 1), que combina calibração e discriminação (quanto menor, melhor).

Um modelo descalibrado pode ser corrigido depois do treino. No ***Platt scaling*** (PLATT, 1999),
ajusta-se, para cada classe, uma regressão logística de uma variável sobre o *logit* do modelo,
$p = \sigma(a z + b)$, usando um conjunto que não foi usado no treino (aqui, a validação). Como a
transformação é crescente quando $a > 0$, ela não muda a ordem dos escores, e portanto não muda a AUC;
muda apenas a escala, para que os valores possam ser interpretados como probabilidades. Uma limitação
importante é que a probabilidade calibrada vale para a prevalência da população em que foi ajustada; num
hospital com outra prevalência, ela deixa de valer.

## 2.10 Interpretabilidade: CAM e Grad-CAM

Redes profundas são frequentemente chamadas de "caixas-pretas". Em medicina, isso é um problema
concreto: além da resposta, o profissional precisa saber em que ela se baseou. Os **mapas de ativação
de classe** (*Class Activation Maps*, CAM), propostos por Zhou et al. (2016), aproveitam a estrutura
final das redes convolucionais com *pooling* global. Sejam $A^k$ os mapas de características que entram
no *pooling* global (na DenseNet-121, a saída do último bloco denso depois da normalização em lote e da
ReLU finais: 1.024 mapas de 7 × 7 numa entrada de 224 × 224), $Z$ o número de posições de cada mapa
(7 × 7 = 49) e $w^c_k$ e $b_c$ os pesos e o viés da camada linear para a classe $c$. Como o *logit* da
classe é $z_c = \sum_k w^c_k \cdot \frac{1}{Z}\sum_{i,j} A^k_{ij} + b_c$, o mapa

$$M^c = \sum_k w^c_k A^k$$

mostra quanto cada posição da imagem contribui para $z_c$ (o viés é uma constante e não depende da
posição). Ampliado para o tamanho da imagem e
sobreposto a ela, torna-se um mapa de calor. O CheXNet usou exatamente esse método para localizar as
doenças.

O **Grad-CAM** (SELVARAJU et al., 2017) generaliza o CAM para qualquer arquitetura. O peso de cada mapa é
a média espacial do gradiente do *logit* em relação a ele, e aplica-se uma ReLU para manter apenas as
regiões que aumentam o escore:

$$\alpha^c_k = \frac{1}{Z} \sum_{i,j} \frac{\partial z_c}{\partial A^k_{ij}}, \qquad
L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\Big(\sum_k \alpha^c_k A^k\Big).$$

Quando os mapas $A^k$ são os da última camada antes do *pooling* global seguido de uma camada linear,
$\partial z_c / \partial A^k_{ij} = w^c_k / Z$ em todas as posições, e portanto $\alpha^c_k = w^c_k / Z$:
o Grad-CAM é igual ao CAM (com a ReLU) a menos de um fator positivo, que desaparece quando o mapa é
normalizado para o intervalo [0, 1]. Essa equivalência liga o método usado neste trabalho ao do CheXNet
e foi verificada numericamente no código (Seção 3.9).

Um mapa de calor convincente não garante que o modelo "olhou para o lugar certo". Para avaliar a
localização de forma objetiva, usa-se o ***pointing game*** (ZHANG et al., 2018): conta-se como acerto
quando o ponto de máximo do mapa cai dentro da região marcada por um especialista. A taxa de acertos
deve ser comparada com uma referência trivial, como a de sempre apontar o centro da imagem, para saber
se o resultado é melhor do que o acaso.

## 2.11 Trabalhos relacionados

**Wang et al. (2017)** construíram o ChestX-ray8, depois ampliado para 14 doenças (ChestX-ray14), com
112.120 radiografias frontais de 30.805 pacientes do NIH Clinical Center. Os rótulos foram extraídos dos
laudos por processamento de linguagem natural, com uma precisão estimada pelos autores em torno de 90%
[CONFERIR: valor exato no artigo]. Os autores também publicaram cerca de mil caixas delimitadoras
marcadas por radiologistas e um primeiro conjunto de resultados de referência, com redes pré-treinadas
no ImageNet. A versão publicada no CVPR avalia apenas 8 doenças; os resultados para as 14 classes vêm da
versão revisada do artigo no arXiv (arXiv:1705.02315) e são os reproduzidos na Tabela 2 do CheXNet, com
AUC média de 0,738, calculada neste trabalho a partir dos valores por classe [CONFERIR: versão do arXiv
e tabela de onde saem os valores; ao citar as duas versões, a ABNT pede WANG et al., 2017a e 2017b].

**Rajpurkar et al. (2017)**, com o **CheXNet**, treinaram uma DenseNet-121 de 121 camadas, pré-treinada
no ImageNet, no ChestX-ray14. Para pneumonia, compararam o modelo com quatro radiologistas num conjunto
de 420 imagens rotuladas por eles, e relataram F1 de 0,435 para o modelo contra 0,387 para a média dos
radiologistas [CONFERIR: valores]. Estendido às 14 classes, o CheXNet obteve AUC média de 0,841, com
0,8094 para atelectasia, 0,8638 para efusão e 0,7680 para pneumonia. Os autores usaram uma divisão
aleatória própria (70/10/20, sem pacientes em comum), BCE sem ponderação, Adam com taxa inicial de
0,001, lotes de 16 imagens, apenas espelhamento horizontal como aumento de dados e escolha do modelo pela
menor perda de validação.

**Irvin et al. (2019)** publicaram o **CheXpert**, com 224.316 radiografias de 65.240 pacientes do
Stanford Hospital, rotuladas para 14 observações por um rotulador automático que também identifica
**incerteza** nos laudos ("não se pode excluir pneumonia"). O artigo compara políticas para os rótulos
incertos, entre elas ignorá-los, tratá-los como negativos ou como positivos, e mostra que a melhor
escolha varia conforme a observação. O conjunto de validação do CheXpert, com 200 estudos de 200
pacientes, foi anotado de forma independente por três radiologistas; as anotações foram binarizadas e o
rótulo de referência de cada observação é o voto da maioria, de modo que os rótulos finais não têm a
categoria "incerto" [CONFERIR no artigo]; por isso, ele é usado aqui como conjunto de teste externo.

**Baltruschat et al. (2019)** compararam sistematicamente abordagens para o ChestX-ray14 (arquiteturas,
transferência de aprendizado, resolução de entrada, uso de dados não visuais) e mostraram que os
resultados variam de forma relevante conforme a divisão dos dados, o que torna aproximada a comparação
entre trabalhos que usam divisões diferentes. **Guendel et al. (2018)** propuseram redes densas com
informação de localização e avaliaram no ChestX-ray14 com a divisão oficial.

Dois trabalhos questionam o que esses números significam. **Zech et al. (2018)** mostraram que um modelo
de detecção de pneumonia treinado com dados de alguns hospitais perdeu desempenho em outro hospital e que
as redes conseguiam identificar de qual hospital e de qual equipamento vinha a imagem, um atalho que pode
inflar resultados internos. **Oakden-Rayner (2020)** revisou visualmente amostras do ChestX-ray14 e
encontrou erros frequentes nos rótulos, sobretudo em algumas classes, e imagens em que a doença rotulada
já estava sendo tratada (por exemplo, pneumotórax com dreno), o que permite ao modelo acertar pelo motivo
errado. **Litjens et al. (2017)** revisaram o aprendizado profundo em imagens médicas de forma ampla e
destacaram a falta de dados rotulados e a necessidade de interpretabilidade como desafios centrais.

O Quadro 1 resume os trabalhos mais próximos.

**Quadro 1 – Trabalhos relacionados**

| Trabalho | Dados | Modelo | Contribuição para este trabalho |
|---|---|---|---|
| Wang et al. (2017) | ChestX-ray14 | CNNs pré-treinadas | Dataset, caixas delimitadoras, primeira referência de AUC |
| Rajpurkar et al. (2017) | ChestX-ray14 | DenseNet-121 | Arquitetura e metodologia de referência; CAM |
| Irvin et al. (2019) | CheXpert | DenseNet-121 | Conjunto externo; rótulos de radiologistas (maioria de três) na validação |
| Baltruschat et al. (2019) | ChestX-ray14 | ResNet-50 e variações | Efeito da divisão dos dados nos resultados |
| Zech et al. (2018) | 3 hospitais | DenseNet-121 | Generalização entre hospitais; atalhos |
| Oakden-Rayner (2020) | ChestX-ray14 | — | Qualidade dos rótulos |
| Selvaraju et al. (2017) | — | Grad-CAM | Método de interpretabilidade |

Fonte: elaborado pelo autor.

[CONFERIR: arquitetura usada por Baltruschat et al. e por Zech et al.; ajustar o quadro se necessário.]
