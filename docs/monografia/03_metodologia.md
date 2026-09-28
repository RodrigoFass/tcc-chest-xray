# 3 MATERIAIS E MÉTODOS

Este capítulo descreve como o trabalho foi feito, com detalhe suficiente para ser reproduzido. Todo o
código, as configurações de cada experimento, as divisões dos dados e as predições estão num
repositório público [CONFERIR: link do GitHub, ou "disponível mediante solicitação"], e cada número
apresentado no Capítulo 4 pode ser recalculado a partir dele.

## 3.1 Visão geral

A Figura 1 resume o fluxo do trabalho. As radiografias do NIH ChestX-ray14 são pré-processadas uma
única vez e divididas por paciente em treino, validação e teste. Uma DenseNet-121 pré-treinada no
ImageNet é ajustada no treino; a validação define quando parar, qual configuração usar, o limiar de
decisão de cada doença e a recalibração dos escores. O teste é usado apenas no final, para as métricas
reportadas. Sobre o modelo final, geram-se os mapas de calor Grad-CAM, a avaliação externa no CheXpert e
a interface de demonstração.

**Figura 1 – Fluxo do trabalho**

```
NIH ChestX-ray14 (112.120 imagens)
  └─ pré-processamento: tons de cinza, 256 × 256
      └─ divisão por paciente: treino 70% | validação 15% | teste 15%
          ├─ treino da DenseNet-121 (experimentos E1–E5)
          │    └─ validação: parada antecipada, escolha do modelo, limiares, Platt
          └─ teste (uma vez, com o modelo escolhido)
               ├─ AUC, AUPRC, limiares, calibração, subgrupos (IC95% por bootstrap)
               ├─ Grad-CAM, caixas dos radiologistas, pointing game
               ├─ validação externa: CheXpert
               └─ interface web de demonstração
```

Fonte: elaborado pelo autor. [Substituir por um diagrama desenhado na versão final.]

## 3.2 Conjunto de dados

Este trabalho utiliza o conjunto de dados público NIH ChestX-ray14, disponibilizado pelo NIH Clinical
Center (National Institutes of Health, Estados Unidos) e descrito por Wang et al. (2017). O conjunto
reúne 112.120 radiografias de tórax em incidência frontal (PA ou AP), de 30.805 pacientes, em arquivos
PNG de 1024 × 1024 pixels. Cada imagem traz até 14 rótulos de doenças torácicas, ou a indicação de
ausência de achados ("No Finding"), extraídos automaticamente dos laudos radiológicos por técnicas de
processamento de linguagem natural, além da idade e do sexo do paciente e da incidência do exame. Um
subconjunto de 880 imagens tem 984 caixas delimitadoras, marcadas por radiologistas, para oito das
doenças.

Os dados foram obtidos da cópia disponibilizada na plataforma Kaggle (conjunto `nih-chest-xrays/data`,
licença CC0 1.0), que contém as mesmas 112.120 imagens e os arquivos de metadados da distribuição do NIH
(`Data_Entry_2017.csv`, `BBox_List_2017.csv`, `train_val_list.txt` e `test_list.txt`). A leitura dos
rótulos foi conferida contra as contagens de referência do conjunto (por exemplo, 11.559 imagens com
atelectasia, 13.317 com efusão e 1.431 com pneumonia).

Das 112.120 imagens, 53,8% não têm achados e 18,5% têm duas ou mais doenças. A prevalência das três
doenças estudadas é de 11,9% para efusão pleural (13.317 imagens), 10,3% para atelectasia (11.559) e
1,3% para pneumonia (1.431). A Tabela 1 mostra todas as classes. A idade mediana é de 49 anos (intervalo
interquartil de 35 a 59 anos); 43,5% das imagens são de pacientes do sexo feminino, e 40,0% dos exames
foram feitos na incidência AP. O número de imagens por paciente varia de 1 a 184 (mediana de 1; 56,8%
dos pacientes têm uma única imagem). O arquivo de metadados contém 16 idades impossíveis (acima de 100
anos), mantidas nos dados e excluídas apenas do gráfico de idade.

**Tabela 1 – Imagens por classe no NIH ChestX-ray14 e fração de exames AP em cada classe**

| Classe | Imagens | Prevalência | Exames AP |
|---|---|---|---|
| Infiltração | 19.894 | 17,7% | 53,0% |
| **Efusão pleural** | **13.317** | **11,9%** | **50,5%** |
| **Atelectasia** | **11.559** | **10,3%** | **50,4%** |
| Nódulo | 6.331 | 5,6% | 34,0% |
| Massa | 5.782 | 5,2% | 38,3% |
| Pneumotórax | 5.302 | 4,7% | 35,7% |
| Consolidação | 4.667 | 4,2% | 67,4% |
| Espessamento pleural | 3.385 | 3,0% | 28,6% |
| Cardiomegalia | 2.776 | 2,5% | 43,7% |
| Enfisema | 2.516 | 2,2% | 40,4% |
| Edema | 2.303 | 2,1% | 88,0% |
| Fibrose | 1.686 | 1,5% | 16,5% |
| **Pneumonia** | **1.431** | **1,3%** | **56,0%** |
| Hérnia | 227 | 0,2% | 15,4% |
| Sem achados | 60.361 | 53,8% | 34,9% |
| Todas as imagens | 112.120 | — | 40,0% |

Fonte: elaborado pelo autor a partir de Wang et al. (2017) (`results/tables/eda_contagens_por_classe.csv`).
Em negrito, as três doenças estudadas.

As figuras da análise exploratória (`results/figures/eda_*`) mostram a prevalência das classes, a
coocorrência entre doenças, a distribuição de idade e sexo, o número de imagens por paciente, a fração
de exames AP e exemplos de imagens [inserir as figuras escolhidas].

Três imagens não contêm anatomia visível, apenas um fundo uniforme: uma delas rotulada como atelectasia
(00007160_002.png) e duas como sem achados (00010007_121.png e 00012249_001.png). Elas foram mantidas,
para não alterar o conjunto em relação aos trabalhos da literatura; com 3 em 112.120 imagens (nenhuma no
conjunto de teste), o efeito é desprezível, e elas ilustram o ruído de rotulagem discutido por
Oakden-Rayner (2020).

## 3.3 Pré-processamento

As imagens foram convertidas uma única vez para tons de cinza e reduzidas de 1024 × 1024 para 256 × 256
pixels, com filtro de Lanczos. Das 112.120 imagens, 519 estavam no formato RGBA, mas com os três canais
de cor iguais e o canal alfa constante, de modo que a conversão para tons de cinza não altera o
conteúdo. As imagens reduzidas foram empacotadas num único arquivo, o que torna a cópia entre máquinas
rápida e dispensa o conjunto original de cerca de 42 GB durante o treino.

Na entrada da rede, a imagem inteira é redimensionada para 224 × 224 pixels, sem recorte central (que
cortaria cerca de 6% de cada borda, onde ficam os seios costofrênicos e aparece a efusão pleural),
replicada nos três canais esperados pela DenseNet-121 pré-treinada no ImageNet e normalizada com a média
e o desvio-padrão do ImageNet.

Durante o treino, e só nele, aplica-se aumento de dados com as seguintes transformações aleatórias, em
sequência: recorte aleatório com redimensionamento para 224 × 224, mantendo de 85% a 100% da área e
proporção entre 0,95 e 1,05; espelhamento horizontal com probabilidade de 50%; rotação de até ±10°; e
variação de brilho de até ±20%. As transformações são pequenas de propósito: a anatomia do tórax tem
posição e orientação bastante previsíveis, e deformações grandes criariam imagens irreais. O
espelhamento horizontal troca os lados do tórax (o coração passa para a direita); foi mantido por ser a
única transformação usada pelo CheXNet e porque as doenças estudadas ocorrem nos dois lados.

## 3.4 Divisão em treino, validação e teste

A divisão em treino, validação e teste (70%, 15% e 15% das imagens) foi feita por paciente, para que
imagens de um mesmo paciente nunca apareçam em mais de um conjunto. Uma divisão por imagem colocaria
exames do mesmo paciente no treino e no teste, e o modelo poderia reconhecer o paciente em vez da
doença, inflando o desempenho medido. Os 30.805 pacientes foram embaralhados com semente fixa (42) e
alocados inteiros, em sequência, até atingir 70% e 85% das imagens. Alocar pacientes até atingir a
proporção de **imagens**, e não sortear 70% dos pacientes, foi necessário porque alguns pacientes têm
mais de cem imagens, o que desviaria as proporções.

O resultado tem 78.486 imagens de treino (21.419 pacientes), 16.812 de validação (4.596 pacientes) e
16.822 de teste (4.790 pacientes). Um teste automático confirma que nenhum paciente aparece em dois
conjuntos. A prevalência das três doenças estudadas ficou a menos de 20% (em termos relativos) da
prevalência geral em todos os conjuntos, sem necessidade de estratificação; a maior diferença foi a da
pneumonia na validação (1,50% contra 1,28%). A Tabela 2 mostra os números. Os subgrupos usados na
análise por subgrupo também ficaram equilibrados: de 41,5% a 44,3% de mulheres, cerca de 40% de exames
AP e idade mediana de 48 a 49 anos nos três conjuntos.

**Tabela 2 – Divisão dos dados por paciente**

| | Treino | Validação | Teste | Total |
|---|---|---|---|---|
| Imagens | 78.486 (70,0%) | 16.812 (15,0%) | 16.822 (15,0%) | 112.120 |
| Pacientes | 21.419 | 4.596 | 4.790 | 30.805 |
| Pneumonia | 985 (1,26%) | 252 (1,50%) | 194 (1,15%) | 1.431 (1,28%) |
| Atelectasia | 8.158 (10,39%) | 1.730 (10,29%) | 1.671 (9,93%) | 11.559 (10,31%) |
| Efusão pleural | 9.175 (11,69%) | 2.019 (12,01%) | 2.123 (12,62%) | 13.317 (11,88%) |

Fonte: elaborado pelo autor (`results/tables/prevalencia_splits.csv`).

Além dessa divisão, que é a principal, o trabalho usa a **divisão oficial** do NIH (`test_list.txt` e
`train_val_list.txt`) num experimento separado (E5, Seção 3.7), para comparação direta com trabalhos que
a adotam. Nela, a validação foi tirada de dentro do conjunto oficial de treino e validação, por paciente,
na proporção 70:15, resultando em 71.255 imagens de treino, 15.269 de validação e 25.596 de teste, sem
pacientes em comum. O teste oficial é mais "doente" que o restante do conjunto (efusão pleural em 18,2%
das imagens, contra 11,9% no total; atelectasia em 12,8%, contra 10,3%; pneumonia em 2,2%, contra 1,3%),
o que afeta sobretudo a AUPRC, que depende da prevalência.

## 3.5 Modelo

O modelo é uma DenseNet-121 (HUANG et al., 2017) com os pesos pré-treinados no ImageNet distribuídos
pela biblioteca torchvision. A camada de classificação original, de 1.000 classes, foi substituída por
uma camada linear de 1.024 entradas e 14 saídas, uma por doença, com vieses iniciados em zero. A rede
produz *logits*; a função sigmoide é aplicada apenas na inferência, porque a função de perda usada no
treino já a incorpora de forma numericamente estável.

Uma alteração técnica foi feita na implementação padrão: a ReLU que o torchvision aplica depois do
último bloco denso é executada "no lugar" (*in place*), sobrescrevendo o tensor de entrada, o que
interfere nos mecanismos usados para calcular o Grad-CAM. No modelo deste trabalho, essa ReLU é uma
camada separada e sem sobrescrita, o que não muda nenhum resultado numérico e a torna a camada-alvo
natural do Grad-CAM (Seção 3.9). O modelo final tem, portanto, a sequência: blocos densos da DenseNet-121,
ReLU, *pooling* médio global e camada linear.

## 3.6 Treino

O treino usou a entropia cruzada binária (BCE) média sobre as 14 classes, o otimizador Adam com taxa de
aprendizado inicial de $10^{-4}$ e demais parâmetros padrão, e lotes de 32 imagens. A taxa de
aprendizado é dividida por 10 sempre que a AUC média de validação passa duas épocas seguidas sem melhorar
(*ReduceLROnPlateau* com paciência 1: uma época sem melhora é tolerada, e a segunda provoca a redução).
O treino para quando a AUC média de validação não melhora por 5 épocas (parada antecipada), com limite
de 30 épocas, e o modelo guardado é o da época com a maior AUC média de validação.

A escolha da AUC média de validação como critério, em vez da perda de validação usada pelo CheXNet,
alinha o critério de parada com a métrica que o trabalho reporta. A perda de validação também não é
comparável entre configurações com e sem ponderação de classes.

Para reprodutibilidade, a semente 42 foi fixada no Python, no NumPy e no PyTorch, e a biblioteca cuDNN
foi configurada para usar algoritmos determinísticos. Mesmo assim, treinos na GPU podem variar
ligeiramente entre execuções; por isso, a configuração final foi repetida com outras duas sementes
(43 e 44), mantendo a mesma divisão dos dados. O treino usa precisão mista (*float16*) na GPU, o que
acelera o treino sem efeito prático na métrica; as métricas finais do Capítulo 4, porém, são calculadas
a partir de predições em precisão completa (*float32*). A cada época, o estado completo do treino é
salvo, de modo que um treino interrompido continua de onde parou e produz o mesmo resultado de um treino
sem interrupção (o que foi verificado por um teste automático em CPU).

## 3.7 Experimentos

Cada experimento é definido por um arquivo de configuração que herda da configuração base e altera
apenas o que o distingue, como mostra o Quadro 2.

**Quadro 2 – Experimentos**

| ID | Experimento | Diferença em relação à base | Pergunta |
|---|---|---|---|
| E1 | Base | — (BCE sem ponderação) | Desempenho da abordagem proposta |
| E2 | Ponderação de classes | `pos_weight` = negativos/positivos de cada classe | A ponderação melhora as classes raras? |
| E3 | Sem aumento de dados | configuração vencedora, sem aumento de dados | Quanto o aumento de dados contribui? |
| E1 (seeds 43 e 44) | Repetições | configuração vencedora, outra semente de treino | Quanto o resultado varia só pela aleatoriedade do treino? |
| E4 | Sem transferência de aprendizado | configuração vencedora, pesos aleatórios | Quanto o pré-treino no ImageNet contribui? |
| E5 | Divisão oficial | configuração vencedora, divisão oficial do NIH | Resultado comparável a trabalhos que usam a divisão oficial |

Fonte: elaborado pelo autor.

A **escolha da configuração final foi feita apenas na validação**, antes de qualquer predição no teste,
e registrada no log de decisões do projeto. O critério foi a AUC média das 14 classes na validação, em
precisão completa. O E1 obteve 0,838 (IC95% 0,829–0,845) e o E2, 0,832 (0,824–0,840); a diferença
pareada E1 − E2 foi de +0,005 (−0,001 a 0,010), não significativa na média, mas favorável ao E1 em
atelectasia (+0,012; 0,006 a 0,018) e efusão (+0,004; 0,001 a 0,007), sem diferença significativa em
pneumonia. Com a média empatada no limite, o E1 foi escolhido também por ser mais simples e ter treinado
mais rápido. Os experimentos E3, E4, E5 e as repetições partem, portanto, do E1.

O E4 usa o mesmo limite de épocas e a mesma parada antecipada. Como redes treinadas do zero convergem
mais devagar, o resultado mede o custo de não usar transferência de aprendizado **com o mesmo
orçamento de treino**, e não o limite da arquitetura. O E5 tem outro conjunto de teste e é reportado
numa tabela separada, nunca lado a lado com E1–E4.

## 3.8 Avaliação

As predições (*logits* e escores) de cada modelo são geradas uma única vez para a validação e para o
teste e salvas em arquivos CSV; toda a avaliação é calculada a partir delas.

### 3.8.1 Métricas e intervalos de confiança

Para cada classe, calculam-se a AUC e a AUPRC (precisão média), sempre com a prevalência ao lado, e a
média da AUC nas 14 classes, que é a métrica resumo usada pela literatura. Todos os valores têm
intervalo de confiança de 95% por **bootstrap por paciente**, com 1.000 reamostragens e semente fixa
(Seção 2.8). As diferenças entre experimentos são avaliadas por **bootstrap pareado**, com as mesmas
reamostras para os dois modelos; uma diferença é considerada significativa quando o intervalo de 95% não
contém zero. Para as repetições com sementes diferentes, reportam-se a média e o desvio-padrão amostral.

### 3.8.2 Limiares de decisão

O limiar de cada classe é o que maximiza o índice de Youden na **validação** (Seção 2.7.1). No teste,
com esse limiar fixo, calculam-se acurácia, sensibilidade, especificidade, VPP, VPN e F1, e as matrizes
de confusão das três doenças estudadas.

### 3.8.3 Calibração

A calibração das três doenças estudadas é avaliada no teste por diagramas de confiabilidade, com dez
faixas de mesmo número de imagens, e pelo escore de Brier. Os escores são recalibrados por *Platt
scaling* (Seção 2.9), com os parâmetros de cada classe ajustados **apenas na validação** e aplicados ao
teste, onde se comparam a calibração e o Brier antes e depois. Não se usou a regressão isotônica, uma
alternativa mais flexível, porque com poucas dezenas de casos positivos nas classes raras ela sobreajusta.
O objetivo da recalibração é permitir que a interface mostre o valor como uma probabilidade estimada, se
a calibração no teste justificar.

### 3.8.4 Análise por subgrupos

Para as três doenças estudadas, a AUC é calculada, com IC95%, separadamente por sexo, por faixa etária
(menos de 40, de 40 a 60 e mais de 60 anos) e por incidência (PA e AP). A análise verifica se o modelo
funciona de forma parecida para grupos diferentes de pacientes e expõe um possível atalho: como exames
AP costumam ser de pacientes acamados e mais graves, e como as três doenças são mais frequentes nesses
exames (Tabela 1), o modelo pode ter aprendido a reconhecer a incidência em vez da doença. Nesse caso,
o desempenho **dentro** de cada incidência seria bem menor que o geral.

### 3.8.5 Comparação com a literatura

A AUC de cada classe é comparada com as reportadas por Wang et al. (2017) e pelo CheXNet (RAJPURKAR et
al., 2017), tomadas da Tabela 2 do artigo do CheXNet (versão 3 no arXiv), que reporta as duas. A
comparação é aproximada, porque os trabalhos usam divisões diferentes dos dados, e o desempenho no
ChestX-ray14 varia bastante com a divisão (BALTRUSCHAT et al., 2019).

## 3.9 Interpretabilidade

Os mapas de calor são gerados com o Grad-CAM (SELVARAJU et al., 2017), pela biblioteca
`pytorch-grad-cam`. A camada-alvo padrão é a ReLU após o último bloco denso (Seção 3.5). Como ela é
seguida apenas do *pooling* global e da camada linear, o Grad-CAM nela é igual ao CAM usado pelo CheXNet
(Seção 2.10); essa igualdade foi verificada numericamente num teste automático, que compara o mapa do
Grad-CAM com o mapa $\sum_k w^c_k A^k$ calculado diretamente dos pesos. Como alternativa, avalia-se também
a saída do último bloco denso antes da normalização final, e a escolha entre as duas camadas é feita
pelo *pointing game* e pela inspeção visual. Os escores exibidos junto aos mapas vêm da mesma função de
inferência usada na avaliação.

Três análises são feitas no conjunto de teste:

a) **Galeria de acertos e erros:** para cada doença estudada, exemplos de verdadeiros positivos, falsos
   positivos e falsos negativos, no limiar da validação. Para que as figuras não pareçam escolhidas a
   dedo, a seleção segue uma regra fixa, definida antes de gerar as figuras: os três verdadeiros
   positivos e os três falsos positivos de **maior** escore e os três falsos negativos de **menor**
   escore, com no máximo uma imagem por paciente e desempate pelo nome do arquivo.

b) **Comparação com os radiologistas:** para todas as imagens de teste com caixa delimitadora no
   `BBox_List_2017.csv` (coordenadas na escala de 1024 pixels, convertidas para 224; no arquivo, a
   infiltração aparece como "Infiltrate"), o mapa de calor é mostrado com a caixa do radiologista e o
   ponto de máximo do mapa. Das 984 caixas, 153 estão em imagens do conjunto de teste, sendo 22 de
   atelectasia, 20 de efusão e 20 de pneumonia.

c) ***Pointing game*:** fração das imagens em que o ponto de máximo do mapa cai dentro da caixa, por
   doença e por camada-alvo, com intervalo de Wilson de 95%. Uma imagem com mais de uma caixa da mesma
   doença conta uma vez, como acerto se o ponto cair em qualquer delas. Como referência, calcula-se a
   mesma taxa para o centro da imagem.

## 3.10 Validação externa

Para avaliar a generalização para outra instituição, o modelo final, **sem nenhum novo treino**, é
aplicado ao conjunto de validação do CheXpert (IRVIN et al., 2019), do Stanford Hospital, cujos rótulos
foram definidos por consenso de radiologistas e não têm a categoria "incerto". Usam-se apenas as imagens
frontais e as sete classes presentes nos dois conjuntos: atelectasia, cardiomegalia, efusão pleural
("Pleural Effusion" no CheXpert), pneumonia, pneumotórax, consolidação e edema. As imagens do CheXpert
não são quadradas; na configuração principal, elas são redimensionadas para quadrado, como qualquer
imagem enviada à interface, e, como verificação, também com preenchimento das bordas em preto,
preservando as proporções. A AUC de cada classe, com IC95% por bootstrap por paciente, é comparada com
a do teste do NIH. O conjunto é pequeno e tem poucos casos de algumas doenças; classes com menos de 30
casos positivos são reportadas, mas não sustentam conclusões.

## 3.11 Sistema de demonstração

O sistema final é uma interface web feita com a biblioteca Gradio, publicada no Hugging Face Spaces
[PREENCHER: link do Space], que roda em CPU. O usuário envia uma radiografia (PNG ou JPG) e recebe:

a) uma frase de resumo, com as doenças cujo valor ficou acima do limiar ("Achados acima do limiar:
   Efusão pleural, Atelectasia") ou, se nenhuma ficou, "Nenhum achado acima do limiar entre as 14 doenças
   avaliadas". A interface nunca diz "normal" ou "saudável", porque o modelo só conhece 14 doenças e todo
   limiar deixa passar casos;
b) uma tabela com o valor das 14 doenças, as três estudadas em destaque, cada uma com o limiar da
   validação e a indicação "acima" ou "abaixo do limiar";
c) o mapa de calor Grad-CAM da doença escolhida;
d) um aviso fixo: "Protótipo acadêmico. Não usar para diagnóstico.", com a observação de que imagens
   muito diferentes das do NIH (foto de tela, criança, incidência lateral) geram resultados sem sentido.

O valor exibido é o escore recalibrado por *Platt scaling*, e o limiar é mostrado na mesma escala. Ele
é chamado de [PREENCHER: "probabilidade estimada" ou "escore do modelo", conforme a decisão da Seção 4.5],
com a observação de que a calibração vale para a população do NIH. O modelo exportado para a interface
reproduz os escores da avaliação com diferença máxima de 3,6 × 10⁻⁷ nas três imagens de teste
conferidas (o critério era 10⁻⁴), e cada análise, incluindo o mapa de calor, levou entre 0,3 e 0,4 s em
CPU no computador de desenvolvimento (Intel Core i5-10400F) [PREENCHER: e cerca de X s no Hugging Face
Spaces].

## 3.12 Ambiente e reprodutibilidade

O treino foi feito num computador pessoal com uma GPU NVIDIA GeForce RTX 2060, em Windows, com Python
3.10.11, PyTorch 2.14.0 (CUDA 12.6) e torchvision 0.29.0. A primeira época do E1 levou 8,8 minutos
(cerca de 168 imagens de treino por segundo, com o cuDNN determinístico), a segunda, 7,5 minutos, e,
a partir da terceira, cerca de 6,3 minutos (225 imagens por segundo); o E1 completo, com 13 épocas, levou 85,5 minutos. As versões de todas as bibliotecas e o *commit* do código ficam
registrados junto com cada experimento.

O código é organizado como um pacote Python, com configurações em YAML, e tem testes automáticos (mais
de 130) que verificam, entre outras coisas, que nenhum paciente aparece em dois conjuntos, que as
métricas coincidem com as da biblioteca scikit-learn, que um treino interrompido e retomado dá o mesmo
resultado e que a interface reproduz os escores avaliados. Os testes rodam a cada alteração, num serviço
de integração contínua.

## 3.13 Aspectos éticos

Os dados são disponibilizados anonimizados: o arquivo de metadados identifica cada paciente apenas por um
número sequencial (Patient ID) e traz, como dados demográficos, somente a idade e o sexo, sem nome, data
de nascimento, datas de exame ou qualquer outro identificador pessoal. Como o trabalho utiliza
exclusivamente essa base pública e anonimizada, sem contato com pacientes e sem acesso a dados que
permitam identificá-los, ele não foi submetido a Comitê de Ética em Pesquisa [CONFERIR com o orientador
se cabe citar a Resolução CNS nº 510/2016, art. 1º, parágrafo único]. Conforme solicitado pelo NIH, o
trabalho cita Wang et al. (2017) e reconhece o NIH Clinical Center como fornecedor dos dados. O uso do
CheXpert segue o acordo de uso de pesquisa da Universidade de Stanford [CONFERIR os termos no momento do
download].
