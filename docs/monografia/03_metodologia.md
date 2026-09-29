# 3 MATERIAIS E MÉTODOS

Este capítulo descreve como o trabalho foi feito, com detalhe suficiente para ser reproduzido. Todo o
código, as configurações de cada experimento, as divisões dos dados e as predições no NIH estão num
repositório público (<https://github.com/RodrigoFass/tcc-chest-xray>), e cada número apresentado no
Capítulo 4 pode ser recalculado a partir dele. A exceção são as predições por imagem no CheXpert, que
trazem os rótulos desse conjunto e por isso não são redistribuídas (Seção 3.13); delas, o repositório
guarda as métricas e tabelas agregadas.

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
          ├─ treino da DenseNet-121 (E1–E4 e sementes; o E5 usa a divisão oficial)
          │    └─ validação: parada antecipada, limiares, Platt e escolha da configuração (E1),
          │       feita antes de qualquer predição de teste
          └─ teste (só depois da escolha; cada modelo avaliado uma vez)
               ├─ todos os experimentos: AUC, AUPRC, limiares, calibração, subgrupos
               │  (IC95% por bootstrap) e diferenças pareadas
               └─ modelo escolhido (E1): Grad-CAM, caixas dos radiologistas, pointing game,
                  validação externa no CheXpert e interface web de demonstração
```

Fonte: elaborado pelo autor. [Substituir por um diagrama desenhado na versão final.]

## 3.2 Conjunto de dados

Este trabalho utiliza o conjunto de dados público NIH ChestX-ray14, disponibilizado pelo NIH Clinical
Center (National Institutes of Health, Estados Unidos) e descrito por Wang et al. (2017a). O conjunto
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
dos pacientes têm uma única imagem). Há 5.241 imagens (4,7%), de 1.600 pacientes, de menores de 18 anos
(idade mínima registrada de 1 ano), mantidas em todos os conjuntos. O arquivo de metadados contém 16
idades impossíveis (acima de 100 anos, provavelmente erros de registro). Essas imagens foram mantidas nos
dados e no treino, mas ficaram de fora do gráfico e das estatísticas de idade e da análise por faixa
etária (Seção 3.8.4).

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

Fonte: elaborado pelo autor a partir de Wang et al. (2017a) (`results/tables/eda_contagens_por_classe.csv`).
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

Uma alteração técnica foi feita na implementação padrão: a ReLU que o torchvision aplica depois da
normalização em lote final, que segue o último bloco denso, é executada "no lugar" (*in place*),
sobrescrevendo o tensor de entrada, o que interfere nos mecanismos usados para calcular o Grad-CAM. No
modelo deste trabalho, essa ReLU é uma camada separada e sem sobrescrita, o que não muda nenhum
resultado numérico e permite usá-la como camada-alvo do Grad-CAM, uma das duas comparadas na Seção 3.9. O modelo final tem,
portanto, a sequência: parte convolucional da DenseNet-121 (convolução inicial, blocos densos e camadas
de transição), normalização em lote final, ReLU, *pooling* médio global e camada linear.

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
acelera o treino sem efeito prático na métrica; as métricas finais do Capítulo 4, exceto as curvas de
treino da Seção 4.1, são calculadas a partir de predições em precisão completa (*float32*). A cada
época, o estado completo do treino (pesos, otimizador, *scheduler*, contadores e geradores aleatórios) é
salvo, de modo que um treino interrompido pode continuar de onde parou. Um teste automático em CPU, sem
precisão mista e com as imagens carregadas no processo principal, confirmou que o treino retomado
reproduz o treino sem interrupção. No treino real, as imagens são carregadas por vários processos
paralelos, e nessa condição a retomada altera a ordem das imagens e os sorteios do aumento de dados: o
resultado seria estatisticamente comparável, mas não idêntico. Pelo registro de sessões de cada
experimento, nenhum dos sete treinos deste trabalho precisou ser retomado.

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
média da AUC nas 14 classes, que é a métrica resumo usada pela literatura. Esses valores têm
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
alternativa mais flexível, por dois motivos: ela tende a sobreajustar quando há poucos casos positivos,
como na hérnia, com apenas 27 na validação; e, por ser uma função em degraus, cria empates entre escores e
pode alterar a AUC. O Platt tem só dois parâmetros e, como a inclinação ajustada foi positiva em todas as
classes, preserva a ordem dos escores e, portanto, a AUC (Seção 2.9).
O objetivo da recalibração é permitir que a interface mostre o valor como uma probabilidade estimada, se
a calibração no teste justificar.

### 3.8.4 Análise por subgrupos

Para as três doenças estudadas, a AUC é calculada, com IC95%, separadamente por sexo, por faixa etária
(menos de 40, de 40 a 60 e mais de 60 anos; as idades acima de 100 anos ficam fora das faixas, e no
teste é uma única imagem, com idade registrada de 155 anos) e por incidência (PA e AP). A análise verifica se o modelo
funciona de forma parecida para grupos diferentes de pacientes e expõe um possível atalho: como exames
AP costumam ser de pacientes acamados e mais graves, e como as três doenças são mais frequentes nesses
exames (Tabela 1), o modelo pode ter aprendido a reconhecer a incidência em vez da doença. Nesse caso,
o desempenho **dentro** de cada incidência seria bem menor que o geral.

### 3.8.5 Comparação com a literatura

A AUC de cada classe é comparada com as reportadas por Wang et al. (2017b) e pelo CheXNet (RAJPURKAR et
al., 2017), tomadas da Tabela 2 do artigo do CheXNet (versão 3 no arXiv), que reporta as duas. A
comparação é aproximada, porque os trabalhos usam divisões diferentes dos dados, e o desempenho no
ChestX-ray14 varia bastante com a divisão (BALTRUSCHAT et al., 2019).

## 3.9 Interpretabilidade

Os mapas de calor são gerados com o Grad-CAM (SELVARAJU et al., 2017), pela biblioteca
`pytorch-grad-cam`, em duas camadas-alvo candidatas. A primeira é a ReLU após a normalização em lote
final (Seção 3.5). Como ela é seguida apenas do *pooling* global e da camada linear, o Grad-CAM nela
coincide com o CAM usado pelo CheXNet depois de aplicada a ReLU, a menos de um fator positivo que
desaparece na normalização (Seção 2.10). Essa equivalência foi verificada numericamente num teste
automático, que compara o mapa do Grad-CAM com $\mathrm{ReLU}\big(\sum_k w^c_k A^k\big)$, calculado
diretamente dos pesos, com ambos normalizados para o intervalo [0, 1]. A segunda é a saída do último
bloco denso, antes da normalização em lote final e da ReLU (Seção 3.5). A camada usada na galeria, nas
figuras com as caixas e na interface é a de maior taxa de acerto no *pointing game* (item c, somando as
três doenças estudadas), com a comparação visual das duas camadas como apoio. Como essa escolha usa as
caixas do conjunto de teste, as duas camadas são reportadas. Os escores exibidos junto aos mapas vêm da
mesma função de inferência usada na avaliação.

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
foram definidos pelo voto da maioria de três radiologistas que anotaram cada estudo de forma
independente; as anotações foram binarizadas antes da votação, e por isso os rótulos finais não têm a
categoria "incerto".

A Stanford não distribui mais o arquivo original com os rótulos dessa validação. Hoje, as imagens estão
no CheXpert Plus (CHAMBON et al., 2024), e os rótulos dos radiologistas estão no CheXlocalize (SAPORTA
et al., 2022). Os rótulos próprios do CheXpert Plus são extraídos automaticamente dos laudos e por isso
não servem como referência. O CheXlocalize traz, para cada imagem da validação, contornos desenhados
pelos radiologistas sobre as observações com rótulo positivo. Segundo a documentação do CheXlocalize,
o arquivo de anotações da validação inclui exatamente as imagens com ao menos um rótulo positivo e, em
cada uma, só as observações positivas: 187 imagens e 643 anotações, os mesmos números do arquivo
baixado. Os rótulos foram reconstruídos a partir dessas anotações: uma observação é positiva numa imagem
se estiver anotada nela, e negativa caso contrário. Entre as 234 imagens da validação, isso dá, por
exemplo, 80 imagens com atelectasia e 67 com efusão pleural.

Usam-se apenas as imagens frontais, 202 imagens de 200 pacientes, e as seis classes que existem nos dois
conjuntos e são anotadas no CheXlocalize: atelectasia, cardiomegalia, efusão pleural ("Pleural
Effusion" no CheXpert), pneumotórax, consolidação e edema. A pneumonia fica de fora, porque o
CheXlocalize não a anota. As imagens do CheXpert Plus são PNG de 8 bits convertidas pela própria
Stanford e podem diferir ligeiramente dos JPG da distribuição original. Esse desenho foi registrado no
log de decisões antes de o modelo ser aplicado ao CheXpert. As imagens do CheXpert não são quadradas; na configuração principal, elas são redimensionadas para quadrado, como qualquer
imagem enviada à interface, e, como verificação, também com preenchimento das bordas em preto,
preservando as proporções. A AUC de cada classe, com IC95% por bootstrap por paciente, é comparada com
a do teste do NIH. O conjunto é pequeno e tem poucos casos de algumas doenças; classes com menos de 30
casos positivos são reportadas, mas não sustentam conclusões.

## 3.11 Sistema de demonstração

O sistema final é uma página web publicada no Hugging Face Spaces (<https://huggingface.co/spaces/rotriguin/tcc-raio-x>), em que o
modelo roda no navegador de quem acessa, com a biblioteca ONNX Runtime Web: a radiografia não é enviada a
nenhum servidor. A escolha se deve a uma mudança do serviço: o Hugging Face passou a cobrar pelos Spaces
que executam Python, como os feitos com a biblioteca Gradio, e manteve gratuitos os Spaces estáticos,
que só servem arquivos. Uma versão com Gradio, que faz a mesma análise em Python, fica como versão local
e de reserva para a apresentação. O usuário envia uma radiografia (PNG ou JPG) e recebe:

a) uma frase de resumo, com as doenças cujo valor ficou acima do limiar ("Achados acima do limiar:
   Efusão pleural, Atelectasia") ou, se nenhuma ficou, "Nenhum achado acima do limiar entre as 14 doenças
   avaliadas". A interface nunca diz "normal" ou "saudável", porque o modelo só conhece 14 doenças e todo
   limiar deixa passar casos;
b) uma tabela com o valor das 14 doenças, as três estudadas em destaque, cada uma com o limiar da
   validação e a indicação "acima" ou "abaixo do limiar";
c) o mapa de calor Grad-CAM da doença escolhida;
d) um aviso fixo: "Protótipo acadêmico. Não usar para diagnóstico.", com a observação de que o modelo
   foi treinado principalmente com radiografias frontais de adultos e de que imagens muito diferentes
   dessas (foto de tela, criança pequena, incidência lateral, outro exame) geram resultados sem sentido.

O valor exibido é o escore recalibrado por *Platt scaling*, e o limiar é mostrado na mesma escala. Ele
é chamado de "probabilidade estimada", porque as curvas de calibração do teste ficaram próximas da
diagonal (Seção 4.5), com a observação de que a calibração vale para a população do NIH. Se a
calibração não tivesse se sustentado no teste, o valor seria chamado de "escore do modelo".

Para rodar no navegador, o modelo foi exportado para o formato ONNX com uma segunda saída: o mapa Grad-CAM
de cada classe na camada `denseblock4`, calculado em forma fechada. Entre essa camada e o *logit* da
classe $c$ há apenas a normalização em lote final (no modo de avaliação, uma função afim por canal,
$B_k = s_k A_k + t_k$), a ReLU, o *pooling* médio e a camada linear. Por isso, o gradiente do *logit*
é $\partial z_c / \partial A^k_{ij} = w^c_k \, s_k \, [B^k_{ij} > 0] / Z$, e o peso do Grad-CAM, a
média desse gradiente, é $\alpha^c_k = w^c_k \, s_k \, n_k / Z^2$, em que $n_k$ é o número de
posições com $B^k_{ij} > 0$. O pré-processamento (conversão para tons de cinza e os dois
redimensionamentos da biblioteca Pillow) e o pós-processamento do mapa de calor foram reescritos em
JavaScript seguindo os mesmos passos do Python. Testes automáticos comparam cada etapa com a original,
e a própria página tem um autoteste que refaz, no navegador, a análise das imagens de exemplo e a
compara com a do Python. Os critérios de aceitação foram diferença menor que 10⁻⁴ nos escores e 10⁻³
nos mapas de calor, com a imagem vista pela rede idêntica pixel a pixel. Os resultados dessas
verificações estão na Seção 4.10.

## 3.12 Ambiente e reprodutibilidade

O treino foi feito num computador pessoal com uma GPU NVIDIA GeForce RTX 2060, em Windows, com Python
3.10.11, PyTorch 2.14.0 (CUDA 12.6) e torchvision 0.29.0. A primeira época do E1 levou 8,8 minutos
(cerca de 168 imagens de treino por segundo, com o cuDNN determinístico), a segunda, 7,5 minutos, e,
a partir da terceira, cerca de 6,3 minutos (225 imagens por segundo); o E1 completo, com 13 épocas,
levou 85,5 minutos. As versões do Python, do PyTorch, do CUDA e do cuDNN e das principais bibliotecas
usadas no treino e na avaliação (torchvision, NumPy, pandas, scikit-learn, Pillow, pytorch-grad-cam,
Matplotlib, seaborn e PyYAML), o modelo da GPU e o *commit* do código ficam registrados junto com cada
experimento (arquivo `environment.json`); as demais dependências têm versão fixada em `requirements.txt`.

O código é organizado como um pacote Python, com configurações em YAML, e tem 150 testes automáticos.
Eles verificam, entre outras coisas:
- que nenhum paciente aparece em dois conjuntos;
- que as métricas coincidem com as da biblioteca scikit-learn;
- que um treino interrompido e retomado dá o mesmo resultado, na configuração do teste (CPU, com as
  imagens carregadas no processo principal);
- que as duas versões da interface reproduzem os escores e os mapas de calor avaliados.

Os testes rodam a cada alteração, num serviço de integração contínua.

## 3.13 Aspectos éticos

Os dados são disponibilizados anonimizados: o arquivo de metadados identifica cada paciente apenas por um
número sequencial (Patient ID) e traz, como dados demográficos, somente a idade e o sexo, sem nome, data
de nascimento, datas de exame ou qualquer outro identificador pessoal. Como o trabalho utiliza
exclusivamente essa base pública e anonimizada, sem contato com pacientes e sem acesso a dados que
permitam identificá-los, ele não foi submetido a Comitê de Ética em Pesquisa [CONFERIR com o orientador
se cabe citar a Resolução CNS nº 510/2016, art. 1º, parágrafo único]. Conforme solicitado pelo NIH, o
trabalho cita Wang et al. (2017a) e reconhece o NIH Clinical Center como fornecedor dos dados. O uso do
CheXpert segue o acordo de uso para pesquisa da Stanford AIMI (*Stanford Research Agreement*), aceito na
plataforma Redivis para os dois conjuntos usados, o CheXpert Plus e o CheXlocalize. Por isso, o
repositório do trabalho não redistribui imagens nem rótulos do CheXpert, só as métricas agregadas.
