# 4 RESULTADOS

Este capítulo apresenta os resultados na ordem em que foram obtidos: o treino (Seção 4.1), o desempenho
do modelo final no teste (Seções 4.2 a 4.5), as comparações entre experimentos (Seção 4.6), a análise
por subgrupos (Seção 4.7), os mapas de calor (Seção 4.8), a validação externa (Seção 4.9) e o sistema de
demonstração (Seção 4.10). Salvo indicação em contrário, os valores são do conjunto de teste da divisão
principal (16.822 imagens de 4.790 pacientes), com intervalo de confiança de 95% (IC95%) por bootstrap
por paciente entre parênteses. O modelo final é o E1, escolhido na validação (Seção 3.7).

## 4.1 Treino

A Figura 2 mostra as curvas de treino do E1 e do E2. No E1, a AUC média de validação subiu rapidamente
nas primeiras épocas (0,802 na primeira, 0,834 na quinta) e atingiu o máximo, 0,838, na oitava época,
logo depois da primeira redução da taxa de aprendizado (de $10^{-4}$ para $10^{-5}$). Nas cinco épocas
seguintes, a AUC de validação ficou estável ou caiu levemente, e a parada antecipada encerrou o treino
na 13ª época, depois de 85,5 minutos. A perda de treino continuou caindo até o fim (de 0,165 para
0,121), enquanto a perda de validação ficou praticamente constante a partir da terceira época (entre
0,144 e 0,146): o modelo passou a ajustar detalhes do treino que não se generalizam, e a parada antecipada
cumpriu o papel de evitar esse sobreajuste. Os valores de AUC desta seção vêm do registro do treino,
calculados em precisão mista a cada época; na reavaliação em precisão completa usada para a escolha do
modelo (Seção 3.7), as AUCs médias de validação na melhor época são 0,838 (E1) e 0,832 (E2).

O E2, com ponderação de classes, teve curva de validação mais irregular, atingiu o máximo (0,833 no
registro do treino) na 14ª época e parou na 19ª, depois de 121,7 minutos. A perda de validação do E2 passou a subir depois da
oitava época (de 0,941 para 1,087 na 19ª), enquanto a AUC ainda melhorava até a 14ª, o que ilustra por que a perda não é um bom critério de parada quando a
função de perda é ponderada: ela passa a refletir a escala dos escores, e não só a ordenação dos exames.

**Figura 2 – Curvas de treino do E1 e do E2**

(`results/figures/curvas_treino_e1_baseline_e2_posweight.png`)

Fonte: elaborado pelo autor.

[PREENCHER: se a figura for refeita com os demais experimentos (`python -m chestxray.evaluate --curves
results/runs/e1_baseline results/runs/e3_noaug results/runs/e4_scratch`), comentar a convergência do E4,
que parte de pesos aleatórios.]

## 4.2 Desempenho do modelo final

A Tabela 3 mostra a AUC e a AUPRC do E1 nas 14 classes. A AUC média foi de **0,841 (0,833–0,848)**.
Entre as três doenças estudadas, a efusão pleural teve a maior AUC, **0,886 (0,874–0,896)**, seguida da
atelectasia, **0,816 (0,802–0,829)**, e da pneumonia, **0,751 (0,714–0,787)**. As maiores AUCs foram de
hérnia (0,939), cardiomegalia (0,920) e enfisema (0,915), doenças com sinais anatômicos marcantes; a
hérnia, porém, tem só 28 casos no teste, e seu intervalo vai de 0,862 a 0,993. As menores foram de
infiltração (0,720) e pneumonia (0,751), as duas classes de aspecto mais inespecífico. A Figura 3 mostra
as curvas ROC das três doenças estudadas.

**Tabela 3 – AUC e AUPRC do modelo final (E1) no conjunto de teste**

| Doença | Casos | Prevalência | AUC (IC95%) | AUPRC (IC95%) |
|---|---|---|---|---|
| Atelectasia * | 1.671 | 9,9% | 0,816 (0,802–0,829) | 0,361 (0,328–0,393) |
| Cardiomegalia | 455 | 2,7% | 0,920 (0,902–0,935) | 0,346 (0,266–0,414) |
| Efusão pleural * | 2.123 | 12,6% | 0,886 (0,874–0,896) | 0,543 (0,510–0,576) |
| Infiltração | 2.946 | 17,5% | 0,720 (0,705–0,733) | 0,344 (0,318–0,369) |
| Massa | 876 | 5,2% | 0,851 (0,832–0,870) | 0,338 (0,293–0,387) |
| Nódulo | 950 | 5,6% | 0,788 (0,770–0,807) | 0,230 (0,201–0,266) |
| Pneumonia * | 194 | 1,2% | 0,751 (0,714–0,787) | 0,044 (0,033–0,064) |
| Pneumotórax | 863 | 5,1% | 0,888 (0,856–0,910) | 0,370 (0,314–0,423) |
| Consolidação | 698 | 4,1% | 0,783 (0,764–0,801) | 0,125 (0,107–0,147) |
| Edema | 357 | 2,1% | 0,896 (0,879–0,912) | 0,171 (0,138–0,218) |
| Enfisema | 331 | 2,0% | 0,915 (0,893–0,933) | 0,324 (0,261–0,385) |
| Fibrose | 254 | 1,5% | 0,823 (0,795–0,849) | 0,079 (0,060–0,110) |
| Espessamento pleural | 541 | 3,2% | 0,797 (0,775–0,818) | 0,152 (0,123–0,188) |
| Hérnia | 28 | 0,2% | 0,939 (0,862–0,993) | 0,488 (0,218–0,707) |
| Média (14 classes) | | | 0,841 (0,833–0,848) | |

Fonte: elaborado pelo autor (`results/tables/e1_baseline/metricas_test`). * Doenças estudadas.

**Figura 3 – Curvas ROC das três doenças estudadas no teste (E1)**

(`results/figures/e1_baseline/roc_foco_test.png`; as 14 classes estão em `roc_14_test.png`, para o
apêndice)

Fonte: elaborado pelo autor.

A AUPRC conta outra parte da história (Figura 4). Para a efusão pleural, 0,543, cerca de quatro vezes a
prevalência (12,6%); para a atelectasia, 0,361, 3,6 vezes a prevalência (9,9%); e para a pneumonia,
apenas **0,044**, 3,9 vezes a prevalência de 1,15%. Ou seja, a AUPRC fica acima do acaso nas três
doenças, mas essas razões não se comparam entre classes: com a mesma curva ROC, quanto mais rara a
doença, maior tende a ser a razão entre AUPRC e prevalência. Na pneumonia somam-se dois efeitos: a
doença é rara, e a ordenação é a mais fraca das três (AUC 0,751). Reponderando o teste para a mesma
prevalência da pneumonia (1,15%), a ordenação da efusão daria AUPRC de cerca de 0,11 e a da
atelectasia, cerca de 0,07, contra 0,044 da pneumonia; já a ordenação da pneumonia, com a prevalência
da efusão (12,6%), daria cerca de 0,33. A raridade explica a maior parte da AUPRC baixa, mas não toda.
Na prática, poucos dos exames de maior escore têm de fato pneumonia. Esse é o limite prático mais
importante do modelo para a pneumonia e é discutido no Capítulo 5.

**Figura 4 – Curvas precisão-revocação das três doenças estudadas no teste (E1)**

(`results/figures/e1_baseline/pr_foco_test.png`; a linha pontilhada de cada cor é a prevalência)

Fonte: elaborado pelo autor.

## 4.3 Comparação com a literatura

A Tabela 4 compara a AUC do E1 com as de Wang et al. (2017) e do CheXNet. O modelo supera a referência
de Wang et al. nas 14 classes, e a média (0,841) é igual à do CheXNet (0,8414). Por classe, o valor do
CheXNet está dentro do IC95% deste trabalho em 11 das 14 classes, incluindo atelectasia (0,8094 contra
0,816, IC 0,802–0,829) e pneumonia (0,7680 contra 0,751, IC 0,714–0,787). A efusão pleural ficou acima
do CheXNet (0,886, IC 0,874–0,896, contra 0,8638), e infiltração (0,720 contra 0,7345) e enfisema (0,915
contra 0,9371) ficaram abaixo. Como os valores do CheXNet foram obtidos em outro conjunto de teste e não
têm intervalo de confiança publicado, essa comparação indica apenas que os resultados estão **na mesma
faixa**, e não que um modelo seja melhor que o outro.

**Tabela 4 – AUC do modelo final comparada à literatura**

| Doença | Wang et al. (2017) | CheXNet (2017) | Este trabalho (IC95%) |
|---|---|---|---|
| Atelectasia * | 0,716 | 0,8094 | 0,816 (0,802–0,829) |
| Cardiomegalia | 0,807 | 0,9248 | 0,920 (0,902–0,935) |
| Efusão pleural * | 0,784 | 0,8638 | 0,886 (0,874–0,896) |
| Infiltração | 0,609 | 0,7345 | 0,720 (0,705–0,733) |
| Massa | 0,706 | 0,8676 | 0,851 (0,832–0,870) |
| Nódulo | 0,671 | 0,7802 | 0,788 (0,770–0,807) |
| Pneumonia * | 0,633 | 0,7680 | 0,751 (0,714–0,787) |
| Pneumotórax | 0,806 | 0,8887 | 0,888 (0,856–0,910) |
| Consolidação | 0,708 | 0,7901 | 0,783 (0,764–0,801) |
| Edema | 0,835 | 0,8878 | 0,896 (0,879–0,912) |
| Enfisema | 0,815 | 0,9371 | 0,915 (0,893–0,933) |
| Fibrose | 0,769 | 0,8047 | 0,823 (0,795–0,849) |
| Espessamento pleural | 0,708 | 0,8062 | 0,797 (0,775–0,818) |
| Hérnia | 0,767 | 0,9164 | 0,939 (0,862–0,993) |
| Média (14 classes) | 0,738 | 0,8414 | 0,841 (0,833–0,848) |

Fonte: elaborado pelo autor, com os valores da literatura da Tabela 2 de Rajpurkar et al. (2017)
(`results/tables/e1_baseline/comparacao_literatura`). * Doenças estudadas.

## 4.4 Desempenho no limiar de decisão

A Tabela 5 mostra o desempenho das três doenças no limiar escolhido na validação pelo índice de Youden
(as 14 classes estão em `results/tables/e1_baseline/limiares_teste`), e a Figura 5, as matrizes de
confusão. Na efusão pleural, o modelo detectou 82,5% dos casos (1.751 de 2.123) com especificidade de
78,7%; na atelectasia, 74,2% com especificidade de 73,4%; na pneumonia, 68,6% (133 de 194) com
especificidade de 67,8%.

**Tabela 5 – Desempenho no teste com o limiar de Youden da validação (E1)**

| Doença | Limiar | Sensibilidade | Especificidade | VPP | VPN | F1 |
|---|---|---|---|---|---|---|
| Pneumonia | 0,011 | 0,686 | 0,678 | 0,024 | 0,995 | 0,047 |
| Atelectasia | 0,094 | 0,742 | 0,734 | 0,235 | 0,963 | 0,357 |
| Efusão pleural | 0,108 | 0,825 | 0,787 | 0,359 | 0,969 | 0,500 |

Fonte: elaborado pelo autor (`results/tables/e1_baseline/limiares_teste`).

**Figura 5 – Matrizes de confusão das três doenças estudadas no teste (E1)**

(`results/figures/e1_baseline/matriz_confusao_foco_test.png`)

Fonte: elaborado pelo autor.

O VPP e o VPN mostram o efeito da prevalência. Na pneumonia, o modelo marcou 5.480 exames como
positivos, e só 133 deles (2,4%) tinham o rótulo de pneumonia. Entre os exames abaixo do limiar, 99,5%
não tinham pneumonia, mas esse valor alto se deve sobretudo à prevalência: sem modelo algum, 98,8% dos
exames já não têm a doença. Na prática, o resultado positivo leva a chance de pneumonia de 1,2% para
2,4%, e o negativo a reduz para 0,5% (razões de verossimilhança de cerca de 2,1 e 0,46). Com 31% dos
casos (61 de 194) abaixo do limiar, nesse ponto de operação o modelo sozinho não serve bem nem para
confirmar nem para descartar a doença. Um limiar mais alto trocaria sensibilidade por menos alarmes
falsos, e um mais baixo faria o contrário. Os limiares são baixos (0,011 para pneumonia) porque o modelo, treinado sem ponderação, dá escores próximos
da prevalência de cada doença (ver a Seção 4.5).

## 4.5 Calibração

A Figura 6 mostra os diagramas de confiabilidade das três doenças no teste, antes e depois do *Platt
scaling* ajustado na validação, e a Tabela 6, o escore de Brier. O E1 já saiu do treino bem calibrado:
os pontos dos escores brutos ficam próximos da diagonal, e o Brier praticamente não muda com a
recalibração (de 0,0776 para 0,0771 na efusão, de 0,0763 para 0,0757 na atelectasia e de 0,0113 para
0,0112 na pneumonia). A recalibração corrige principalmente a faixa de escores mais altos, em que o
modelo bruto superestimava a chance de doença (por exemplo, na atelectasia, a última faixa tinha escore
médio de 0,45 e 38% de casos; depois do Platt, 0,38). Na pneumonia, mesmo a faixa de escores mais altos
tem escore recalibrado médio de cerca de 5% (e 4,4% de casos), coerente com a baixa prevalência; exames
individuais chegam a 24%.

**Tabela 6 – Escore de Brier no teste, antes e depois da recalibração**

| Doença | E1: bruto | E1: após Platt | E2: bruto | E2: após Platt |
|---|---|---|---|---|
| Pneumonia | 0,0113 | 0,0112 | 0,1258 | 0,0113 |
| Atelectasia | 0,0763 | 0,0757 | 0,1819 | 0,0774 |
| Efusão pleural | 0,0776 | 0,0771 | 0,1422 | 0,0792 |

Fonte: elaborado pelo autor (`results/tables/e1_baseline/calibracao_teste` e
`results/tables/e2_posweight/calibracao_teste`).

**Figura 6 – Diagramas de confiabilidade no teste (E1)**

(`results/figures/e1_baseline/calibracao_foco_test.png`)

Fonte: elaborado pelo autor.

O contraste com o E2 confirma o efeito esperado da ponderação de classes (Seção 2.6): com `pos_weight`,
os escores brutos ficam muito acima da frequência real, e o Brier da pneumonia é mais de dez vezes o do
E1 (0,1258 contra 0,0113). Depois do Platt, os dois modelos ficam com Brier praticamente igual: o
excesso do E2 vinha da escala dos escores, que a recalibração corrige. A ordenação, e portanto a AUC,
não muda com o Platt, que é uma transformação crescente (Seção 2.9). A pequena vantagem que sobra para
o E1 na atelectasia (0,0757 contra 0,0774) e na efusão (0,0771 contra 0,0792) é coerente com a AUC maior
dele nessas doenças; na pneumonia, a diferença é desprezível (0,0112 contra 0,0113).

Com as curvas recalibradas próximas da diagonal nas três doenças, a interface de demonstração
[PREENCHER: decisão do autor. Recomendação a partir desta figura: chamar o valor de "probabilidade
estimada", com a nota de que ela vale para a população do NIH, e registrar a decisão na seção 10 do
plano.]

## 4.6 Comparação entre experimentos

A Tabela 7 mostra os experimentos no mesmo conjunto de teste, e a Tabela 8, as diferenças pareadas em
relação ao E1.

**Tabela 7 – Experimentos na divisão principal (teste)**

| Experimento | AUC média (IC95%) | Pneumonia: AUC | Atelectasia: AUC | Efusão: AUC | Épocas (melhor / total) | Treino (min) |
|---|---|---|---|---|---|---|
| E1 (base) | 0,841 (0,833–0,848) | 0,751 (0,714–0,787) | 0,816 (0,802–0,829) | 0,886 (0,874–0,896) | 8 / 13 | 85,5 |
| E2 (`pos_weight`) | 0,835 (0,828–0,842) | 0,761 (0,728–0,791) | 0,802 (0,787–0,816) | 0,882 (0,870–0,893) | 14 / 19 | 121,7 |
| E3 (sem aumento de dados) | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| E4 (sem ImageNet) | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |

Fonte: elaborado pelo autor (`results/tables/experimentos_test`; a tabela completa, com a AUPRC, tem
mais colunas e pode ir para o apêndice).

**Tabela 8 – Diferenças pareadas de AUC em relação ao E1 (teste)**

| Comparação | Classe | Diferença (IC95%) | Significativa? |
|---|---|---|---|
| E2 − E1 | Média (14 classes) | −0,006 (−0,010 a +0,0002) | não |
| E2 − E1 | Atelectasia | −0,014 (−0,021 a −0,008) | sim |
| E2 − E1 | Efusão pleural | −0,004 (−0,008 a −0,0003) | sim |
| E2 − E1 | Pneumonia | +0,009 (−0,011 a 0,029) | não |
| E3 − E1 | [PREENCHER] | | |
| E4 − E1 | [PREENCHER] | | |

Fonte: elaborado pelo autor (`results/tables/diferencas_pareadas_test`). Uma diferença é significativa
quando o IC95% não contém o zero; limites próximos de zero são mostrados com quatro casas decimais.

### 4.6.1 Ponderação de classes (E2)

A ponderação não melhorou o modelo. Na média das 14 classes, a diferença foi de −0,006 (IC95% −0,010 a
+0,0002), no limite da significância: o intervalo inclui o zero por muito pouco. Na atelectasia e na
efusão, o E2 foi significativamente pior (−0,014 e −0,004); na pneumonia, a mais rara das três doenças
estudadas (1,2% do teste) e, entre elas, aquela em que a ponderação mais poderia ajudar, a diferença foi
positiva, +0,009, mas não significativa (−0,011 a 0,029). O mesmo padrão já aparecia na validação,
antes de qualquer contato com o teste: média sem diferença significativa, E2 significativamente pior em
atelectasia e efusão e pneumonia sem diferença significativa (lá com o sinal oposto, E2 − E1 = −0,008;
IC95% −0,031 a 0,016). A oscilação do sinal na pneumonia é compatível com ruído, e o resultado no teste é
coerente com a escolha do E1. Somando o
efeito na calibração (Seção 4.5) e o treino 42% mais longo, a ponderação não trouxe vantagem neste
problema.

### 4.6.2 Aumento de dados (E3)

[PREENCHER a partir de `experimentos_test` e `diferencas_pareadas_test`: AUC média do E3 e diferença
E3 − E1 com IC95%; se significativa, em que direção; épocas até a parada (sem aumento de dados, o
sobreajuste costuma chegar mais cedo, o que deve aparecer nas curvas de treino).]

### 4.6.3 Variação entre sementes

**Tabela 9 – Configuração final repetida com três sementes (teste)**

[PREENCHER com `results/tables/seeds_test.md`: AUC média e das três doenças para as sementes 42, 43 e
44, e a linha de média ± desvio-padrão.]

Fonte: elaborado pelo autor.

[PREENCHER: comparar o desvio-padrão entre sementes com as diferenças da Tabela 8. Uma diferença entre
experimentos menor que a variação entre sementes não deve ser interpretada como efeito da configuração.]

### 4.6.4 Transferência de aprendizado (E4)

[PREENCHER: AUC média do E4 e diferença E4 − E1 com IC95%; número de épocas até a parada. Lembrar que o
E4 teve o mesmo orçamento de épocas (Seção 3.7).]

### 4.6.5 Divisão oficial do NIH (E5)

**Tabela 10 – E1 treinado e testado na divisão oficial do NIH (E5)**

[PREENCHER com `results/tables/e5_official/metricas_test.md` e `comparacao_literatura.md`: AUC e AUPRC
das três doenças, com a prevalência ao lado (no teste oficial, 2,2% de pneumonia, 12,8% de atelectasia e
18,2% de efusão), e a AUC média. Não colocar lado a lado com E1–E4: o conjunto de teste é outro.]

Fonte: elaborado pelo autor.

## 4.7 Análise por subgrupos

A Figura 7 e a Tabela 11 mostram a AUC das três doenças por sexo, faixa etária e incidência.

**Tabela 11 – AUC por subgrupo no teste (E1)**

| Subgrupo | Pneumonia | Atelectasia | Efusão pleural |
|---|---|---|---|
| Masculino | 0,744 (0,696–0,784) | 0,829 (0,812–0,845) | 0,888 (0,872–0,901) |
| Feminino | 0,756 (0,689–0,826) | 0,797 (0,775–0,819) | 0,882 (0,867–0,896) |
| < 40 anos | 0,741 (0,685–0,800) | 0,824 (0,797–0,847) | 0,899 (0,880–0,917) |
| 40–60 anos | 0,754 (0,696–0,809) | 0,815 (0,794–0,835) | 0,879 (0,863–0,894) |
| > 60 anos | 0,758 (0,650–0,857) | 0,786 (0,761–0,811) | 0,879 (0,858–0,899) |
| PA | 0,722 (0,659–0,782) | 0,828 (0,813–0,843) | 0,908 (0,896–0,920) |
| AP | 0,746 (0,697–0,792) | 0,789 (0,769–0,810) | 0,844 (0,826–0,863) |
| Geral | 0,751 (0,714–0,787) | 0,816 (0,802–0,829) | 0,886 (0,874–0,896) |

Fonte: elaborado pelo autor (`results/tables/e1_baseline/subgrupos_teste`; o número de casos de cada
subgrupo está na tabela de origem).

**Figura 7 – AUC por subgrupo no teste (E1)**

(`results/figures/e1_baseline/subgrupos_foco_test.png`)

Fonte: elaborado pelo autor.

Por **sexo**, o desempenho foi semelhante na efusão e na pneumonia; na atelectasia, a AUC foi maior nos
homens (0,829) que nas mulheres (0,797), com intervalos que quase não se sobrepõem. Por **idade**, a
efusão e a pneumonia ficaram estáveis, e a atelectasia caiu com a idade (de 0,824 abaixo de 40 anos para
0,786 acima de 60).

A maior diferença apareceu na **incidência**. Nos exames AP, a AUC foi menor que nos PA para a efusão
(0,844 contra 0,908, sem sobreposição dos intervalos) e para a atelectasia (0,789 contra 0,828). Na
pneumonia, os intervalos são largos e se sobrepõem (PA 0,722; AP 0,746), mas as duas estimativas ficaram
abaixo da AUC geral (0,751). O Capítulo 5 discute o que esses padrões indicam sobre um possível atalho
pela incidência.

## 4.8 Mapas de calor

[PREENCHER depois de rodar `python -m chestxray.gradcam --config configs/experiments/e1_baseline.yaml`.
Sugestão de estrutura:]

**Figura 8 – Grad-CAM: acertos e erros na efusão pleural**

(`results/figures/e1_baseline/gradcam/galeria_effusion.png`; as galerias de pneumonia e atelectasia vão
para o texto ou para o apêndice)

Fonte: elaborado pelo autor.

[PREENCHER: descrever o que os mapas mostram em cada grupo. Perguntas a responder: nos verdadeiros
positivos de efusão, o mapa se concentra nos seios costofrênicos e nas bases? Nos falsos positivos, há
algo em comum (dispositivos, cabos, exames AP, outra doença na mesma região)? Nos falsos negativos, o
achado é sutil ou o rótulo parece errado?]

**Figura 9 – Comparação das camadas-alvo**

(`results/figures/e1_baseline/gradcam/camadas_foco.png`)

Fonte: elaborado pelo autor.

**Tabela 12 – *Pointing game* nas imagens de teste com caixa delimitadora**

[PREENCHER com `results/tables/e1_baseline/pointing_game.md`: por doença e camada, número de imagens,
acertos, taxa com IC95% de Wilson e a taxa do centro da imagem. Destacar as três doenças estudadas
(atelectasia, efusão e pneumonia, com cerca de 20 imagens cada).]

Fonte: elaborado pelo autor.

[PREENCHER: qual camada foi escolhida e por quê; se a taxa de acerto de cada doença ficou acima da taxa
do centro da imagem (se o IC de Wilson não cobre a taxa do centro, a diferença é clara).]

**Figura 10 – Mapas de calor e caixas dos radiologistas (efusão pleural)**

(`results/figures/e1_baseline/gradcam/caixas_effusion.png`)

Fonte: elaborado pelo autor.

## 4.9 Validação externa no CheXpert

[PREENCHER depois de rodar `python -m chestxray.external ...`.]

**Tabela 13 – AUC no conjunto de validação do CheXpert, comparada ao teste do NIH**

[PREENCHER com `results/tables/e1_baseline/validacao_externa_chexpert.md`: casos, AUC no CheXpert com
IC95%, AUC no teste do NIH e diferença, para as 7 classes em comum; marcar com † as classes com menos de
30 casos.]

Fonte: elaborado pelo autor.

[PREENCHER: número de imagens frontais e de pacientes; se a AUC caiu, em quais classes, e se a queda é
maior que a incerteza; lembrar que os rótulos do CheXpert de validação são de radiologistas, enquanto os
do NIH vêm de laudos processados automaticamente, e que isso pode aumentar ou reduzir a AUC. Registrar
também o resultado com `--fit pad`, se diferente.]

## 4.10 Sistema de demonstração

**Figura 11 – Interface de demonstração**

[PREENCHER: captura de tela da interface no Hugging Face Spaces com um exemplo em que a efusão fica acima
do limiar, mostrando a frase de resumo, a tabela e o mapa de calor.]

Fonte: elaborado pelo autor.

A exportação do modelo para a interface foi conferida contra as predições da avaliação: nas três imagens
de teste verificadas, a diferença máxima entre os escores da interface e os do arquivo de predições foi
de 3,6 × 10⁻⁷ (o critério era 10⁻⁴). Cada análise, com o mapa de calor, levou entre 0,3 e 0,4 s em CPU no
computador de desenvolvimento. [PREENCHER: tempo de resposta medido no Space e link.]
