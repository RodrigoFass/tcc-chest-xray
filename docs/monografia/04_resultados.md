# 4 RESULTADOS

Este capítulo apresenta os resultados na ordem em que foram obtidos: o treino (Seção 4.1), o desempenho
do modelo final no teste (Seções 4.2 a 4.5), as comparações entre experimentos (Seção 4.6), a análise
por subgrupos (Seção 4.7), os mapas de calor (Seção 4.8), a validação externa (Seção 4.9) e o sistema de
demonstração (Seção 4.10). Salvo indicação em contrário, os valores são do conjunto de teste da divisão
principal (16.822 imagens de 4.790 pacientes), com intervalo de confiança de 95% (IC95%) por bootstrap
por paciente entre parênteses. O modelo final é o E1, escolhido na validação (Seção 3.7).

## 4.1 Treino

A Figura 2 mostra as curvas de treino dos experimentos da divisão principal. No E1, a AUC média de validação subiu rapidamente
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

Sem aumento de dados (E3), o sobreajuste chegou cedo: a AUC de validação atingiu o máximo (0,829) já
na terceira época e, a partir daí, a perda de treino despencou (de 0,138 para 0,063 na oitava época)
enquanto a perda de validação subia (de 0,145 para 0,175). A parada antecipada encerrou o treino na
oitava época, depois de 51,6 minutos. O E4, que parte de pesos aleatórios, convergiu devagar: a AUC de
validação ainda subia na 15ª época (0,809), deu um salto com a primeira redução da taxa de aprendizado
(0,817 na 18ª) e depois ficou estável até a 30ª, o limite do orçamento, com máximo de 0,817 na 25ª
época e 200,3 minutos de treino. A perda de treino do E4 terminou em 0,137, acima da do E1 (0,121): a
rede treinada do zero não chegou a se ajustar aos dados de treino no mesmo grau.

**Figura 2 – Curvas de treino do E1 ao E4**

(`results/figures/curvas_treino_e1_baseline_e2_posweight_e3_noaug_e4_scratch.png`; as curvas das três
sementes estão em `curvas_treino_e1_baseline_e1_baseline_seed43_e1_baseline_seed44.png` e as do E5 em
`curvas_treino_e5_official.png`, para o apêndice)

Fonte: elaborado pelo autor.

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

A Tabela 4 compara a AUC do E1 com as de Wang et al. (2017b) e do CheXNet. O modelo supera a referência
de Wang et al. nas 14 classes, e a média (0,841) é igual à do CheXNet (0,8414). Por classe, o valor do
CheXNet está dentro do IC95% deste trabalho em 11 das 14 classes, incluindo atelectasia (0,8094 contra
0,816, IC 0,802–0,829) e pneumonia (0,7680 contra 0,751, IC 0,714–0,787). A efusão pleural ficou acima
do CheXNet (0,886, IC 0,874–0,896, contra 0,8638), e infiltração (0,720 contra 0,7345) e enfisema (0,915
contra 0,9371) ficaram abaixo. Como os valores do CheXNet foram obtidos em outro conjunto de teste e não
têm intervalo de confiança publicado, essa comparação indica apenas que os resultados estão **na mesma
faixa**, e não que um modelo seja melhor que o outro.

**Tabela 4 – AUC do modelo final comparada à literatura**

| Doença | Wang et al. (2017b) | CheXNet (2017) | Este trabalho (IC95%) |
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

Fonte: elaborado pelo autor, com os valores da literatura da Tabela 2 de Rajpurkar et al. (2017); os de
Wang et al. são os da versão 4 do artigo no arXiv (Seção 2.11)
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

Com as curvas recalibradas próximas da diagonal nas três doenças, a interface de demonstração chama o
valor exibido de "probabilidade estimada", com a nota de que ela vale para a população do NIH (Seção
3.11).

## 4.6 Comparação entre experimentos

A Tabela 7 mostra os experimentos no mesmo conjunto de teste, e a Tabela 8, as diferenças pareadas em
relação ao E1.

**Tabela 7 – Experimentos na divisão principal (teste)**

| Experimento | AUC média (IC95%) | Pneumonia: AUC | Atelectasia: AUC | Efusão: AUC | Épocas (melhor / total) | Treino (min) |
|---|---|---|---|---|---|---|
| E1 (base) | 0,841 (0,833–0,848) | 0,751 (0,714–0,787) | 0,816 (0,802–0,829) | 0,886 (0,874–0,896) | 8 / 13 | 85,5 |
| E2 (`pos_weight`) | 0,835 (0,828–0,842) | 0,761 (0,728–0,791) | 0,802 (0,787–0,816) | 0,882 (0,870–0,893) | 14 / 19 | 121,7 |
| E3 (sem aumento de dados) | 0,830 (0,822–0,837) | 0,773 (0,739–0,804) | 0,807 (0,792–0,821) | 0,882 (0,871–0,892) | 3 / 8 | 51,6 |
| E4 (sem ImageNet) | 0,821 (0,812–0,829) | 0,758 (0,722–0,790) | 0,792 (0,776–0,806) | 0,877 (0,865–0,888) | 25 / 30 | 200,3 |

Fonte: elaborado pelo autor (`results/tables/experimentos_test`; a tabela completa, com a AUPRC, tem
mais colunas e pode ir para o apêndice).

**Tabela 8 – Diferenças pareadas de AUC em relação ao E1 (teste)**

| Comparação | Classe | Diferença (IC95%) | Significativa? |
|---|---|---|---|
| E2 − E1 | Média (14 classes) | −0,006 (−0,010 a +0,0002) | não |
| E2 − E1 | Atelectasia | −0,014 (−0,021 a −0,008) | sim |
| E2 − E1 | Efusão pleural | −0,004 (−0,008 a −0,0003) | sim |
| E2 − E1 | Pneumonia | +0,009 (−0,011 a 0,029) | não |
| E3 − E1 | Média (14 classes) | −0,010 (−0,015 a −0,006) | sim |
| E3 − E1 | Atelectasia | −0,009 (−0,016 a −0,003) | sim |
| E3 − E1 | Efusão pleural | −0,004 (−0,008 a −0,0002) | sim |
| E3 − E1 | Pneumonia | +0,022 (−0,003 a 0,047) | não |
| E4 − E1 | Média (14 classes) | −0,020 (−0,026 a −0,015) | sim |
| E4 − E1 | Atelectasia | −0,024 (−0,032 a −0,016) | sim |
| E4 − E1 | Efusão pleural | −0,009 (−0,013 a −0,005) | sim |
| E4 − E1 | Pneumonia | +0,006 (−0,018 a 0,031) | não |

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

Sem aumento de dados, o modelo foi pior. A AUC média caiu 0,010 (IC95% −0,015 a −0,006), e a queda foi
significativa na atelectasia (−0,009) e na efusão (−0,004). O resultado já aparecia na validação, com
diferença média de −0,008 (−0,015 a −0,003). Na pneumonia, a diferença no teste foi positiva, +0,022,
mas não significativa (−0,003 a 0,047), e vai na direção oposta à das outras classes. Com 194 casos e
com a variação entre sementes medida na Seção 4.6.3, não há base para concluir que o aumento de dados
prejudique essa classe. Como mostrou a Seção 4.1, sem aumento de dados o sobreajuste começou logo
depois da terceira época, e o treino terminou em pouco mais da metade do tempo do E1.

### 4.6.3 Variação entre sementes

**Tabela 9 – Configuração final repetida com três sementes (teste)**

| Semente | AUC média | Pneumonia: AUC | Atelectasia: AUC | Efusão pleural: AUC |
|---|---|---|---|---|
| 42 (E1) | 0,841 | 0,751 | 0,816 | 0,886 |
| 43 | 0,842 | 0,766 | 0,819 | 0,885 |
| 44 | 0,837 | 0,757 | 0,813 | 0,881 |
| Média ± desvio-padrão | 0,840 ± 0,003 | 0,758 ± 0,008 | 0,816 ± 0,003 | 0,884 ± 0,002 |

Fonte: elaborado pelo autor (`results/tables/seeds_test`, que traz também a AUPRC).

Trocar só a semente, com a mesma divisão dos dados e a mesma configuração, mudou a AUC média em até
0,005 e a da pneumonia em até 0,015 (de 0,751 a 0,766). Na AUC média, a semente 42, usada nas
comparações, ficou no meio das três. Essa variação vem do treino (inicialização da camada final, ordem das imagens e sorteios
do aumento de dados), e os intervalos da Tabela 8 não a incluem: o bootstrap pareado considera a amostra
de teste, mas cada experimento foi treinado uma única vez. Como a diferença entre duas execuções isoladas
tem desvio-padrão cerca de $\sqrt{2}$ vezes o de uma execução, a variação esperada só pela semente
numa diferença da Tabela 8 é de cerca de 0,004 na média, 0,004 na atelectasia, 0,003 na efusão e 0,011
na pneumonia.

Com essa escala, as diferenças da Tabela 8 se dividem em três grupos. São várias vezes maiores que a
variação entre sementes as quedas do E4 na média (−0,020), na atelectasia (−0,024) e na efusão
(−0,009) e a do E2 na atelectasia (−0,014). Ficam em torno de duas vezes essa variação as quedas do E3
na média (−0,010) e na atelectasia (−0,009): são prováveis, mas menos seguras do que o intervalo sugere.
Ficam na mesma ordem da variação entre sementes a queda do E2 na média (−0,006), as da efusão no E2 e
no E3 (−0,004) e todas as diferenças na pneumonia (+0,009, +0,022 e +0,006), que não devem ser
interpretadas como efeito da configuração. Com três sementes, o próprio desvio-padrão é uma estimativa
imprecisa, e esses limites são aproximados.

### 4.6.4 Transferência de aprendizado (E4)

Sem a transferência de aprendizado, com o mesmo orçamento de até 30 épocas (Seção 3.7), o modelo foi o
pior da divisão principal: AUC média de 0,821, 0,020 abaixo do E1 (IC95% −0,026 a −0,015), com quedas
significativas na atelectasia (−0,024) e na efusão (−0,009); na pneumonia, a diferença foi de +0,006,
não significativa (−0,018 a 0,031). O E4 usou todo o orçamento (melhor época na 25ª, fim na 30ª) e levou
200,3 minutos, mais que o dobro do E1. Ainda assim, 0,821 fica bem acima da referência de Wang et al.
(2017b; 0,738, Tabela 4): o ChestX-ray14 é grande o bastante para uma rede treinada do zero aprender boa parte
da tarefa, e a transferência de aprendizado acelerou a convergência (melhor época na 8ª, contra a 25ª) e
melhorou o resultado final.

### 4.6.5 Divisão oficial do NIH (E5)

**Tabela 10 – E1 treinado e testado na divisão oficial do NIH (E5)**

| Doença | Casos | Prevalência | AUC (IC95%) | AUPRC (IC95%) |
|---|---|---|---|---|
| Atelectasia | 3.279 | 12,8% | 0,770 (0,756–0,782) | 0,336 (0,313–0,359) |
| Efusão pleural | 4.658 | 18,2% | 0,829 (0,819–0,838) | 0,519 (0,493–0,545) |
| Pneumonia | 555 | 2,2% | 0,716 (0,694–0,739) | 0,052 (0,044–0,064) |
| Média (14 classes) | | | 0,811 (0,804–0,816) | |

Fonte: elaborado pelo autor (`results/tables/e5_official/metricas_test`; as 14 classes e a comparação
com a literatura estão em `comparacao_literatura`, na mesma pasta).

No teste oficial (25.596 imagens de 2.797 pacientes), a AUC média do E5 foi de 0,811 (0,804–0,816),
contra 0,840 na validação do próprio E5, tirada por paciente do conjunto oficial de treino. A queda
aparece nas três doenças: 0,829 na efusão, 0,770 na atelectasia e 0,716 na pneumonia, contra 0,898,
0,833 e 0,770 na validação. Na divisão principal, a mesma configuração teve AUC de teste praticamente
igual à de validação (0,841 contra 0,838), o que indica que a diferença está no conjunto de teste
oficial, e não no treino. Esse conjunto é diferente do resto da base: tem 9,2 imagens por paciente,
contra 3,6 no conjunto todo, ou seja, concentra pacientes com muitos exames de acompanhamento,
provavelmente internados, e tem prevalências mais altas (Seção 3.4). Trabalhos que usam a divisão
oficial relatam valores na mesma faixa e a mesma diferença em relação às divisões aleatórias:
- **Guendel et al. (2018):** AUC média de 0,807 na divisão oficial e de 0,841 numa divisão aleatória
  por paciente, com o mesmo modelo.
- **Baltruschat et al. (2019):** o melhor modelo chegou a 0,806 na divisão oficial e a 0,822 em
  divisões aleatórias.

O E5 (0,811) está, portanto, na faixa desses trabalhos. A diferença de Guendel et al. entre as duas
divisões é praticamente a observada aqui entre o E5 e o E1 (0,811 contra 0,841). Como o conjunto de
teste é outro, o E5 não é comparado lado a lado com os experimentos da divisão principal.

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

**Figura 8 – Grad-CAM: acertos e erros na efusão pleural**

(`results/figures/e1_baseline/gradcam/galeria_effusion.png`, na camada `denseblock4`; as galerias de
pneumonia e atelectasia vão para o apêndice)

Fonte: elaborado pelo autor.

A Figura 8 mostra a galeria da efusão pleural, escolhida pela regra fixa da Seção 3.9. Nos três
verdadeiros positivos, o mapa se concentra no hemitórax com opacidade e nas bases, onde o líquido se
acumula. Nos falsos positivos, os mapas também caem sobre regiões com alteração visível: num deles há
uma linha horizontal no hemitórax direito, com aspecto de nível hidroaéreo, e outro é um exame AP
portátil, com cateteres e opacidade na base direita. Imagens como essas sugerem que parte dos falsos
positivos pode ser erro de rótulo, e não do modelo, mas confirmar isso exigiria a leitura de um
radiologista. Nos falsos negativos, os escores são próximos de zero (0,001 a 0,003) e os pulmões não
mostram alteração evidente na imagem reduzida; com um escore tão baixo, o mapa não tem significado, e o
calor que aparece nos cantos da imagem só reflete a normalização do mapa para o intervalo [0, 1].

A galeria da pneumonia tem o mesmo padrão nos verdadeiros positivos, com o mapa sobre as opacidades
pulmonares. Dois dos três falsos positivos são exames AP portáteis, com a marcação "PORTABLE" na imagem,
mas o mapa fica sobre os pulmões, e não sobre a marcação ou os dispositivos. Um dos falsos negativos
(escore 0,001) tem opacidades extensas nos dois pulmões: ou é um erro claro do modelo, ou é um caso em
que o rótulo de pneumonia dependeu de informação que não está na imagem.

**Figura 9 – Comparação das camadas-alvo**

(`results/figures/e1_baseline/gradcam/camadas_foco.png`)

Fonte: elaborado pelo autor.

**Tabela 12 – *Pointing game* nas imagens de teste com caixa delimitadora**

| Doença | Imagens | `denseblock4`: acertos, taxa (IC95%) | `relu`: acertos, taxa (IC95%) | Centro da imagem |
|---|---|---|---|---|
| Atelectasia | 22 | 6; 27,3% (13,2–48,2%) | 5; 22,7% (10,1–43,4%) | 4,5% |
| Efusão pleural | 20 | 6; 30,0% (14,5–51,9%) | 8; 40,0% (21,9–61,3%) | 0,0% |
| Pneumonia | 20 | 3; 15,0% (5,2–36,0%) | 0; 0,0% (0,0–16,1%) | 5,0% |
| Três doenças | 62 | 15; 24,2% (15,2–36,2%) | 13; 21,0% (12,7–32,6%) | 3,2% |
| Todas as classes com caixa | 153 | 48; 31,4% (24,6–39,1%) | 42; 27,5% (21,0–35,0%) | |

Fonte: elaborado pelo autor (`results/tables/e1_baseline/pointing_game`, com as oito classes que têm
caixa). Intervalos de Wilson.

A camada `denseblock4` acertou 15 das 62 imagens das três doenças e a `relu`, 13; em todas as classes
com caixa, 48 contra 42 de 153. A diferença não é significativa (teste de McNemar exato nas mesmas
imagens, p = 0,73 nas três doenças e p = 0,15 em todas), e os mapas das duas camadas são quase iguais
(Figura 9). Pela regra da Seção 3.9, a `denseblock4` foi adotada na galeria, nas figuras com as caixas e
na interface. Na atelectasia e na efusão, as duas camadas ficaram acima do centro da imagem, com o
limite inferior do intervalo acima da taxa do centro. Na pneumonia, a `denseblock4` acertou 3 de 20, o
que não se distingue do centro (5%), e a `relu` não acertou nenhuma. Na cardiomegalia, o centro da
imagem acerta todas as 25 imagens, porque o coração fica no centro, o que mostra por que a referência
trivial é necessária. Mesmo nos melhores casos, o pico do mapa cai dentro da caixa em menos da metade
das imagens: os mapas indicam a região geral do achado, mas não o localizam com precisão.

**Figura 10 – Mapas de calor e caixas dos radiologistas (efusão pleural)**

(`results/figures/e1_baseline/gradcam/caixas_effusion.png`)

Fonte: elaborado pelo autor.

## 4.9 Validação externa no CheXpert

O modelo final foi aplicado, sem novo treino, às 202 imagens frontais (200 pacientes) da validação do
CheXpert, com os rótulos dos radiologistas (Seção 3.10). A Tabela 13 compara a AUC de cada classe com a
do teste do NIH.

**Tabela 13 – AUC no conjunto de validação do CheXpert, comparada ao teste do NIH**

| Doença | Casos (CheXpert) | AUC CheXpert (IC95%) | AUC NIH teste | Diferença |
|---|---|---|---|---|
| Atelectasia * | 75 | 0,810 (0,748–0,868) | 0,816 | −0,006 |
| Cardiomegalia | 66 | 0,836 (0,775–0,890) | 0,920 | −0,084 |
| Efusão pleural * | 64 | 0,849 (0,789–0,899) | 0,886 | −0,037 |
| Pneumotórax | 7 | 0,692 (0,455–0,963) † | 0,888 | −0,196 |
| Consolidação | 32 | 0,911 (0,867–0,949) | 0,783 | +0,129 |
| Edema | 42 | 0,841 (0,774–0,904) | 0,896 | −0,055 |
| Média (6 classes) | | 0,823 (0,777–0,876) | 0,865 | −0,041 |

Fonte: elaborado pelo autor (`results/tables/e1_baseline/validacao_externa_chexpert`). * Doenças
estudadas. † Menos de 30 casos: reportado, mas sem conclusão. A pneumonia não tem rótulos de
radiologistas disponíveis nesse conjunto (Seção 3.10).

**Figura 11 – Curvas ROC no CheXpert (validação)**

(`results/figures/e1_baseline/roc_foco_chexpert.png`)

Fonte: elaborado pelo autor.

Nas duas doenças estudadas com rótulos no CheXpert, o desempenho se manteve na mesma faixa. A
atelectasia ficou praticamente igual à do teste do NIH (0,810 contra 0,816). A efusão pleural caiu
0,037, mas a AUC do NIH (0,886) ainda está dentro do IC95% do CheXpert (0,789 a 0,899). O mesmo vale
para o edema (−0,055; IC até 0,904) e para a média das seis classes (0,823 contra 0,865; IC até 0,876).

Duas classes se afastaram do NIH além da incerteza:
- **Cardiomegalia:** caiu de 0,920 para 0,836, com o valor do NIH acima do limite superior do intervalo
  (0,890).
- **Consolidação:** subiu de 0,783 para 0,911, com o valor do NIH abaixo do limite inferior (0,867).

O pneumotórax, com 7 casos, não permite conclusão. A comparação é aproximada: o intervalo leva em conta
só a incerteza do CheXpert, e a AUC do NIH também tem a sua.

Como verificação, a análise foi repetida com preenchimento das bordas em preto em vez do
redimensionamento. Os valores ficaram próximos: média de 0,819; atelectasia, 0,804; efusão, 0,875;
cardiomegalia, 0,837; consolidação, 0,924; edema, 0,825. As conclusões não mudam.

## 4.10 Sistema de demonstração

A Figura 12 mostra a página publicada analisando uma das imagens de exemplo, um verdadeiro positivo de
efusão pleural do conjunto de teste. A frase de resumo lista as doenças acima do limiar; o mapa de calor
é o da efusão, com probabilidade estimada de 90,9% (limiar de 11,3%); e a tabela traz as 14 doenças, com
as três estudadas em destaque. A mensagem acima do resultado informa que a análise foi feita no próprio
navegador, em 0,6 s.

**Figura 12 – Interface de demonstração publicada no Hugging Face Spaces**

(`results/figures/interface_demo.png`)

Fonte: elaborado pelo autor.

As verificações da Seção 3.11 passaram com folga:

- **Modelo ONNX contra a avaliação:** nas cinco imagens de teste conferidas, os escores do modelo ONNX
  diferiram dos do arquivo de predições em no máximo 5,4 × 10⁻⁷, e os mapas de calor diferiram dos do
  Grad-CAM em Python em no máximo 8,6 × 10⁻⁶.
- **Autoteste no navegador (quatro imagens de exemplo):** a imagem vista pela rede foi idêntica pixel a
  pixel, os escores diferiram em no máximo 1,8 × 10⁻⁷ e os mapas de calor em no máximo 6,9 × 10⁻⁶. Os
  critérios eram 10⁻⁴ e 10⁻³.
- **Mesmo resultado nas duas versões:** para a mesma imagem, a página e a versão Gradio mostram a mesma
  frase de resumo e a mesma tabela.

O modelo, com 29 MB, é baixado uma vez quando a página abre. Depois disso, cada análise, com o mapa de
calor, levou entre 0,3 e 0,6 s no navegador do computador de desenvolvimento (Intel Core i5-10400F), e
a versão Gradio levou entre 0,3 e 0,4 s em CPU. A página está publicada em <https://huggingface.co/spaces/rotriguin/tcc-raio-x>, e o autoteste,
rodado no endereço público, deu os mesmos resultados que no computador de desenvolvimento.
[PREENCHER, opcional: tempo de resposta num celular.]
