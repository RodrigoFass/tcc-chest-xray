# Textos de apoio para a monografia

Rascunhos escritos a partir dos dados e das decisões do projeto, para adaptar ao estilo da
monografia. Os números vêm de `results/tables/`, de `data/splits/split_info.json` e do notebook
`notebooks/01_eda.ipynb`; se os dados forem reprocessados, confira-os de novo.

## Origem, licença e anonimização dos dados (Metodologia)

> Este trabalho utiliza o conjunto de dados público NIH ChestX-ray14, disponibilizado pelo NIH
> Clinical Center (National Institutes of Health, Estados Unidos) e descrito por Wang et al.
> (2017). O conjunto reúne 112.120 radiografias de tórax em incidência frontal (PA ou AP), de
> 30.805 pacientes, em arquivos PNG de 1024 × 1024 pixels. Cada imagem traz até 14 rótulos de
> doenças torácicas, ou a indicação de ausência de achados ("No Finding"), extraídos
> automaticamente dos laudos radiológicos por técnicas de processamento de linguagem natural, além
> da idade e do sexo do paciente e da incidência do exame. Um subconjunto de 880 imagens tem 984
> caixas delimitadoras, marcadas por radiologistas, para oito das doenças.
>
> Os dados foram obtidos da cópia disponibilizada na plataforma Kaggle (conjunto
> `nih-chest-xrays/data`, licença CC0 1.0), que contém as mesmas 112.120 imagens e os arquivos de
> metadados da distribuição do NIH (`Data_Entry_2017.csv`, `BBox_List_2017.csv`,
> `train_val_list.txt` e `test_list.txt`).
>
> Os dados são disponibilizados anonimizados: o arquivo de metadados identifica cada paciente
> apenas por um número sequencial (Patient ID) e traz, como dados demográficos, somente a idade e o
> sexo, sem nome, data de nascimento, datas de exame ou qualquer outro identificador pessoal. Como
> o trabalho utiliza exclusivamente essa base pública e anonimizada, sem contato com pacientes e
> sem acesso a dados que permitam identificá-los, ele não foi submetido a Comitê de Ética em
> Pesquisa. Conforme solicitado pelo NIH, o trabalho cita Wang et al. (2017) e reconhece o NIH
> Clinical Center como fornecedor dos dados.

Nota (não é parte do texto): se a UVV ou a banca pedirem uma base normativa para a dispensa do
Comitê de Ética, trabalhos parecidos costumam citar a Resolução CNS nº 510/2016 (art. 1º, parágrafo
único), que dispensa de registro no sistema CEP/CONEP as pesquisas com informações de acesso ou de
domínio público e com bancos de dados sem possibilidade de identificação individual. Ela foi
escrita para as Ciências Humanas e Sociais; confirme com o orientador se vale citá-la.

## Características do conjunto (Metodologia, junto com as figuras da EDA)

> Das 112.120 imagens, 53,8% não têm achados e 18,5% têm duas ou mais doenças. A prevalência das
> três doenças estudadas é de 11,9% para efusão pleural (13.317 imagens), 10,3% para atelectasia
> (11.559) e 1,3% para pneumonia (1.431). A idade mediana é de 49 anos (intervalo interquartil de
> 35 a 59 anos); 43,5% das imagens são de pacientes do sexo feminino, e 40,0% dos exames foram
> feitos na incidência AP. O número de imagens por paciente varia de 1 a 184 (mediana de 1; 56,8%
> dos pacientes têm uma única imagem). O arquivo de metadados contém 16 idades impossíveis (acima
> de 100 anos), mantidas nos dados e excluídas apenas do gráfico de idade.
>
> Três imagens não contêm anatomia visível, apenas um fundo uniforme: uma delas rotulada como
> atelectasia (00007160_002.png) e duas como sem achados (00010007_121.png e 00012249_001.png).
> Elas foram mantidas, para não alterar o conjunto em relação aos trabalhos da literatura; com 3
> em 112.120 imagens (nenhuma no conjunto de teste), o efeito é desprezível, e elas ilustram o
> ruído de rotulagem discutido por Oakden-Rayner (2020).

## Pré-processamento (Metodologia)

> As imagens foram convertidas uma única vez para tons de cinza e reduzidas de 1024 × 1024 para
> 256 × 256 pixels, com filtro de Lanczos. Das 112.120 imagens, 519 estavam no formato RGBA, mas
> com os três canais de cor iguais e o canal alfa constante, de modo que a conversão para tons de
> cinza não altera o conteúdo. Na entrada da rede, a imagem inteira é redimensionada para
> 224 × 224 pixels, sem recorte central (que cortaria os seios costofrênicos, onde aparece a
> efusão pleural), replicada nos três canais esperados pela DenseNet-121 pré-treinada no ImageNet
> e normalizada com a média e o desvio-padrão do ImageNet.

## Divisão em treino, validação e teste (Metodologia)

> A divisão em treino, validação e teste (70%, 15% e 15% das imagens) foi feita por paciente,
> para que imagens de um mesmo paciente nunca apareçam em mais de um conjunto: os 30.805 pacientes
> foram embaralhados com semente fixa (42) e alocados inteiros, em sequência, até atingir 70% e
> 85% das imagens. O resultado tem 78.486 imagens de treino (21.419 pacientes), 16.812 de validação
> (4.596 pacientes) e 16.822 de teste (4.790 pacientes). A prevalência das três doenças estudadas
> ficou a menos de 20% (em termos relativos) da prevalência geral em todos os conjuntos, sem
> necessidade de estratificação; a maior diferença foi a da pneumonia na validação (1,50% contra
> 1,28%). Os subgrupos usados na análise por subgrupo também ficaram equilibrados: de 41,5% a
> 44,3% de mulheres, cerca de 40% de exames AP e idade mediana de 48 a 49 anos nos três conjuntos.

## Pontos para a Discussão

- **Possível atalho pela incidência:** 40,0% dos exames são AP (em geral, pacientes no leito), mas
  entre as imagens com pneumonia são 56,0%, com efusão 50,5% e com atelectasia 50,4%. O modelo
  pode aprender a reconhecer o paciente acamado em vez da doença; a análise por subgrupo PA × AP
  (Fase 3) mede isso.
- **Divisão oficial do NIH (experimento E5):** o conjunto de teste oficial tem prevalência bem
  maior que o resto (efusão pleural 18,2% contra 11,9% no total; atelectasia 12,8% contra 10,3%;
  pneumonia 2,2% contra 1,3%). Isso ajuda a explicar por que resultados com divisões diferentes
  não são diretamente comparáveis (Baltruschat et al., 2019) e afeta sobretudo a AUPRC, que depende
  da prevalência.
- **Ruído nos rótulos:** além das três imagens sem anatomia, os exemplos sorteados da EDA incluem
  uma imagem rotulada "sem achados" que é um exame AP portátil com vários cabos e eletrodos
  (Oakden-Rayner, 2020).

## Referências usadas acima

- WANG, X.; PENG, Y.; LU, L.; LU, Z.; BAGHERI, M.; SUMMERS, R. M. ChestX-ray8: Hospital-scale
  Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of
  Common Thorax Diseases. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
  2017, p. 2097-2106.
- OAKDEN-RAYNER, L. Exploring Large-scale Public Medical Image Datasets. Academic Radiology,
  v. 27, n. 1, 2020.
- BALTRUSCHAT, I. M. et al. Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray
  Classification. Scientific Reports, v. 9, 2019.

Conferir volume, páginas e DOI antes de usar (seção 11 do plano).
