# Plano de implementação do TCC: detecção de doenças pulmonares em raio X de tórax

**Revisão 2.2 — 28/09/2026.** O que mudou em relação às versões anteriores está no Apêndice A.

> **Para quem for implementar:** este arquivo é a especificação do projeto. Leia inteiro antes de começar.
> Trabalhe **uma fase por vez**, na ordem. Ao terminar cada fase, confira os critérios de aceite,
> faça um commit e pare para eu revisar antes de seguir para a próxima.
> Código e comentários podem ser em inglês; mensagens para mim, README e textos da monografia em português.
> Se algo que eu disser na conversa contrariar este arquivo, vale o que eu disser; registre a mudança na seção 10.

---

## 1. Contexto e objetivo

TCC de graduação (Rodrigo Fassarella). O objetivo é um sistema que recebe uma radiografia de tórax
frontal e devolve a **probabilidade de cada doença pulmonar**, com foco em três patologias:
**Pneumonia, Atelectasia (Atelectasis) e Efusão pleural (Effusion)**.

A hipótese a demonstrar: uma **DenseNet-121** pré-treinada no ImageNet e ajustada (fine-tuning) no
**NIH ChestX-ray14** atinge AUC-ROC competitiva com a literatura (CheXNet, Rajpurkar et al. 2017;
Wang et al. 2017). O sistema também mostra **mapas de calor Grad-CAM** para explicar cada predição.

O trabalho é acadêmico: o sistema é uma prova de conceito de apoio ao diagnóstico, **não** um
dispositivo médico. Isso deve aparecer no README e na interface.

### Expectativa realista de resultado

Os números do CheXNet vêm de uma divisão aleatória própria dos autores, e Baltruschat et al. (2019)
mostraram que o desempenho no ChestX-ray14 muda bastante conforme a divisão usada. A meta, portanto,
não é "bater o CheXNet", e sim ficar na mesma faixa e explicar as diferenças com rigor.
Pneumonia é a classe mais difícil em todos os trabalhos (menor AUC e só ~1,3% de prevalência):
um resultado abaixo das outras duas é esperado, não um erro.

### Números de referência do dataset (para conferir o parsing dos rótulos)

- 112.120 imagens de 30.805 pacientes; 14 rótulos + "No Finding" (~60.361 imagens sem achado).
- Imagens positivas: Atelectasis ~11.559 (10,3%), Effusion ~13.317 (11,9%), Pneumonia ~1.431 (1,3%).
  A classe mais rara é Hernia (~227).

Se as contagens da Fase 1 derem muito diferentes disso, o parsing está errado.

## 2. Decisões já tomadas (vêm do documento do TCC 1, não mudar sem me perguntar)

| Item | Decisão |
|---|---|
| Linguagem | Python 3.10+ |
| Framework | PyTorch + torchvision |
| Modelo | DenseNet-121, pesos ImageNet, última camada trocada por `Linear(1024, N)` multi-label (sigmoid) |
| Dataset principal | NIH ChestX-ray14 (112.120 imagens, 14 rótulos + "No Finding") |
| Dataset complementar | CheXpert, só como validação externa (Fase 7, no escopo desde 28/09/2026) |
| Divisão | 70% treino / 15% validação / 15% teste |
| Pré-processamento | resize 224×224, normalização (média/desvio do ImageNet), imagem cinza replicada em 3 canais |
| Data augmentation (só treino) | rotação pequena, espelhamento horizontal, ajuste de brilho |
| Otimizador | Adam, lr inicial 1e-4, com redução automática (ReduceLROnPlateau) |
| Perda | Binary Cross-Entropy (`BCEWithLogitsLoss`) |
| Parada | Early stopping pela métrica de validação |
| Métricas | AUC-ROC por doença, acurácia, sensibilidade, especificidade |
| Interpretabilidade | Grad-CAM via biblioteca `pytorch-grad-cam` |
| Análise | matplotlib, seaborn, scikit-learn |
| Treino | Google Colab Pro (GPU T4/A100) |
| Versionamento | Git + GitHub |

**Esclarecimentos desta revisão (não mudam as decisões acima):**

- *Pré-processamento:* na validação e no teste, o "resize 224×224" é aplicado à imagem inteira,
  sem corte central (ver 3.10).
- *Métricas:* a acurácia continua sendo reportada, mas com ressalva: com 1,3% de prevalência, um
  modelo que responde "não" para tudo tem 98,7% de acurácia em Pneumonia. Por isso entra a AUPRC
  como métrica complementar (ver 3.9).
- *Treino:* o TCC 1 previa Colab Pro, mas o ambiente de treino está **em aberto** (ver 3.13).
  O código roda igual em qualquer um dos ambientes considerados; só mudam os caminhos no YAML.

## 3. Decisões padrão que este plano adota (pode mudar se eu pedir)

**3.1 Divisão por paciente, não por imagem.** O NIH tem várias imagens do mesmo paciente
(`Patient ID`). Dividir por imagem vaza informação entre treino e teste e infla a AUC.
A divisão 70/15/15 é feita sobre os pacientes, com seed fixa, e salva em CSV.
O `split.py` também sabe gerar a divisão oficial do NIH (`train_val_list.txt` / `test_list.txt`,
com a validação tirada de dentro do train_val por paciente), escolhida por
`split.strategy: patient_random | official` no YAML. O padrão é `patient_random`; o split oficial
é usado só no experimento E5 e fica em pasta própria, `data/splits/official`. Cada pasta de split
tem um `split_info.json` com as configurações, e o `split.py` se recusa a sobrescrever um split feito
com outras configurações.

**3.2 Treinar nas 14 classes, destacar as 3 do TCC.** Igual ao CheXNet: o modelo prevê as 14
doenças, todas as AUCs são reportadas, e a análise da monografia foca em Pneumonia, Atelectasis e
Effusion. Isso facilita comparar com a literatura. **Confirmado em 28/09/2026** (pergunta 1 da
seção 9). Se um dia for preciso treinar só nas 3 classes, basta mudar a lista de classes no YAML.

**3.3 Desbalanceamento:** `pos_weight` por classe no `BCEWithLogitsLoss` (negativos/positivos do
treino), com um experimento de comparação sem pesos. Observação para a monografia: a AUC depende só
da ordem dos escores, então `pos_weight` costuma mexer pouco nela; o efeito principal é empurrar os
escores para cima (pior calibração) e mudar a sensibilidade no limiar. Por isso E1 × E2 são
comparados também em AUPRC e calibração. O CheXNet, na versão de 14 classes, usou BCE sem pesos.
Hernia fica com `pos_weight` perto de 500; o YAML tem `pos_weight_max` (padrão: sem teto) para
limitar se a loss ficar instável.

**3.4 Imagens pré-processadas e empacotadas.** O NIH original tem ~42 GB (PNG 1024×1024). Na
Fase 1, as imagens são convertidas uma única vez para cinza 256×256 e empacotadas num único
`nih256.tar` (estimativa: 3–6 GB), que vai para onde o treino acontecer. O `preprocess.py` lê
tanto do zip do Kaggle (sem extrair, útil quando o disco é pequeno) quanto de uma pasta já
extraída (caso dos notebooks do Kaggle, onde o dataset já vem montado). O tar existe porque
armazenamentos em nuvem, como o Google Drive, são muito lentos com 112 mil arquivos pequenos; um
arquivo só é copiado em minutos e extraído no disco local. O pré-processamento só usa CPU: pode
rodar no seu computador, no Colab ou no Kaggle, sem gastar GPU. O treino usa RandomResizedCrop
para 224.

**3.5 Código em pacote Python + notebooks finos.** Toda a lógica fica em `src/chestxray`,
instalada com `pip install -e .` (via `pyproject.toml`), para `python -m chestxray.train` funcionar
em qualquer lugar. Os notebooks (Colab ou Kaggle) só clonam o repositório, instalam e chamam os scripts.
Assim dá para testar tudo localmente em CPU com um subconjunto pequeno. Em ambientes
gerenciados (Colab, Kaggle), **não reinstalar torch/torchvision** (usar as versões que já vêm,
que casam com o CUDA da máquina); as demais dependências têm versão fixada, e as versões usadas
em cada treino ficam registradas. Nenhum caminho fixo no código (`/content`, `/kaggle`, etc.):
tudo vem de `configs/paths/<ambiente>.yaml` (`data_dir`, `runs_dir`, `checkpoint_dir`).

**3.6 Configuração em YAML e seed fixa.** `configs/base.yaml` tem tudo; cada experimento herda
dela e só sobrescreve o que muda. Seed fixa em random, numpy e torch, com
`cudnn.deterministic=True` e `cudnn.benchmark=False`. **Não** ligar
`torch.use_deterministic_algorithms(True)` em modo estrito: o backward do pooling adaptativo da
DenseNet na GPU não tem versão determinística e dá erro (se quiser, usar `warn_only=True`).
Mesmo com seed, dois treinos na GPU podem variar um pouco; por isso a 3.9 prevê repetir a
configuração final com 3 seeds.

**3.7 Limiar de decisão** para sensibilidade/especificidade escolhido na validação (índice de
Youden) e aplicado no teste. AUC não depende de limiar.

**3.8 Interface de demonstração com Gradio.** Hospedagem principal no **Hugging Face Spaces**
(CPU gratuita, link permanente, não depende de nenhuma máquina sua estar ligada no dia da
defesa). Backup: a mesma interface rodando no seu computador (CPU basta) ou num notebook com
`share=True`. Plano C: vídeo curto gravado da demo funcionando.

**3.9 Avaliação estatística.**
- IC95% por **bootstrap por paciente** (reamostra pacientes, não imagens): imagens do mesmo
  paciente são correlacionadas, e o bootstrap por imagem dá intervalos estreitos demais.
  1000 reamostragens, seed fixa.
- Diferença entre dois experimentos: **bootstrap pareado** (as mesmas reamostras para os dois
  modelos), com IC95% da diferença de AUC. Se o intervalo contém zero, a monografia diz que não
  houve diferença significativa.
- AUPRC (average precision) sempre ao lado da prevalência da classe: em classes raras ela mostra o
  que a AUC esconde.
- Configuração final repetida com 3 seeds (média ± desvio), se o orçamento de GPU deixar.

**3.10 Transforms.** Validação/teste: `Resize((224, 224))` na imagem inteira. Nada de
`CenterCrop`: cortar 256→224 tira ~6% de cada borda, e os seios costofrênicos (onde aparece o
derrame pleural) ficam perto dos cantos inferiores. Treino:
`RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05))` + o resto do augmentation da
seção 2.

**3.11 Modelo com ReLU final não in-place.** A DenseNet do torchvision faz
`F.relu(features, inplace=True)` no `forward`, o que pode atrapalhar os hooks do Grad-CAM. O modelo
é envolvido numa classe própria com `nn.ReLU(inplace=False)` como módulo, que vira a camada-alvo
natural do Grad-CAM (Fase 4).

**3.12 Prioridades.** Cada entrega tem prioridade (essencial / importante / complementar),
definida na seção 6.1. Os complementares estão no escopo e são feitos por último; se o prazo apertar,
são os primeiros a ser reconsiderados (a decisão de cortar é do Rodrigo).

**3.13 Ambiente de treino (em aberto; decidir antes da Fase 2).** As fases 0 e 1 não precisam de
GPU: rodam no seu computador, em CPU, com a amostra do Kaggle. O código é o mesmo em qualquer
ambiente; só mudam o YAML de caminhos e o notebook ou comando que chama o treino.

| Opção | A favor | Contra |
|---|---|---|
| Google Colab Pro | Previsto no TCC 1; GPUs boas; integra com o Drive | Pago; sessões caem; precisa copiar o tar do Drive a cada sessão |
| Kaggle Notebooks | GPU gratuita com cota semanal; o NIH já está no Kaggle e basta anexá-lo ao notebook (sem baixar 42 GB) | Cota semanal e limite de horas por sessão (conferir os valores atuais); resultados precisam ser salvos como output ou dataset |
| GPU NVIDIA própria | Sem cota nem queda de sessão; dados no disco local | Precisa de placa NVIDIA (CUDA); pouca memória de vídeo obriga batch menor; o PC fica ocupado durante o treino |

Como decidir: com a Fase 2 pronta, rodar 1 época do E1 nos ambientes disponíveis, anotar o tempo
e multiplicar pelos ~5 treinos essenciais e importantes da seção 6.1 (até 30 épocas cada; o early
stopping costuma parar antes). Escolher o que couber no prazo e no bolso e registrar na seção 10.
Placas AMD não entram como opção padrão, porque o PyTorch fora de CUDA dá bem mais trabalho para
configurar.

## 4. Estrutura do repositório

```
tcc-chest-xray/
├── README.md                    # como rodar, resultados, aviso de uso não clínico, citação do NIH
├── PLANO_TCC.md                 # este arquivo
├── pyproject.toml               # pacote instalável (pip install -e .)
├── requirements.txt             # dependências com versão fixada (uso local)
├── .github/workflows/tests.yml  # (opcional) pytest em CPU a cada push
├── configs/
│   ├── base.yaml                # caminhos, classes, hiperparâmetros
│   ├── debug.yaml               # herda base: 200 imagens, 1 época, CPU
│   ├── paths/                   # local.yaml, colab.yaml, kaggle.yaml (só caminhos)
│   └── experiments/             # e1_baseline.yaml, e2_posweight.yaml, e3_noaug.yaml, ...
├── data/                        # NÃO versionado; só data/splits/ é versionado
│   └── splits/                  # train.csv, val.csv, test.csv, split_info.json; official/ (E5)
├── docs/
│   └── textos_monografia.md     # rascunhos de texto para a monografia (origem dos dados etc.)
├── src/chestxray/
│   ├── __init__.py
│   ├── config.py                # carrega YAML com herança (base → experimento)
│   ├── data/
│   │   ├── download.py          # Kaggle (completo ou amostra)
│   │   ├── preprocess.py        # lê do zip, 256×256 cinza, empacota em tar
│   │   ├── split.py             # 70/15/15 por paciente (ou split oficial)
│   │   └── dataset.py           # Dataset PyTorch + transforms
│   ├── models/densenet.py       # DenseNet-121 multi-label, ReLU final não in-place
│   ├── train.py                 # treino, AMP, checkpoints, retomada, early stopping
│   ├── inference.py             # checkpoint → predições (usado por evaluate, gradcam e app)
│   ├── evaluate.py              # métricas a partir das predições, figuras, tabelas
│   ├── gradcam.py               # mapas de calor
│   └── utils.py                 # seed, logging, device, registro de versões
├── app/
│   ├── app.py                   # interface Gradio
│   └── requirements.txt         # dependências do Hugging Face Spaces (torch CPU)
├── notebooks/
│   ├── 01_eda.ipynb               # análise exploratória (figuras p/ monografia)
│   ├── 02_train_<ambiente>.ipynb  # só o do ambiente escolhido (Colab ou Kaggle); GPU própria usa linha de comando
│   └── 03_results.ipynb           # avaliação, Grad-CAM, figuras finais
├── results/
│   ├── runs/<experimento>/      # log.csv, config resolvida, versões, predições (sem .pt)
│   ├── figures/
│   └── tables/
└── tests/                       # pytest rápido com dados sintéticos
```

Checkpoints (`.pt`) ficam fora do repositório, no caminho definido em `configs/paths/`.

## 5. Fases

Cada fase tem entregáveis e critérios de aceite. Só avance quando todos os critérios estiverem OK.

### Fase 0: Esqueleto do projeto

**Fazer**
- Criar a estrutura acima, `pyproject.toml` e `requirements.txt` (torch, torchvision, pandas,
  numpy, scikit-learn, matplotlib, seaborn, pillow, pyyaml, tqdm, grad-cam, gradio, pytest,
  kaggle). Versões fixadas, exceto torch/torchvision (só versão mínima).
- `.gitignore`: `data/*` (exceto `data/splits/`), `checkpoints/`, `*.pt`, `*.tar`, `.venv/`,
  `kaggle.json`, `.env`.
- README inicial com o aviso de uso não clínico.
- `utils.py`: `set_seed()` conforme 3.6, `get_device()`, logging e uma função que registra
  versões de Python, torch, CUDA e o nome da GPU.
- `config.py` lendo YAML com herança (`extends: base.yaml`).
- (Opcional) GitHub Actions rodando `pytest -m "not slow"` em CPU.

**Critérios de aceite**
- `pip install -e .` funciona num ambiente limpo.
- `pytest` roda (mesmo que com um teste trivial).
- Primeiro commit feito.

### Fase 1: Dados

**Fazer**
- `download.py`: Kaggle CLI, dataset `nih-chest-xrays/data`, baixado **sem** `--unzip`; com
  `--sample`, baixa `nih-chest-xrays/sample` (~5.600 imagens) para desenvolvimento. Credenciais
  por variável de ambiente (no Colab, pelos *Secrets*), nunca em arquivo versionado. O README
  explica como gerar o token no Kaggle. Num notebook do Kaggle não é preciso baixar nada, basta
  anexar o dataset. Plano B se faltar disco: baixar os 12 pacotes oficiais do NIH e processar um
  por vez, apagando cada um depois.
- `preprocess.py`:
  - lê cada PNG de dentro do zip (`zipfile`) ou de uma pasta já extraída (`--source zip|dir`), converte com `.convert("L")` (algumas imagens do NIH
    vêm com 4 canais), redimensiona para 256×256 com filtro de alta qualidade e salva em PNG;
  - paralelo (multiprocessing), retomável (pula o que já existe), com manifesto CSV
    (nome, modo e tamanho originais, status);
  - no fim, empacota em `nih256.tar` junto com os arquivos originais de metadados
    (`Data_Entry_2017.csv`, `BBox_List_2017.csv`, `train_val_list.txt`, `test_list.txt`), pronto
    para ir para o ambiente de treino;
  - roda só em CPU, onde for mais cômodo (seu computador, Colab ou Kaggle); o README documenta o
    jeito usado.
- `split.py`:
  - lê `Data_Entry_2017.csv` e transforma `Finding Labels` (separados por `|`) em 14 colunas
    binárias com os nomes exatos do CSV (atenção a `Pleural_Thickening`); "No Finding" = tudo zero;
  - `patient_random`: 70/15/15 sobre os `Patient ID` únicos, seed fixa;
  - `official`: teste = `test_list.txt`; treino/validação = `train_val_list.txt` dividido por paciente;
  - aceita também o CSV da amostra do Kaggle (`sample_labels.csv`; conferir as colunas);
  - salva `data/splits/{train,val,test}.csv` com imagem, paciente, 14 rótulos, idade, sexo e
    `View Position`, e `results/tables/prevalencia_splits.csv`.
- `dataset.py`: `Dataset` que lê o CSV e devolve `(tensor 3×224×224, vetor de 14 rótulos float, id da imagem)`.
  - Treino: `RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05))`, `RandomHorizontalFlip`,
    `RandomRotation(10)`, `ColorJitter(brightness=0.2)`, replicar em 3 canais, `ToTensor`,
    `Normalize` (ImageNet).
  - Validação/teste: `Resize((224, 224))`, replicar em 3 canais, `ToTensor`, `Normalize`.
- `notebooks/01_eda.ipynb`: contagem e prevalência por classe; co-ocorrência de doenças (heatmap);
  imagens por paciente; distribuição de idade e sexo (o CSV tem algumas idades impossíveis, acima
  de 100 anos: filtrar só nos gráficos e citar na monografia); proporção PA × AP por classe;
  exemplos de imagens por classe. Figuras em `results/figures/eda_*.png`.

**Critérios de aceite**
- Contagens batem com os números de referência da seção 1.
- Teste automático provando que **nenhum Patient ID aparece em mais de um split**.
- Proporções dos splits entre 69–71% / 14–16% / 14–16% das imagens.
- Prevalência das 3 classes do TCC com diferença relativa de no máximo 20% entre os splits
  (ex.: Pneumonia entre ~1,0% e ~1,5%). Se falhar, trocar para estratificação multirrótulo por
  paciente (`iterative-stratification`) e registrar na seção 10. **Nunca** procurar uma seed
  "que passe".
- `nih256.tar` com 112.120 imagens; um teste abre 100 imagens aleatórias sem erro.
- DataLoader devolve batch com shape `(B, 3, 224, 224)` e rótulos `(B, 14)` em float.
- Figuras da EDA geradas.

### Fase 2: Modelo e treino

**Fazer**
- `models/densenet.py`: `torchvision.models.densenet121(weights=IMAGENET1K_V1)` (ou `weights=None`
  quando `pretrained: false`, para o E4), dentro de uma classe própria:
  `features → nn.ReLU(inplace=False) → adaptive_avg_pool → flatten → Linear(1024, num_classes)`.
  Saída em logits (sigmoid só na inferência). Opção `memory_efficient` para GPUs com pouca memória.
- `train.py`:
  - `BCEWithLogitsLoss(pos_weight=...)` calculado do treino, liga/desliga e `pos_weight_max` no YAML;
  - Adam lr=1e-4, betas padrão; `ReduceLROnPlateau(mode="max", factor=0.1, patience=1)` na AUC
    média de validação;
  - early stopping com paciência 5 na AUC média de validação; `max_epochs: 30`;
  - AUC de validação por classe: se o conjunto não tiver positivo ou negativo de uma classe, ela
    vira NaN e a média usa `nanmean` (isso acontece com o `debug.yaml`);
  - AMP quando houver GPU (fp16 com GradScaler; bf16 se a GPU suportar); batch configurável
    (16, 32 ou 64); `num_workers` configurável; `pin_memory=True`;
  - `last.pt` a cada época com modelo, otimizador, scheduler, scaler, época, melhor métrica,
    contador do early stopping e estados de RNG; `best.pt` quando a AUC de validação melhora.
    Salvar de forma **atômica** (grava `.tmp` e renomeia), para o checkpoint não ficar
    corrompido se a sessão ou o PC cair no meio da escrita;
  - retomada automática se existir `last.pt` no diretório do experimento;
  - por experimento, `results/runs/<nome>/` com `log.csv` (época, loss treino/val, AUC por classe,
    AUC média, lr, tempo da época), config resolvida, hash do commit e versões das bibliotecas.
- `configs/debug.yaml`: herda de `base.yaml`; 200 imagens (da amostra do Kaggle ou sintéticas),
  1 época, CPU, batch 8, `num_workers: 0`.
- `configs/experiments/`: um YAML por experimento da Fase 3.
- Execução no ambiente escolhido (3.13): um notebook fino (`02_train_colab.ipynb` ou
  `02_train_kaggle.ipynb`) ou, na GPU própria, só a linha de comando. Em todos os casos:
  - colocar os dados no disco local;
  - `pip install -e .` (sem reinstalar torch no Colab/Kaggle);
  - conferir a GPU com `nvidia-smi`;
  - rodar `python -m chestxray.train --config configs/experiments/e1_baseline.yaml --paths configs/paths/<ambiente>.yaml`,
    com checkpoints num lugar que sobreviva ao fim da sessão (Drive no Colab, output no Kaggle,
    disco no PC).
- Orçamento de GPU: anotar o tempo da primeira época e recalcular quantos experimentos cabem. Se a
  utilização da GPU ficar baixa no `nvidia-smi`, o gargalo é a CPU carregando imagens: ajustar
  `num_workers` antes de trocar de GPU.

**Critérios de aceite**
- `python -m chestxray.train --config configs/debug.yaml` roda do início ao fim em CPU sem erro e
  gera checkpoint e `log.csv`.
- Teste de retomada: 2 épocas seguidas e 1 época + interrupção + retomada terminam com o mesmo
  número de épocas no log, sem linha duplicada.
- Teste de sanidade (marcado `slow`): o modelo consegue **overfit** em 16–32 imagens (loss de
  treino cai perto de zero), provando que o pipeline aprende.
- (Feito por mim, no ambiente escolhido) o E1 termina e `best.pt` fica salvo fora da sessão.

### Fase 3: Avaliação e experimentos

**Fazer**
- `inference.py`: carrega um checkpoint e salva as predições (logits e escores sigmoid, por imagem)
  de validação e teste em `results/runs/<exp>/preds_{val,test}.csv`. Inferência sempre em fp32
  (sem AMP). É a única função de pré-processamento + predição do projeto; `gradcam.py` e o app
  usam a mesma.
- `evaluate.py`, que roda em CPU a partir dos CSVs de predição:
  - AUC-ROC por classe e média (sklearn `roc_auc_score`), com IC95% por bootstrap por paciente (3.9);
  - AUPRC por classe, sempre com a prevalência ao lado;
  - limiar por classe via Youden na validação; no teste, acurácia, sensibilidade, especificidade
    e F1 por classe;
  - curvas ROC das 3 classes do TCC (uma figura) e das 14 (grade); curvas precisão-revocação das 3;
  - matriz de confusão das 3 classes do TCC;
  - curva de calibração (reliability diagram) e Brier score das 3 classes;
  - recalibração dos escores por *Platt scaling*: por classe, uma regressão logística de uma
    variável sobre o logit, ajustada **só na validação** e aplicada no teste. Os parâmetros ficam em
    `results/runs/<exp>/calibration.json`. Curva de calibração e Brier score antes e depois,
    no teste. A transformação é crescente: não muda a AUC nem a ordem dos escores, e o limiar de
    Youden continua sendo o mesmo ponto de corte, só convertido para a escala calibrada. Não usar
    regressão isotônica: com poucas dezenas de positivos nas classes raras (Hernia), ela
    sobreajusta. Objetivo: permitir que a interface mostre o valor como probabilidade (Fase 5).
    Limitação a citar: a probabilidade calibrada vale para a prevalência do NIH; em outra
    população, com outra prevalência, ela deixa de valer;
  - análise por subgrupo nas 3 classes: sexo, faixa etária (<40, 40–60, >60 anos) e posição
    PA × AP, com IC95%. Mostra se o modelo funciona igual para todos e expõe um possível atalho:
    exames AP costumam ser de pacientes acamados, mais graves;
  - tabela comparativa com a literatura em `results/tables/comparacao_literatura.{csv,md,tex}`.
- Valores de referência, conferidos na Tabela 2 do CheXNet (arXiv:1711.05225v3) em 27/09/2026.
  A coluna "Wang et al." é a que o próprio CheXNet reporta; citar assim na monografia.

  | Doença | Wang et al. (2017) | CheXNet (2017) |
  |---|---|---|
  | **Atelectasis** | **0,716** | **0,8094** |
  | Cardiomegaly | 0,807 | 0,9248 |
  | **Effusion** | **0,784** | **0,8638** |
  | Infiltration | 0,609 | 0,7345 |
  | Mass | 0,706 | 0,8676 |
  | Nodule | 0,671 | 0,7802 |
  | **Pneumonia** | **0,633** | **0,7680** |
  | Pneumothorax | 0,806 | 0,8887 |
  | Consolidation | 0,708 | 0,7901 |
  | Edema | 0,835 | 0,8878 |
  | Emphysema | 0,815 | 0,9371 |
  | Fibrosis | 0,769 | 0,8047 |
  | Pleural Thickening | 0,708 | 0,8062 |
  | Hernia | 0,767 | 0,9164 |

  Na monografia, deixar explícito que a comparação é aproximada. O CheXNet de 14 classes usou
  divisão aleatória própria 70/10/20 sem sobreposição de pacientes, BCE sem pesos, Adam com
  lr 1e-3, batch 16, só espelhamento horizontal como augmentation e escolha do modelo pela menor
  loss de validação. E o desempenho no ChestX-ray14 varia bastante com a divisão usada
  (Baltruschat et al., 2019).

- Experimentos (cada um é um YAML em `configs/experiments/` e uma linha na tabela final):

  | ID | Experimento | Prioridade |
  |---|---|---|
  | E1 | Baseline: DenseNet-121 ImageNet, BCE sem `pos_weight` | essencial |
  | E2 | E1 + `pos_weight` | essencial |
  | E3 | Melhor entre E1 e E2 (pela validação), sem data augmentation | importante |
  | E-seeds | Melhor entre E1 e E2 repetida com mais 2 seeds (3 no total) | importante |
  | E4 | Melhor entre E1 e E2, treinada do zero (sem ImageNet) | complementar |
  | E5 | Melhor entre E1 e E2, no split oficial do NIH | complementar |

  - E4 converge devagar; com o mesmo `max_epochs` e early stopping, o resultado mostra o custo de
    não usar transfer learning com o mesmo orçamento, não o limite da arquitetura. Dizer isso na
    monografia.
  - E5 tem outro conjunto de teste: vai numa tabela separada, comparada com trabalhos que usam o
    split oficial, e não lado a lado com E1–E4. O YAML do E5 usa `split.strategy: official` e
    `paths.splits_dir: data/splits/official`. Visto na revisão da Fase 1: o split oficial não repete
    pacientes (71.255 / 15.269 / 25.596 imagens), mas o teste oficial é bem mais "doente" que o resto
    (Efusão 18,2% contra 11,9% no total; Atelectasia 12,8% contra 10,3%; Pneumonia 2,2% contra 1,3%).
    Isso reprova o critério de 20% da Fase 1 (que vale só para o split principal) e mexe muito na
    AUPRC, que depende da prevalência: no E5, comparar AUPRC sempre com a prevalência ao lado.
- Tabela final: por experimento, AUC média (14 classes), AUC e AUPRC das 3 classes com IC95%,
  épocas até o early stopping e tempo de treino. Mais uma tabela de diferenças pareadas
  (E2−E1, E3−melhor, E4−melhor) com IC95%.

**Critérios de aceite**
- `python -m chestxray.evaluate --run results/runs/<exp> --split test` gera
  `results/runs/<exp>/metrics_test.json`, as tabelas e as figuras em `results/figures/`.
- O modelo final é escolhido **só pela validação**, e a escolha fica registrada na seção 10 antes
  de rodar o teste. Nenhum hiperparâmetro é ajustado olhando o teste.
- Tabela final com todos os experimentos lado a lado e tabela de diferenças pareadas.
- `calibration.json` ajustado só com a validação; curvas de calibração e Brier score no teste,
  antes e depois da recalibração.

### Fase 4: Interpretabilidade (Grad-CAM)

**Fazer**
- `gradcam.py` com `pytorch_grad_cam.GradCAM`. Camada-alvo padrão: a ReLU final (3.11). Nessa
  arquitetura (pooling global seguido de camada linear), Grad-CAM na última camada coincide com o
  CAM usado pelo CheXNet (Zhou et al., 2016), o que ajuda a amarrar a fundamentação teórica. Testar
  também `denseblock4` e registrar qual gera mapas mais nítidos.
- Função `explain(image_path, class_name) -> (escore, imagem com heatmap sobreposto)`, usando
  `inference.py`.
- Galeria para a monografia: para cada uma das 3 doenças, verdadeiros positivos, falsos positivos
  e falsos negativos (limiar da validação), com heatmap. **Critério de escolha fixo e declarado**
  (ex.: os 3 de maior escore em VP e FP, os 3 de menor escore em FN), para as figuras não parecerem
  escolhidas a dedo. Salvar em `results/figures/gradcam/`.
- Bounding boxes: `BBox_List_2017.csv` (~1.000 caixas, incluindo as 3 doenças do TCC, em
  coordenadas de 1024×1024; escalar para 224). Usar **só imagens que caíram no teste**. Desenhar a
  caixa do radiologista junto com o heatmap.
- Atenção: no `BBox_List_2017.csv` a infiltração se chama `Infiltrate` (não `Infiltration`). São 984
  caixas em 880 imagens; 153 caixas caíram no nosso teste, 22 de Atelectasia, 20 de Efusão e 20 de
  Pneumonia.
- (Complementar) métrica simples de localização, o *pointing game*: porcentagem dos casos em que
  o ponto máximo do heatmap cai dentro da caixa, por classe.

**Critérios de aceite**
- Heatmaps gerados sem erro para imagens de teste.
- Galeria com pelo menos 3 exemplos por categoria (VP/FP/FN) para cada doença do TCC, com o
  critério de escolha escrito no README.
- Figuras com as bounding boxes para as imagens de teste que têm caixa.

### Fase 5: Sistema de demonstração

**Fazer**
- `app/app.py` com Gradio, usando `inference.py` e `gradcam.py`:
  - upload de PNG/JPG (DICOM fica fora do escopo);
  - frase de resumo acima da tabela. Com achados: "Achados acima do limiar: Efusão,
    Atelectasia". Sem achados: "Nenhum achado acima do limiar entre as 14 doenças avaliadas".
    **Nunca** "Normal", "Saudável" ou "Sem doença": o modelo só conhece 14 doenças (não vê
    tuberculose, fraturas etc.) e todo limiar deixa passar falsos negativos;
  - tabela com o escore das 14 doenças, as 3 do TCC em destaque, cada uma com o limiar da
    validação e a indicação "acima/abaixo do limiar";
  - heatmap Grad-CAM da doença escolhida;
  - exemplos clicáveis com imagens do conjunto de teste;
  - aviso fixo e visível: "Protótipo acadêmico. Não usar para diagnóstico.", mais uma nota de que
    imagens muito diferentes das do NIH (foto de tela, criança, incidência lateral) geram
    resultados sem sentido.
- Mostrar o escore recalibrado da Fase 3. Se, no teste, a curva de calibração depois do Platt
  scaling ficar próxima da diagonal nas 3 classes do TCC, a interface chama o valor de
  **"probabilidade estimada"**, com a nota de que ela vale para a população do NIH; a decisão vai
  para a seção 10. Caso contrário, ou se a recalibração não for feita, chamar de
  **"escore do modelo"**, não "probabilidade": com `pos_weight`, o valor bruto não é uma
  probabilidade calibrada.
- Publicar no Hugging Face Spaces (Gradio, CPU gratuita), com só o `state_dict` do modelo
  (~30 MB) num repositório de modelo do Hugging Face ou via Git LFS no Space. Spaces gratuitos
  "dormem" sem uso: abrir o link alguns minutos antes da defesa.
- Backup: a mesma interface rodando no seu computador (CPU basta) ou num notebook com `share=True`.
- Gravar um vídeo curto da demo (plano C).

**Critérios de aceite**
- Com `best.pt`, 3 imagens do teste dão os mesmos escores brutos do `preds_test.csv` (tolerância
  1e-4), e os valores exibidos batem com o `calibration.json` aplicado a esses escores.
- A frase de resumo aparece nos dois casos (com e sem achados acima do limiar).
- Resposta em até ~5 s por imagem em CPU, com Grad-CAM.
- Link do Space abrindo numa janela anônima.

### Fase 6: Material para a monografia

**Fazer**
- `notebooks/03_results.ipynb` que reúne todas as tabelas e figuras finais.
- `results/RESUMO_RESULTADOS.md` em português com: configuração final; AUCs e AUPRCs com IC95%;
  comparação com a literatura e por que ela é aproximada; efeito de cada experimento com as
  diferenças pareadas; subgrupos; calibração; observações do Grad-CAM; e limitações:
  - rótulos do NIH extraídos automaticamente de laudos, com ruído relevante, principalmente em
    Pneumonia (Oakden-Rayner, 2020);
  - divisão diferente da dos artigos;
  - um único dataset, de um único hospital; modelos de raio X de tórax costumam perder desempenho
    em dados de outros hospitais (Zech et al., 2018);
  - possível atalho por posição AP/PA e por dispositivos visíveis na imagem;
  - resolução 224×224 e ausência de dados clínicos.
- README final com passo a passo de reprodução, uma seção "uso pretendido e limitações" e a
  citação pedida pelo NIH (artigo de Wang et al., 2017, e reconhecimento do NIH Clinical Center
  como fornecedor dos dados).
- Figuras em PNG 300 dpi **e** PDF vetorial, com títulos e eixos em português e **vírgula
  decimal**. Tabelas em CSV, Markdown e LaTeX. Assim a escolha entre LaTeX e Word não trava nada.

**Critérios de aceite**
- Todas as figuras e tabelas nos formatos acima, em português.
- Quem clonar o repositório reproduz as tabelas a partir dos CSVs de predição seguindo o README,
  sem GPU; com GPU, reproduz também o treino.

### Fase 7 (complementar): validação externa no CheXpert

No escopo desde 28/09/2026; feita depois das fases 0–6.
- Usar apenas o **conjunto de validação do CheXpert** (rotulado por consenso de radiologistas, sem
  rótulos incertos) para testar o modelo treinado no NIH, sem treinar nada no CheXpert. O download
  exige cadastro na Stanford; eu faço.
- Mapear as classes em comum (Atelectasis, Pleural Effusion → Effusion, Pneumonia e as demais que
  existirem nos dois), só imagens frontais.
- Reportar AUC com IC95% e comparar com o teste do NIH. O conjunto é pequeno (poucas centenas de
  imagens) e tem pouquíssimos casos de Pneumonia: reportar, mas não tirar conclusão sobre essa classe.
- Treinar no CheXpert com as políticas U-Ones/U-Zeros/U-Ignore fica como trabalho futuro.

## 6. Prioridades e cronograma

### 6.1 Prioridades

| Nível | Entregas |
|---|---|
| **Essencial** (sem isso não há defesa) | Fases 0–2; E1 e E2; Fase 3 (métricas, IC, comparação com a literatura); galeria Grad-CAM; demo Gradio funcionando |
| **Importante** | E3; 3 seeds da configuração final; subgrupos; calibração (curva e recalibração por Platt scaling); bounding boxes; Hugging Face Spaces |
| **Complementar** (no escopo desde 28/09/2026; feito por último) | E4; E5; pointing game; Fase 7 |

### 6.2 Cronograma relativo

A data de entrega ainda não está definida, então o cronograma é contado em semanas a partir do
início. A semana 1 é a semana em que o trabalho começar. A escrita da Metodologia começa junto,
porque não depende de resultado.

| Semana | Técnico | Escrita | Precisa de GPU? |
|---|---|---|---|
| 1 | Fase 0; Fase 1 com a amostra do Kaggle (~5.600 imagens) | Metodologia: dataset, divisão, modelo | Não |
| 2 | Fase 1 completa (download, pré-processamento, split, EDA); escolher o ambiente de treino (3.13) | Metodologia: figuras da EDA | Não |
| 3 | Fase 2; E1 e E2 | Metodologia: treino e métricas | Sim |
| 4 | Fase 3; E3 e seeds | — | Sim |
| 5 | Fases 4 e 5 | Resultados: tabelas | Pouco |
| 6 | Fase 6; Hugging Face Spaces | Resultados e discussão | Não |
| 7 | Complementares: E4, E5, pointing game, Fase 7 (CheXpert) | Conclusão, resumo, revisão ABNT | Sim (E4 e E5) |
| 8+ | Revisão com o orientador, slides, ensaio da defesa | Ajustes finais | Não |

Ou seja: cerca de 7 semanas de trabalho até a monografia ficar pronta para o orientador revisar.

**Marcos**
- **Fim da semana 2 (antes da Fase 2):**
  - ambiente de treino escolhido (3.13): medido com 1 época do E1 no começo da Fase 2;
  - perguntas 1, 2 e 6 da seção 9: **resolvidas em 28/09/2026**.
- **Fim da semana 5:** congelamento dos experimentos essenciais e importantes. Depois disso, só
  complementares, análise e escrita.

**Quando a data de entrega (D) for definida**, contar de trás para frente e registrar na seção 10:
- D − 1 semana: monografia completa; só revisão ABNT e ajustes.
- D − 3 semanas: experimentos congelados.
- D − 4 a 5 semanas: fases 4 e 5 prontas.

Se o início + 7 semanas + o tempo de revisão do orientador passar de D, cortar pela lista da 6.1
(complementares primeiro, depois os itens "importante").

## 7. Regras de trabalho para a implementação

- Rodar `pytest` e o treino com `configs/debug.yaml` antes de cada commit que mexer em dados ou treino.
- Nunca versionar imagens, checkpoints, o `kaggle.json` ou qualquer credencial.
- Não mudar as decisões da seção 2 sem me perguntar. Decisões da seção 3 podem ser questionadas com justificativa.
- Não usar o conjunto de teste para escolher hiperparâmetros.
- Nunca sobrescrever o diretório de um experimento já rodado; cada experimento tem o seu.
- Commits pequenos, um por fase ou subtarefa, com mensagem descritiva.
- Atualizar a seção 10 sempre que uma decisão mudar.
- Ao fim de cada fase, me mandar: o que foi feito, como testar, e o que eu preciso rodar fora daqui (GPU, downloads).

## 8. Riscos e plano B

| Risco | Plano |
|---|---|
| Sessão (Colab/Kaggle) ou PC cai no meio do treino | Checkpoint atômico a cada época + retomada automática |
| Créditos ou cota de GPU acabam | GPU mais barata; cortar E4 e E5; trocar de ambiente (o código é o mesmo) |
| Armazenamento cheio | Dataset de 3–6 GB; guardar só `last.pt` e `best.pt` por experimento |
| Data de entrega mais cedo que o previsto | Cortar pela 6.1; a Metodologia já estará escrita |
| AUC abaixo da literatura | Esperado com divisão por paciente e rótulos ruidosos: reportar com IC e discutir; nunca ajustar olhando o teste |
| Orientador pede só 3 classes | Trocar a lista de classes no YAML e retreinar E1 e E2 |
| Atraso no cronograma | Seguir a 6.1: os complementares são os primeiros a ser reconsiderados |
| Demo falha no dia da defesa | Hugging Face Spaces → app rodando no seu computador → vídeo |

## 9. Perguntas em aberto

1. Treinar nas 14 classes e destacar 3 (padrão deste plano) ou treinar só nas 3?
   **Resolvido (28/09/2026):** 14 classes, para comparar com a literatura; a análise continua focada nas 3.
2. Divisão própria 70/15/15 por paciente ou lista oficial de teste do NIH (`test_list.txt`)?
   **Resolvido (28/09/2026):** a divisão própria é a principal (decisão do TCC 1; o CheXNet também
   usou divisão própria), e o E5 no split oficial entra no escopo para a comparação direta.
3. A interface Gradio basta como "sistema", ou a banca espera outra forma de entrega?
   **Recomendação:** Gradio publicado no Hugging Face Spaces, com vídeo de backup.
4. CheXpert entra no escopo ou fica como trabalho futuro?
   **Resolvido (28/09/2026):** entra como validação externa (Fase 7, complementar); treinar no
   CheXpert fica como trabalho futuro. O download exige cadastro na Stanford, feito pelo Rodrigo.
5. Monografia em LaTeX ou Word? **Resolvido pela Fase 6** (figuras e tabelas nos dois formatos).
   Falta só confirmar o modelo ABNT exigido pela UVV.
6. É preciso parecer do Comitê de Ética para usar uma base pública e anonimizada?
   **Resolvido (28/09/2026):** não. A monografia descreve a origem e a anonimização dos dados;
   rascunho em `docs/textos_monografia.md`.
7. (Rodrigo) Onde treinar: Colab, Kaggle ou GPU NVIDIA própria (ver 3.13)? *Antes da Fase 2.*
   **Resolvido (28/09/2026): RTX 2060 deste PC.** A 1ª época real do E1 levou 8,8 min
   (~178 img/s com o cuDNN determinístico da seção 3.6). A comparação com o PC da RTX 4060 continua
   opcional (basta copiar o `data/nih256.tar`), mas não é necessária.
8. (Rodrigo) Datas de entrega da monografia e da defesa. Quando definidas, aplicar a contagem
   regressiva da 6.2 e registrar as datas na seção 10.

## 10. Log de decisões

| Data | Decisão | Motivo |
|---|---|---|
| 27/09/2026 | Revisão 2 do plano | Ver Apêndice A |
| 27/09/2026 | Ambiente de treino em aberto (Colab, Kaggle ou GPU própria); código independente de ambiente | Ainda não definido onde treinar |
| 27/09/2026 | Cronograma relativo (em semanas), com contagem regressiva a partir da data de entrega | Data de entrega ainda não definida |
| 28/09/2026 | Versões exatas só no `requirements.txt`; o `pyproject.toml` exige apenas versões mínimas | Assim o `pip install -e .` não troca pacotes pré-instalados do Colab/Kaggle; as versões reais de cada treino ficam registradas por `utils.save_environment_info` |
| 28/09/2026 | Versões fixadas = as mais novas que ainda suportam Python 3.10 (numpy 2.2, pandas 2.3, scikit-learn 1.7) | Mantém o "Python 3.10+" da seção 2; as mesmas versões têm wheels para 3.11–3.13 (Colab/Kaggle) |
| 28/09/2026 | Arquivos de `configs/paths/` só podem ter a seção `paths`; caminhos relativos são resolvidos a partir da raiz do repositório | Hiperparâmetros não podem variar escondidos por ambiente; notebooks rodam de dentro de `notebooks/` e quebrariam caminhos relativos à pasta atual |
| 28/09/2026 | Recalibração dos escores por Platt scaling (por classe, ajustada na validação), prioridade "importante"; a interface só chama o valor de "probabilidade estimada" se a calibração no teste ficar boa | Pedido do Rodrigo: poder ler o número como chance, com base em evidência |
| 28/09/2026 | Frase de resumo na interface; nunca "Normal" ou "Sem doença" | O modelo só conhece 14 doenças e há falsos negativos |
| 28/09/2026 | Credencial do Kaggle pela chave "Legacy" (`kaggle.json`), com o kaggle CLI 1.6.17 | Os tokens novos exigem kaggle CLI ≥ 1.8, que só roda em Python ≥ 3.11; assim o ambiente continua no 3.10 |
| 28/09/2026 | kaggle CLI fixado em 1.6.17, não 1.7.x | A 1.7.4.5 (via `kagglesdk`) faz a requisição sem streaming e guarda o arquivo inteiro na memória antes de gravar; no download completo chegou a 10 GB de RAM e foi interrompida. A 1.6.17 grava em blocos e retoma downloads |
| 28/09/2026 | Divisão `patient_random`: pacientes embaralhados com a seed e acumulados inteiros até 70% e 85% das **imagens** | O critério de aceite mede proporções de imagens; como há pacientes com mais de 100 imagens, sortear 70% dos pacientes desviaria essas proporções |
| 28/09/2026 | Zip bruto no HD (`E:/datasets/nih`); imagens processadas no SSD (`data/nih256`) | O C: tem ~45 GB livres; o E: tem ~800 GB |
| 28/09/2026 | Amostra do Kaggle isolada em `data/sample/` (`configs/paths/local_sample.yaml`) | Splits, tabelas e figuras da amostra não se misturam com os reais, que são versionados |
| 28/09/2026 | Divisão final: `patient_random`, seed 42 → 78.486 / 16.812 / 16.822 imagens (70,00 / 14,99 / 15,00%). Estratificação multirrótulo não foi necessária | Passou no critério de prevalência de primeira: maior diferença relativa de 17,5% (Pneumonia na validação: 1,50% contra 1,28% no total); Atelectasia 3,6%, Efusão 6,3%. Nenhuma outra seed foi testada |
| 28/09/2026 | EDA em `src/chestxray/eda.py` (o notebook só chama e comenta) e figuras já no padrão da Fase 6 (português, vírgula decimal, PNG 300 dpi + PDF), via `src/chestxray/plotting.py` | Lógica testável e sem retrabalho na Fase 6 |
| 28/09/2026 | Perguntas 1, 2, 4 e 6 da seção 9 resolvidas seguindo as recomendações: 14 classes; divisão própria como principal; CheXpert só como validação externa; sem parecer do Comitê de Ética, com a origem dos dados descrita na monografia | Decisão do Rodrigo |
| 28/09/2026 | Os itens "se sobrar tempo" (E4, E5, pointing game, Fase 7) passam a se chamar "complementares" e entram no escopo; continuam por último na ordem de execução | Decisão do Rodrigo |
| 28/09/2026 | `split_info.json` em cada pasta de split, e o `split.py` recusa sobrescrever um split feito com outras configurações; o split oficial (E5) vai para `data/splits/official` | Revisão da Fase 1: rodar o split oficial com a configuração padrão teria apagado a divisão principal |
| 28/09/2026 | 3 imagens sem anatomia visível (00007160_002, rotulada Atelectasis; 00010007_121 e 00012249_001, No Finding) ficam no dataset | Defeito da própria base, não do pré-processamento; 3 em 112.120 (2 no treino, 1 na validação, 0 no teste) não mudam os resultados, e manter o conjunto completo preserva a comparação com a literatura. Citar na monografia como exemplo de ruído |
| 28/09/2026 | Rodrigo delegou a Fase 2 ("pode fazer o que achar melhor"): implementar, testar e já treinar E1 e E2 na RTX 2060 durante a noite; a Fase 3 espera a revisão dele | Aproveitar a noite de GPU; o plano manda parar entre as fases |
| 28/09/2026 | Precisão mista automática: fp16 com GradScaler em GPUs antes de Ampere (RTX 2060), bf16 a partir de Ampere (RTX 4060) | O bf16 não é nativo na RTX 2060 e seria lento |
| 28/09/2026 | Validação durante o treino também em precisão mista; as métricas finais da Fase 3 são calculadas em fp32 | Validação ~2× mais rápida; a escolha do `best.pt` pela AUC praticamente não muda |
| 28/09/2026 | `log.csv` reescrito a cada época a partir do histórico salvo no `last.pt` | Retomada sem linha duplicada nem faltando; um teste confirma que treino interrompido e retomado dá o mesmo resultado do treino direto (CPU) |
| 28/09/2026 | No Windows, o treino chama `SetThreadExecutionState` para o PC não suspender | Não altera configuração do sistema; vale só enquanto o processo roda |
| 28/09/2026 | YAMLs do E3, E4, E5 e das seeds extras criados na Fase 3, depois da escolha entre E1 e E2 pela validação | Todos partem do vencedor, que ainda não existe |
| 28/09/2026 | Ambiente de treino (3.13): RTX 2060 deste PC, batch 32, fp16 | 1ª época do E1 em 8,8 min (~178 img/s; 3,5 GB de memória de vídeo; 97% de uso da GPU). Com ~10–15 épocas por treino, os 5 treinos essenciais e importantes cabem em ~8–10 h de GPU, e os complementares em mais ~6 h: sem custo nem cota de Colab/Kaggle |
| 28/09/2026 | **Escolha pela validação, antes de qualquer predição de teste: E1 (sem `pos_weight`) é a configuração vencedora.** E3, as seeds extras, E4 e E5 partem do E1 | Critério: AUC média das 14 classes na validação (o mesmo do early stopping), em fp32. E1 0,838 (IC95% 0,829–0,845) × E2 0,832 (0,824–0,840); diferença pareada E1−E2 na média +0,005 (−0,001 a 0,010), não significativa; nas classes do TCC, E1 melhor em Atelectasia (+0,012; 0,006 a 0,018) e Efusão (+0,004; 0,001 a 0,007), Pneumonia sem diferença significativa. Com a média empatada no limite, o E1 também é mais simples e treinou mais rápido |
| 28/09/2026 | Seeds extras da configuração final: 43 e 44 (`e1_baseline_seed43`, `e1_baseline_seed44`); a seed da divisão continua 42 | Fixadas antes de treinar; todas as seeds usam a mesma divisão |
| 28/09/2026 | Split oficial do E5 gerado em `data/splits/official`: 71.255 / 15.269 / 25.596 imagens, sem paciente repetido (val tirada do `train_val_list.txt` por paciente, na proporção 70:15) | Prevalências do teste oficial acima do resto, como visto na revisão da Fase 1 |
| 28/09/2026 | `train.py --config a b c` treina uma fila de experimentos; uma falha não interrompe os seguintes | Fila da Fase 3 roda numa noite só, com um comando |
| 28/09/2026 | `--restart` no `train.py` (apaga a execução anterior do experimento), para uso em debug; `results/runs/debug/` fora do git | O treino de debug precisa poder rodar de novo; nos experimentos reais a regra de nunca sobrescrever continua valendo |
| 28/09/2026 | Código das Fases 4, 5 e 7 escrito e testado antes dos treinos da fila (sessão na nuvem, sem GPU nem imagens); rodar com o `best.pt` fica para o PC | Adiantar o que não depende do PC enquanto a fila da Fase 3 não roda |
| 28/09/2026 | Grad-CAM: camada padrão `relu`, `denseblock4` comparada; a escolha final sai do pointing game e das figuras `camadas_foco` | Critério objetivo além do visual; um teste confere que Grad-CAM na `relu` é igual ao CAM |
| 28/09/2026 | Galeria: VP e FP de maior escore, FN de menor escore, no máximo 1 imagem por paciente, desempate pelo nome | Critério fixo e declarado, para as figuras não parecerem escolhidas a dedo; sem repetir paciente |
| 28/09/2026 | Pointing game por imagem (várias caixas da mesma classe contam uma vez), IC95% de Wilson e o centro da imagem como referência trivial | Poucas caixas por classe (~20): Wilson se comporta bem com n pequeno; o centro mostra se o acerto é melhor que o acaso |
| 28/09/2026 | App carrega um pacote exportado (`app/model/`: pesos, config, calibração, exemplos), fora do git; o Space instala o pacote `chestxray` do GitHub | O Space não precisa do dataset nem dos resultados; pesos nunca versionados no git |
| 28/09/2026 | App mostra o escore recalibrado (Platt) e o limiar na mesma escala; o nome ("escore do modelo" ou "probabilidade estimada") é escolhido na exportação (`--label`) | Segue a Fase 5; a decisão depende das curvas de calibração do teste |
| 28/09/2026 | CheXpert: 7 classes em comum (Atelectasis, Cardiomegaly, Effusion ← Pleural Effusion, Pneumonia, Pneumothorax, Consolidation, Edema); imagens não quadradas redimensionadas para quadrado (`--fit resize`, padrão) ou com bordas pretas (`--fit pad`) | Igual ao que o app faz com qualquer imagem; `pad` fica como verificação de sensibilidade |
| 28/09/2026 | `evaluate --seeds` resume as 3 seeds (média ± desvio padrão amostral) | Plano 3.9 |

## 11. Referências

Referências do plano original:
- HUANG, G. et al. Densely Connected Convolutional Networks. CVPR, 2017.
- RAJPURKAR, P. et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv:1711.05225, 2017.
- WANG, X. et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. CVPR, 2017.
- IRVIN, J. et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. AAAI, 2019.
- SELVARAJU, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 2017.
- LITJENS, G. et al. A Survey on Deep Learning in Medical Image Analysis. Medical Image Analysis, 2017.

Adicionadas na revisão 2 (conferir volume, páginas e DOI antes de usar na monografia):
- BALTRUSCHAT, I. M. et al. Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray Classification. Scientific Reports, v. 9, 2019.
- GUENDEL, S. et al. Learning to recognize Abnormalities in Chest X-Rays with Location-Aware Dense Networks. arXiv:1803.04565, 2018.
- OAKDEN-RAYNER, L. Exploring Large-scale Public Medical Image Datasets. Academic Radiology, v. 27, n. 1, 2020.
- ZECH, J. R. et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLOS Medicine, v. 15, n. 11, 2018.
- ZHOU, B. et al. Learning Deep Features for Discriminative Localization. CVPR, 2016.

---

## Apêndice A: o que mudou na revisão 2

**Correções técnicas**
- Validação/teste sem `CenterCrop`: a imagem inteira é redimensionada para 224, preservando os
  cantos inferiores, onde aparece o derrame pleural.
- ReLU final não in-place e camada-alvo do Grad-CAM definida (com a equivalência Grad-CAM × CAM do CheXNet).
- Aviso sobre `use_deterministic_algorithms`, que quebra o treino da DenseNet na GPU em modo estrito.
- `max_epochs`, AUC indefinida em classes sem positivos (NaN + `nanmean`), checkpoint atômico.
- Pacote instalável (`pyproject.toml`), sem reinstalar torch no Colab; YAML com herança.

**Dados e infraestrutura**
- Pré-processamento em CPU, com saída num único tar (lido do zip ou de pasta; ver 3.4).
- Números de referência do dataset para conferir o parsing; regra objetiva de prevalência entre splits.
- `inference.py` único para avaliação, Grad-CAM e app; predições salvas em CSV; avaliação roda em CPU.

**Rigor da avaliação**
- Bootstrap por paciente, bootstrap pareado entre experimentos, AUPRC, calibração, subgrupos
  (sexo, idade, PA/AP) e 3 seeds da configuração final.
- Tabela da literatura preenchida e conferida no artigo do CheXNet, com as diferenças de protocolo.
- Experimentos com ID e prioridade; E5 opcional no split oficial.
- Galeria Grad-CAM com critério de escolha fixo; pointing game opcional.

**Entrega e planejamento**
- Demo no Hugging Face Spaces, com backup local e vídeo; saída chamada de "escore".
- Figuras em PNG e PDF com vírgula decimal; tabelas em CSV, Markdown e LaTeX.
- Fase 7 reduzida a validação externa no conjunto de validação do CheXpert.
- Cronograma semana a semana com marcos, prioridades, riscos e log de decisões.
- Novas perguntas: comitê de ética, GPU própria, datas de entrega e defesa.

### Revisão 2.1 (mesmo dia)

- Ambiente de treino deixou de ser Colab fixo: nova seção 3.13 compara Colab, Kaggle e GPU
  própria, com um critério de decisão (tempo de 1 época × número de treinos) antes da Fase 2.
- Código sem caminhos fixos: `configs/paths/<ambiente>.yaml`; `preprocess.py` lê de zip ou de
  pasta; notebook de treino só do ambiente escolhido.
- Cronograma sem datas: semanas relativas ao início, marcos relativos e contagem regressiva para
  aplicar quando a data de entrega (D) for definida.
- Fases 0 e 1 marcadas como sem necessidade de GPU, para começar já.

### Revisão 2.2 (28/09/2026)

- Fase 3: recalibração dos escores por Platt scaling (ajustada só na validação). Fase 5: frase de
  resumo acima da tabela, nunca "Normal", e "probabilidade estimada" só se a calibração se
  sustentar no teste.
- Perguntas 1, 2, 4 e 6 da seção 9 resolvidas; os itens "se sobrar tempo" viraram "complementares"
  e entraram no escopo (E4, E5, pointing game, Fase 7).
- Notas da execução das Fases 0 e 1: kaggle CLI 1.6.17 (a 1.7 guarda o download inteiro na
  memória); divisão por paciente buscando as proporções de imagens; `split_info.json` protege o
  split principal, e o E5 vai para `data/splits/official`; o teste do split oficial é mais "doente"
  (afeta a AUPRC do E5); rótulo `Infiltrate` no arquivo de caixas; 3 imagens sem anatomia mantidas.
- Nova pasta `docs/` com rascunhos de texto para a monografia.
