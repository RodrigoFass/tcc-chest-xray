# Detecção de doenças pulmonares em raio X de tórax

Trabalho de Conclusão de Curso de Rodrigo Fassarella.

> **Protótipo acadêmico. Não usar para diagnóstico.**
> Este projeto é uma prova de conceito de apoio ao diagnóstico, desenvolvida para fins de
> pesquisa e ensino. **Não é um dispositivo médico**, não foi validado clinicamente e não deve
> ser usado para tomar decisões sobre pacientes.

O sistema recebe uma radiografia de tórax frontal e devolve um escore para cada uma das 14
doenças do NIH ChestX-ray14, com foco em **Pneumonia, Atelectasia e Efusão pleural**. O modelo é
uma DenseNet-121 pré-treinada no ImageNet e ajustada no ChestX-ray14, e cada predição vem com um
mapa de calor Grad-CAM. A especificação completa está em [PLANO_TCC.md](PLANO_TCC.md).

## Andamento

| Fase | Conteúdo | Situação |
|---|---|---|
| 0 | Esqueleto do projeto | pronta |
| 1 | Dados (download, pré-processamento, divisão, EDA) | pronta |
| 2 | Modelo e treino | pronta; E1 e E2 treinados |
| 3 | Avaliação e experimentos | E1 e E2 avaliados (E1 escolhido); E3–E5 e seeds prontos para treinar |
| 4 | Grad-CAM | — |
| 5 | Demonstração (Gradio) | — |
| 6 | Material para a monografia | — |

## Instalação local

Requer Python 3.10 ou mais novo. No Windows (PowerShell), a partir da raiz do repositório:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
# Com GPU NVIDIA (CUDA 12.6). Sem GPU, troque "cu126" por "cpu".
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

Se o PowerShell bloquear o `Activate.ps1`, dá para chamar o Python do ambiente direto:
`.venv\Scripts\python.exe -m pytest`. No Linux, use `python3 -m venv .venv` e
`source .venv/bin/activate`.

Para conferir o ambiente (versões, CUDA e nome da GPU):

```bash
python -m chestxray.utils
```

### Colab e Kaggle

Nesses ambientes o torch já vem instalado e casado com o CUDA da máquina, então **não** o
reinstale. O `requirements.txt` só exige uma versão mínima de torch e torchvision, e o
`pyproject.toml` só exige versões mínimas de tudo; assim, o comando abaixo mantém o torch da
máquina:

```bash
pip install -r requirements.txt
pip install -e .
```

As versões realmente usadas em cada treino ficam registradas junto com o experimento.

## Testes

```bash
pytest                 # todos
pytest -m "not slow"   # só os rápidos (o que o CI roda)
```

## Configuração

Cada experimento é um YAML que herda de [configs/base.yaml](configs/base.yaml) com
`extends: ...` e só sobrescreve o que muda (exemplo: [configs/debug.yaml](configs/debug.yaml)).
Os caminhos que dependem da máquina (`data_dir`, `runs_dir`, `checkpoint_dir`) ficam em
`configs/paths/<ambiente>.yaml` (`local`, `colab`, `kaggle`), que só pode conter a seção `paths`.
Caminhos relativos são resolvidos a partir da raiz do repositório, de onde quer que o comando
seja executado.

## Dados

### 1. Credenciais do Kaggle

1. Crie uma conta em [kaggle.com](https://www.kaggle.com) e abra **Settings → API**.
2. Clique em **Create Legacy API Key**. O navegador baixa um arquivo `kaggle.json`.
3. Salve o arquivo em `~/.kaggle/kaggle.json` (no Windows, `C:\Users\<você>\.kaggle\kaggle.json`),
   **fora do repositório**. Ele é uma senha: nunca o versione nem o cole em lugar nenhum.

Por que "Legacy": os tokens novos ("API Tokens") exigem o kaggle CLI 1.8 ou mais novo, que só roda em
Python 3.11+. O projeto usa o kaggle CLI **1.6.17**, que lê o `kaggle.json`. Não use a série 1.7: ela
carrega o download inteiro na memória antes de gravar o arquivo, o que esgota a RAM com os ~42 GB do
dataset completo (o `download.py` recusa essa versão).

No Colab, em vez do arquivo, crie os *Secrets* `KAGGLE_USERNAME` e `KAGGLE_KEY` e exporte-os como
variáveis de ambiente. Num notebook do Kaggle não é preciso baixar nada: anexe o dataset
"NIH Chest X-rays" ao notebook e use a pasta dele como `--input` no pré-processamento.

### 2. Baixar, pré-processar, dividir e explorar

Comece pela amostra do Kaggle (~5.600 imagens, 4,2 GB), que fica isolada em `data/sample/` e nunca
se mistura com os resultados reais:

```bash
python -m chestxray.data.download --sample
python -m chestxray.data.preprocess --input E:/datasets/nih/sample.zip --paths configs/paths/local_sample.yaml
python -m chestxray.data.split --paths configs/paths/local_sample.yaml
python -m chestxray.eda --paths configs/paths/local_sample.yaml
```

Depois, o dataset completo (~42 GB de download), com os caminhos de `configs/paths/local.yaml`:

```bash
python -m chestxray.data.download
python -m chestxray.data.preprocess
python -m chestxray.data.split
python -m chestxray.eda
```

| Etapa | O que faz | Onde grava |
|---|---|---|
| `download` | Baixa o zip pelo kaggle CLI, sem extrair | `raw_dir` (`E:/datasets/nih`) |
| `preprocess` | Lê cada PNG de dentro do zip (ou de uma pasta), converte para cinza 256×256 e empacota tudo num tar. Roda em paralelo e pode ser interrompido: na próxima vez continua de onde parou | `data_dir/images/`, `data_dir/manifest.csv` e `data/nih256.tar` |
| `split` | Lê os rótulos, divide 70/15/15 por paciente (seed fixa) e confere as prevalências. Não sobrescreve um split feito com outras configurações (ver `split_info.json`); o split oficial do NIH, usado no E5, vai para `data/splits/official` | `data/splits/*.csv` e `split_info.json` (versionados) e `results/tables/prevalencia_splits.csv` |
| `eda` | Figuras e tabelas da análise exploratória | `results/figures/eda_*.{png,pdf}` e `results/tables/eda_*.csv` |

O notebook [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) roda a EDA e comenta cada figura. Para
abri-lo com a amostra, defina `CHESTXRAY_PATHS=local_sample.yaml` antes de iniciar o Jupyter
(`jupyter notebook`).

Plano B, se faltar disco: baixe os 12 pacotes oficiais do NIH (link abaixo) e rode o
`preprocess --no-tar --input <pasta do pacote>` para cada um, apagando o pacote depois. As imagens se
acumulam em `data_dir`; no fim, copie os CSVs de metadados para `data_dir` e rode o `preprocess` uma
última vez sem `--no-tar`, para gerar o tar.

## Treino

```bash
python -m chestxray.train --config configs/experiments/e1_baseline.yaml
```

- Cada experimento grava em `results/runs/<experimento>/` a configuração resolvida, o
  `environment.json` (versões, GPU e commit de cada sessão), o `log.csv` (uma linha por época), o
  `train.log` e, no fim, o `summary.json`. Os pesos ficam em `checkpoints/<experimento>/`
  (`last.pt` a cada época e `best.pt` na melhor AUC média de validação) e nunca são versionados.
- Se o treino for interrompido (queda de energia, Ctrl+C, fim da sessão do Colab), rode o mesmo
  comando: ele continua da última época salva. Um experimento terminado não roda de novo, e a pasta
  de um experimento nunca é reaproveitada com outra configuração.
- No Windows, o treino pede ao sistema para não suspender enquanto roda (a tela ainda pode apagar);
  nenhuma configuração do Windows é alterada.
- Na GPU, a precisão mista é automática: fp16 em placas anteriores à série RTX 30 (como a RTX 2060)
  e bf16 da RTX 30/40 em diante. O batch 32 usa ~2,3 GB na RTX 2060; se o log avisar que a memória
  está quase cheia, reduza o `batch_size`.
- Teste rápido em CPU (200 imagens, 1 época, ~1 min); o `--restart` apaga a execução de debug
  anterior:

```bash
python -m chestxray.train --config configs/debug.yaml --restart
```

- Vários experimentos em fila, um depois do outro (por exemplo, durante a noite). Se um falhar, os
  seguintes rodam mesmo assim; rodar o mesmo comando de novo retoma o que ficou pela metade e pula
  o que já terminou. A fila da Fase 3 (E3, as duas seeds extras, E4 e E5; ~8–9 h na RTX 2060):

```bash
python -m chestxray.train --config configs/experiments/e3_noaug.yaml configs/experiments/e1_seed43.yaml configs/experiments/e1_seed44.yaml configs/experiments/e4_scratch.yaml configs/experiments/e5_official.yaml
```

  O E5 usa a divisão oficial do NIH, gerada uma vez com
  `python -m chestxray.data.split --config configs/experiments/e5_official.yaml`.

## Avaliação

As predições são geradas uma vez, em fp32; a avaliação roda só a partir delas, em CPU, e não
precisa das imagens. O modelo final é escolhido **só pela validação**, antes de gerar as predições
de teste (plano, seção 10).

```bash
python -m chestxray.inference --config configs/experiments/e1_baseline.yaml --splits val
```

```bash
python -m chestxray.evaluate --run results/runs/e1_baseline --split val
```

Depois da escolha registrada, o mesmo para o teste (`--splits test` e `--split test`). A
avaliação de teste usa a validação para o limiar de cada classe e para a recalibração (Platt,
gravada em `calibration.json`), e grava `metrics_test.json`, tabelas em `results/tables/<experimento>/`
(CSV, Markdown e LaTeX) e figuras em `results/figures/<experimento>/`. Para comparar experimentos
lado a lado, com diferenças pareadas de AUC (as mesmas reamostragens bootstrap para os dois):

```bash
python -m chestxray.evaluate --compare results/runs/e1_baseline results/runs/e2_posweight --pairs e2_posweight:e1_baseline
```

## Estrutura

```
configs/            base.yaml, debug.yaml, paths/ (por ambiente), experiments/
data/               não versionado, exceto data/splits/
docs/               rascunhos de texto para a monografia (origem dos dados, divisão etc.)
src/chestxray/      pacote Python (config, utils, data/, models/, ...)
app/                interface Gradio (Fase 5)
notebooks/          EDA, treino no ambiente escolhido, resultados
results/            runs/, figures/, tables/
tests/              pytest
```

Imagens, checkpoints (`.pt`), o arquivo `nih256.tar` e credenciais (`kaggle.json`, `.env`)
nunca são versionados.

## Dados e citação

Este trabalho usa o NIH ChestX-ray14, disponibilizado pelo **NIH Clinical Center**
(<https://nihcc.app.box.com/v/ChestXray-NIHCC>). Quem usar os dados deve citar:

> WANG, X.; PENG, Y.; LU, L.; LU, Z.; BAGHERI, M.; SUMMERS, R. M. ChestX-ray8: Hospital-scale
> Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of
> Common Thorax Diseases. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
> 2017, p. 2097-2106.
