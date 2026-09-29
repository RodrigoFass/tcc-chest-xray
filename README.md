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
| 3 | Avaliação e experimentos | pronta; E1 a E5 e as seeds 43 e 44 treinados e avaliados (E1 escolhido) |
| 4 | Grad-CAM | pronta; galerias, caixas e pointing game do E1, camada `denseblock4` |
| 5 | Demonstração | pronta; página estática com o modelo no navegador: <https://huggingface.co/spaces/rotriguin/tcc-raio-x> |
| 6 | Material para a monografia | capítulos em `docs/monografia/`; falta a captura de tela da interface |
| 7 | Validação externa (CheXpert) | pronta; 202 imagens frontais, 6 classes, rótulos de radiologistas |

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

Para resumir as 3 seeds da configuração final (média ± desvio padrão):

```bash
python -m chestxray.evaluate --seeds results/runs/e1_baseline results/runs/e1_baseline_seed43 results/runs/e1_baseline_seed44
```

O notebook [notebooks/03_results.ipynb](notebooks/03_results.ipynb) roda, na ordem, tudo o que vem
depois dos treinos (predições, avaliação, comparações, seeds, Grad-CAM, pacote do app e CheXpert),
pulando o que ainda não tiver o que precisa.

## Grad-CAM

```bash
python -m chestxray.gradcam --config configs/experiments/e1_baseline.yaml
```

Precisa do `best.pt`, das imagens e da avaliação de teste (usa o `preds_test.csv` e o limiar do
`calibration.json`). Gera em `results/figures/e1_baseline/gradcam/` e `results/tables/e1_baseline/`:

- **Galeria** (`galeria_<classe>`): para cada doença do TCC, verdadeiros positivos, falsos positivos e
  falsos negativos no limiar de Youden da validação. **Critério de escolha, fixo:** os 3 verdadeiros
  positivos e os 3 falsos positivos de **maior** escore e os 3 falsos negativos de **menor** escore, no
  máximo uma imagem por paciente, desempate pelo nome do arquivo. A lista fica em `gradcam_selecao.csv`.
- **Camadas** (`camadas_foco`): as duas camadas-alvo candidatas lado a lado. Na ReLU final (`relu`),
  o Grad-CAM é igual ao CAM do CheXNet, o que um teste confere; a saída do último bloco denso
  (`denseblock4`) é a padrão, escolhida pelo *pointing game* (seção 10 do plano). As figuras de
  galeria e de caixas usam a primeira camada de `--layers`, que por padrão é a `denseblock4`.
- **Caixas do radiologista** (`caixas_<classe>`): todas as imagens de teste com caixa no
  `BBox_List_2017.csv`, com o heatmap, a caixa e o pico do mapa (verde se cair dentro da caixa).
- **Pointing game** (`pointing_game`): porcentagem de imagens em que o pico do mapa cai dentro da caixa,
  por doença e camada, com IC95% de Wilson e a taxa que o simples centro da imagem obteria.

## Demonstração

A demonstração pública é uma página estática (`webapp/`) em que o modelo roda **no navegador de quem
acessa**, com ONNX Runtime Web: a imagem não é enviada a nenhum servidor. O Hugging Face passou a cobrar
pelos Spaces com Gradio, e os Spaces estáticos continuam gratuitos (seção 10 do plano). A página
reproduz o app Gradio: a mesma frase de resumo, a mesma tabela e o mesmo mapa de calor.

1. Exportar o modelo para a página. Isso gera, em `webapp/model/`, o modelo em ONNX, os limiares, a
   calibração, os exemplos e o autoteste; a pasta nunca vai para o git. O comando confere que o modelo
   ONNX reproduz o `preds_test.csv` e o Grad-CAM do Python:

   ```bash
   python -m chestxray.webdemo --config configs/experiments/e1_baseline.yaml --out webapp/model
   ```

2. Testar localmente e abrir <http://localhost:8765>. O endereço <http://localhost:8765/?selftest>
   refaz no navegador o pipeline das imagens de exemplo e compara com o Python: a imagem vista pela
   rede, os escores e o mapa de calor.

   ```bash
   python -m http.server 8765 --directory webapp
   ```

3. Publicar no Hugging Face Spaces:
   - Crie o Space em <https://huggingface.co/new-space> com SDK **Static**, template **Blank** e
     visibilidade **Public**.
   - Faça login com um token de escrita.
   - Envie a pasta `webapp/`, já com o `webapp/model/` exportado:

   ```bash
   hf auth login
   hf upload <seu-usuario>/<nome-do-space> webapp . --repo-type space
   ```

Na página e no app Gradio, o valor aparece como "probabilidade estimada" (`--label probability`, o
padrão da página). Isso só vale porque as curvas de calibração do E1 no teste, depois do Platt, ficaram
perto da diagonal nas 3 classes do TCC (seção 10 do plano).

**Versão Gradio (local, backup para a defesa).** Faz a mesma análise em Python:

```bash
python -m chestxray.demo --config configs/experiments/e1_baseline.yaml --out app/model --label probability
python app/app.py
```

`python app/app.py --share` gera um link público temporário.

## Validação externa no CheXpert

Usa só o conjunto de **validação** do CheXpert, com rótulos pelo voto da maioria de três radiologistas,
sem incerteza. O modelo é o treinado no NIH, sem treinar nada. Entram só as imagens frontais (202, de 200
pacientes) e as 6 classes que existem nos dois datasets e têm rótulo de radiologista: atelectasia,
cardiomegalia, efusão, pneumotórax, consolidação e edema.

A Stanford AIMI distribui o CheXpert no Redivis e não disponibiliza mais o `valid.csv` original. Por
isso, os dados vêm de duas fontes:
- as imagens da validação vêm do **CheXpert Plus**, cujos próprios rótulos são automáticos e não são
  usados;
- os rótulos dos radiologistas são reconstruídos das anotações do **CheXlocalize**, que marcam as
  observações positivas de cada imagem. O CheXlocalize não anota pneumonia.

Para ter acesso, crie uma conta no Redivis, entre na organização AIMI e aceite o termo de pesquisa nos
dois conjuntos:
- <https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x> (CheXpert Plus);
- <https://stanford.redivis.com/datasets/efx9-5nspnbb4b> (CheXlocalize).

Depois, rode os comandos abaixo. Na primeira vez, o navegador abre para o login no Redivis:

```bash
python -m chestxray.data.download_chexpert --out E:/datasets/chexpert
python -m chestxray.external --config configs/experiments/e1_baseline.yaml --chexpert-root E:/datasets/chexpert
```

O segundo comando também aceita a pasta da distribuição original, com o `valid.csv`, se você a tiver.
Ele gera:
- `results/runs/e1_baseline/preds_chexpert.csv` e `metrics_chexpert.json`;
- a tabela `validacao_externa_chexpert`, com a AUC no CheXpert ao lado da AUC no teste do NIH;
- a figura `roc_foco_chexpert`.

Classes com menos de 30 casos são marcadas com † (sem conclusão sobre elas).

## Estrutura

```
configs/            base.yaml, debug.yaml, paths/ (por ambiente), experiments/
data/               não versionado, exceto data/splits/
docs/               rascunhos de texto para a monografia; docs/monografia/ tem os capítulos
src/chestxray/      pacote Python (config, utils, data/, models/, ...)
app/                interface Gradio (Fase 5), versão local
webapp/             página estática da demonstração (Hugging Face Space), modelo em ONNX no navegador
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

## Licença

O código está sob a licença MIT ([LICENSE](LICENSE)). A licença vale só para o código: os dados do
NIH ChestX-ray14 e do CheXpert seguem os termos de uso das instituições que os distribuem.
