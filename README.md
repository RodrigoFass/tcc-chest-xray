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
| 1 | Dados (download, pré-processamento, divisão, EDA) | — |
| 2 | Modelo e treino | — |
| 3 | Avaliação e experimentos | — |
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

## Estrutura

```
configs/            base.yaml, debug.yaml, paths/ (por ambiente), experiments/
data/               não versionado, exceto data/splits/
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
