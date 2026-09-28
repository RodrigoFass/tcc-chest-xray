# Rascunho da monografia

Rascunho completo dos capítulos do TCC, escrito a partir do plano (`PLANO_TCC.md`), do documento do
TCC 1 e dos resultados versionados em `results/`. O texto está em Markdown para servir tanto para
LaTeX (abnTeX2) quanto para Word: `pandoc 01_introducao.md -o 01_introducao.docx` converte um
capítulo, e as tabelas já existem em `.tex` e `.md` em `results/tables/`.

| Arquivo | Capítulo | Situação |
|---|---|---|
| `00_resumo.md` | Resumo e *abstract* | rascunho; números finais dependem da fila de treinos |
| `01_introducao.md` | 1 Introdução | pronto para revisão |
| `02_fundamentacao.md` | 2 Fundamentação teórica e trabalhos relacionados | pronto para revisão |
| `03_metodologia.md` | 3 Materiais e métodos | pronto para revisão |
| `04_resultados.md` | 4 Resultados | E1 e E2 completos; E3, E4, seeds, E5, Grad-CAM e CheXpert com lacunas |
| `05_discussao.md` | 5 Discussão | pronto para revisão; alguns parágrafos dependem das lacunas |
| `06_conclusao.md` | 6 Conclusão | rascunho |
| `referencias.md` | Referências (ABNT NBR 6023) | conferir cada entrada |

## Convenções

- **`[PREENCHER: ...]`** marca um número ou trecho que depende de algo que ainda não foi rodado (a
  fila de treinos, o Grad-CAM, o CheXpert). O texto entre colchetes diz de qual arquivo o valor sai.
  Para achar todas: `grep -rn "PREENCHER" docs/monografia`.
- **`[CONFERIR: ...]`** marca uma afirmação que o autor precisa confirmar (uma referência, uma
  norma da UVV, um detalhe de um artigo).
- Figuras e tabelas citadas como "Figura X" e "Tabela X" apontam para o arquivo de origem entre
  parênteses; a numeração final depende do modelo da UVV.
- Números seguem o padrão brasileiro (vírgula decimal), como as tabelas e figuras geradas pelo código.
- Citações no sistema autor-data da ABNT (NBR 10520), por exemplo (WANG et al., 2017).

## O que ainda precisa do autor

1. Confirmar o modelo ABNT exigido pela UVV (pergunta 5 da seção 9 do plano) e o nome do orientador.
2. Rodar a fila de treinos e o notebook `notebooks/03_results.ipynb`, e preencher as lacunas.
3. Conferir as referências (volume, páginas, DOI) em `referencias.md`.
4. Revisar o texto na sua voz: este rascunho é um ponto de partida, e a banca vai perguntar sobre
   cada decisão descrita aqui.
