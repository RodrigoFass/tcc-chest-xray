# Rascunho da monografia

Rascunho completo dos capítulos do TCC, escrito a partir do plano (`PLANO_TCC.md`), do documento do
TCC 1 e dos resultados versionados em `results/`. O texto está em Markdown para servir tanto para
LaTeX (abnTeX2) quanto para Word: `pandoc 01_introducao.md -o 01_introducao.docx` converte um
capítulo, e as tabelas já existem em `.tex` e `.md` em `results/tables/`.

| Arquivo | Capítulo | Situação |
|---|---|---|
| `00_resumo.md` | Resumo e *abstract* | completo; falta conferir o limite de palavras da UVV |
| `01_introducao.md` | 1 Introdução | completo, com as citações conferidas |
| `02_fundamentacao.md` | 2 Fundamentação teórica e trabalhos relacionados | completo; valores dos artigos conferidos |
| `03_metodologia.md` | 3 Materiais e métodos | completo; falta confirmar com o orientador a nota sobre o Comitê de Ética |
| `04_resultados.md` | 4 Resultados | completo; opcional: tempo de resposta da interface num celular |
| `05_discussao.md` | 5 Discussão | completo |
| `06_conclusao.md` | 6 Conclusão | completo |
| `referencias.md` | Referências (ABNT NBR 6023) | conferidas em 29/09/2026 (Crossref, PubMed e arXiv) |

Números e afirmações foram conferidos contra os arquivos de `results/` e, no caso da literatura, contra
os próprios artigos (numa revisão completa em 29/09/2026). A figura da interface é
`results/figures/interface_demo.png`, uma captura da página publicada.

## Convenções

- **`[PREENCHER: ...]`** marca um número ou trecho que depende de algo que ainda não foi feito. O texto
  entre colchetes diz de onde o valor sai. Para achar todas: `grep -rn "PREENCHER" docs/monografia`.
- **`[CONFERIR: ...]`** marca uma afirmação que o autor precisa confirmar (uma norma da UVV, uma
  orientação do orientador).
- Figuras e tabelas citadas como "Figura X" e "Tabela X" apontam para o arquivo de origem entre
  parênteses; a numeração final depende do modelo da UVV.
- Números seguem o padrão brasileiro (vírgula decimal), como as tabelas e figuras geradas pelo código.
- Citações no sistema autor-data da ABNT (NBR 10520), por exemplo (WANG et al., 2017a). Wang et al.
  tem duas referências: 2017a é o artigo do CVPR (o conjunto de dados) e 2017b é a versão 4 no arXiv
  (os resultados nas 14 classes, reproduzidos pelo CheXNet).

## O que ainda precisa do autor

1. **Modelo da UVV:** confirmar o modelo ABNT exigido (pergunta 5 da seção 9 do plano), o limite de
   palavras do resumo e o nome do orientador.
2. **Orientador:** confirmar se cabe citar a Resolução CNS nº 510/2016 na seção de aspectos éticos (3.13).
3. **Figuras de rascunho:** trocar a Figura 1 (em texto) por um diagrama desenhado e escolher as figuras
   da análise exploratória (Seção 3.2).
4. **Voz do autor:** revisar o texto na sua própria voz. Este rascunho é um ponto de partida, e a banca
   vai perguntar sobre cada decisão descrita aqui.
