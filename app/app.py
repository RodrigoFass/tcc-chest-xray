"""Demo interface (Phase 5): upload a chest X-ray, see the value of each of the 14 diseases and
the Grad-CAM heatmap of the chosen one.

    python app/app.py                       # local, http://127.0.0.1:7860
    python app/app.py --share               # temporary public link (backup for the defense)
    python app/app.py --model-dir app/model

The model bundle comes from ``python -m chestxray.demo --config ... --out app/model``. On
Hugging Face Spaces this file is the entry point and the bundle sits next to it (see
``app/README.md``). All logic lives in :mod:`chestxray.demo`, which is tested.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from chestxray.demo import DISCLAIMER, PROBABILITY_NOTE, SCOPE_NOTE, DemoModel
from chestxray.plotting import pt

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = Path(os.environ.get("CHESTXRAY_MODEL_DIR", HERE / "model"))


def build_demo(model: DemoModel) -> gr.Blocks:
    choices = [(pt(c), c) for c in model.focus] + [(pt(c), c) for c in model.classes if c not in model.focus]
    notes = [DISCLAIMER, SCOPE_NOTE]
    if model.settings["label"] == "probability":
        notes.append(PROBABILITY_NOTE)

    def analyse(image: Image.Image | None, class_name: str):
        if image is None:
            raise gr.Error("Envie uma radiografia de tórax (PNG ou JPG).")
        image = image.convert("L")
        result = model.analyse(image)
        explanation = model.heatmap(image, class_name)
        caption = (f"### {result.summary}\n\nGrad-CAM de **{pt(class_name)}**: "
                   f"{model.label.lower()} {result.table.loc[result.table['Doença'] == pt(class_name), model.label].iloc[0]}")
        return caption, result.table, explanation.overlay

    with gr.Blocks(title="Raio X de tórax: apoio ao diagnóstico (TCC)") as demo:
        gr.Markdown("# Detecção de doenças pulmonares em raio X de tórax\n"
                    "Trabalho de Conclusão de Curso de Rodrigo Fassarella. DenseNet-121 treinada no "
                    "NIH ChestX-ray14, com mapas de calor Grad-CAM.")
        gr.Markdown("\n\n".join(notes))
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", image_mode="L", label="Radiografia de tórax (frontal)", height=360)
                class_name = gr.Dropdown(choices=choices, value=model.focus[0], label="Doença do mapa de calor")
                button = gr.Button("Analisar", variant="primary")
                if model.examples:
                    gr.Examples(examples=model.examples, inputs=image, label="Exemplos do conjunto de teste")
            with gr.Column(scale=1):
                summary = gr.Markdown()
                heatmap = gr.Image(label="Grad-CAM", height=360, interactive=False)
        table = gr.Dataframe(label=f"{model.label} de cada doença e limiar escolhido na validação",
                             interactive=False, wrap=True)
        button.click(analyse, inputs=[image, class_name], outputs=[summary, table, heatmap], api_name="analisar")
        class_name.change(lambda img, c: analyse(img, c) if img is not None else (gr.skip(),) * 3,
                          inputs=[image, class_name], outputs=[summary, table, heatmap])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--share", action="store_true", help="temporary public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_demo(DemoModel(args.model_dir)).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
