import matplotlib

matplotlib.use("Agg")

from PIL import Image

from chestxray import diagram


def test_workflow_diagram_is_written_as_png_and_pdf(tmp_path):
    diagram.main(["--out", str(tmp_path / "fluxo")])
    png, pdf = tmp_path / "fluxo.png", tmp_path / "fluxo.pdf"
    assert pdf.stat().st_size > 0
    with Image.open(png) as img:
        assert img.width > 1500 and img.height > img.width  # 300 dpi, portrait
