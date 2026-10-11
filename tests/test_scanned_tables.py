from PIL import Image, ImageDraw, ImageFont
from src.scanned_tables import read_ruled_table


def make_grid():
    im = Image.new("RGB", (1050, 600), "white")
    draw = ImageDraw.Draw(im)
    xs = [110, 360, 650, 960]
    ys = [110, 190, 270, 350, 430]
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill="black", width=4)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill="black", width=4)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    content = [["Item", "Price", "Limit"], ["XZ500", "1250", "42"],
               ["XZ600", "2300", "58"], ["XZ700", "3400", "63"]]
    for i, row in enumerate(content):
        for j, val in enumerate(row):
            draw.text((xs[j] + 18, ys[i] + 20), val, fill="black", font=font)
    return im


def test_ruled_grid_cell_recognition():
    result = read_ruled_table(make_grid(), lang="eng")
    assert result is not None
    assert result["num_rows"] == 4
    assert result["num_cols"] == 3
    assert "XZ700" in result["text"]
    assert "3400" in result["text"]
    assert "63" in result["text"]


def test_plain_text_image_is_not_false_positive():
    im = Image.new("RGB", (500, 200), "white")
    ImageDraw.Draw(im).text((30, 40), "Simple plain sentence", fill="black")
    assert read_ruled_table(im, lang="eng") is None


def test_scanned_pdf_runs_table_ocr(tmp_path):
    import io
    import fitz
    from src.parsing import parse_document
    im = make_grid()
    png = io.BytesIO()
    im.save(png, format="PNG")
    path = tmp_path / "scanned.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=800, height=500)
    page.insert_image(fitz.Rect(0, 0, 800, 500), stream=png.getvalue())
    pdf.save(path)
    pdf.close()
    elements = parse_document(path)
    assert any(e.element_type == "table" and e.metadata.get("ocr_method") == "ruled_grid"
               for e in elements)
    assert any("XZ700" in e.content for e in elements if e.element_type == "table")
