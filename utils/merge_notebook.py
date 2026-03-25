import nbformat
import uuid

def merge_notebooks(path_a, path_b, path_c, output_path):
    nb_a = nbformat.read(path_a, as_version=4)
    nb_b = nbformat.read(path_b, as_version=4)
    nb_c = nbformat.read(path_c, as_version=4)

    nb_merged = nb_a
    nb_merged.cells = nb_a.cells + nb_b.cells + nb_c.cells

    # Corriger les IDs dupliqués
    ids_vus = set()
    for cell in nb_merged.cells:
        if cell.id in ids_vus:
            cell.id = str(uuid.uuid4())[:8]
        ids_vus.add(cell.id)

    nbformat.write(nb_merged, output_path)
    print(f"Notebook mergé : {output_path}")

merge_notebooks("../partie_1/Gr06_REXIA2026_tabulaire.ipynb", "../partie_2/Gr06_REXIA2026_image.ipynb", "../partie_3/Gr06_REXIA2026_texte.ipynb", "../Gr06_REXIA2026.ipynb")
#merge_notebooks("../partie_3/partie_3_raphael.ipynb", "../partie_3/gael-rexi-part-3.ipynb", "../partie_3/partie3_classification.ipynb", "../partie_3/xia.ipynb", "../partie_3/Gr06_REXIA2026_texte.ipynb")