import nbformat
import uuid

def merge_notebooks(path_a, path_b, output_path):
    nb_a = nbformat.read(path_a, as_version=4)
    nb_b = nbformat.read(path_b, as_version=4)

    # Contenu des cellules du notebook A pour détecter les doublons
    contenu_a = set(''.join(cell.source) for cell in nb_a.cells)

    # Garder uniquement les cellules du notebook B absentes du notebook A
    cellules_b_uniques = [
        cell for cell in nb_b.cells
        if ''.join(cell.source) not in contenu_a
    ]


    nb_merged = nb_a
    nb_merged.cells = nb_a.cells + cellules_b_uniques

    # Corriger les IDs dupliqués
    ids_vus = set()
    for cell in nb_merged.cells:
        if cell.id in ids_vus:
            cell.id = str(uuid.uuid4())[:8]
        ids_vus.add(cell.id)

    nbformat.write(nb_merged, output_path)
    print(f"Notebook mergé : {output_path}")

merge_notebooks("image_analyse_donnees.ipynb", "image_app_auto.ipynb", "Gr06_REXIA2026_image.ipynb")