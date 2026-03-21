import nbformat

def merge_notebooks(path_a, path_b, output_path):
    # Lire les deux notebooks (avec leurs outputs)
    nb_a = nbformat.read(path_a, as_version=4)
    nb_b = nbformat.read(path_b, as_version=4)

    # Concaténer les cellules (outputs inclus automatiquement)
    nb_merged = nb_a  # on garde les métadonnées du premier
    nb_merged.cells = nb_a.cells + nb_b.cells

    # Optionnel : ajouter une cellule séparateur entre les deux
    nb_merged.cells = nb_a.cells + nb_b.cells

    # Écrire le notebook fusionné
    nbformat.write(nb_merged, output_path)
    print(f"Notebook mergé : {output_path}")

merge_notebooks("image_analyse_donnees.ipynb", "image_app_auto.ipynb", "Gr06_REXIA2026_image.ipynb")