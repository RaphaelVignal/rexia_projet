from huggingface_hub import snapshot_download

# Téléchargement du dataset complet dans le dossier local "Data"
chemin_dossier = snapshot_download(
    repo_id="google/civil_comments",
    repo_type="dataset",  # Indispensable car il s'agit d'un dataset et non d'un modèle par défaut
    local_dir="Data"
)

print(f"Le dataset a été téléchargé avec succès dans : {chemin_dossier}")