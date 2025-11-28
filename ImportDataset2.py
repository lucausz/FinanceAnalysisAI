import os
from datasets import load_dataset

print("1. Chargement du dataset...")
ds = load_dataset("jlohding/sp500-edgar-10k", split="train")

# --- CONFIGURATION ---
TARGET_COMPANIES = 1000
OUTPUT_DIR = "data"  # <--- Le nom de ton dossier principal
companies_data = {} 

print(f"\n2. Extraction des données pour {TARGET_COMPANIES} entreprises...")

# --- EXTRACTION (identique à avant) ---
for i, row in enumerate(ds):
    company_name = row.get('company')
    text_content = row.get('item_7')
    raw_date = row.get('date') 

    if not company_name or not text_content or not raw_date:
        continue

    if len(companies_data) >= TARGET_COMPANIES and company_name not in companies_data:
        continue

    try:
        year = int(str(raw_date)[:4])
    except Exception:
        continue

    if len(str(text_content)) < 100:
        continue

    if company_name not in companies_data:
        companies_data[company_name] = {}
        print(f"   -> Nouvelle entreprise trouvée ({len(companies_data)}/{TARGET_COMPANIES}): {company_name}")
    
    companies_data[company_name][year] = text_content


print(f"\nDonnées collectées. Génération des fichiers dans le dossier '{OUTPUT_DIR}'...")

# --- ÉCRITURE DES FICHIERS ---
files_created = 0

for name, years_dict in companies_data.items():
    print(f"Enregistrement des rapports pour {name} ...")
    # 1. Nettoyage du nom
    safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "").strip()
    
    # 2. Construction du chemin complet : data/NomEntreprise
    # os.path.join s'occupe de mettre les bons slashs (\ sur Windows, / sur Linux)
    full_dir_path = os.path.join(OUTPUT_DIR, safe_name)
    
    valid_years = [y for y in years_dict if 2010 <= y < 2020]
    if not valid_years:
        continue

    try:
        # Cela va créer "data" ET "data/NomEntreprise" d'un coup si nécessaire
        os.makedirs(full_dir_path, exist_ok=True)
    except OSError:
        continue

    for year in valid_years:
        # Construction du chemin du fichier : data/NomEntreprise/Fichier.txt
        filename = f"{safe_name}_item_7_{year}.txt"
        full_file_path = os.path.join(full_dir_path, filename)
        
        content = years_dict[year]
        
        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write(f"Report for {name} in {year}:\n\n{content}")
        files_created += 1

print(f"\nTerminé ! {files_created} fichiers créés dans le dossier '{OUTPUT_DIR}'.")