from datasets import load_dataset

ds = load_dataset("jlohding/sp500-edgar-10k", split="train")

# Filtrer toutes les lignes où le nom contient "NVIDIA"
nvda_ds = ds.filter(lambda x: "NVIDIA" in x["company"])
example = nvda_ds[0]

def build_report_text(row):
    parts = []
    for key in ["item_1", "item_1A", "item_7", "item_7A", "item_8"]:
        if row.get(key):
            parts.append(row[key])
    return "\n\n".join(parts)

text_nvda_0 = build_report_text(example)
print(text_nvda_0[:10000])  # Affiche les 10 000 premiers caractères du rapport construit
