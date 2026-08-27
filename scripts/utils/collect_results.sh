#!/bin/bash

# Sonuçların kaydedileceği CSV dosyası results klasörü içinde olacak
OUTPUT_CSV="results/all_results.csv"

echo "JSON dosyaları taranıyor, metrikler ayrıştırılıyor ve kolonlar özel sıraya diziliyor..."

python3 - << 'EOF'
import os
import json
import glob
import csv

# İç içe geçmiş sözlükleri düzleştiren fonksiyon
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

results = []
all_metrics = set()

# 'results' dizini altındaki tüm summary JSON'larını bul
search_pattern = "results/**/*_summary.json"

for filepath in glob.glob(search_pattern, recursive=True):
    parts = filepath.split(os.sep)
    
    # Güvenlik kontrolü: parts içinde "results" olmalı
    if "results" not in parts:
        continue
        
    res_idx = parts.index("results")
    
    # Dosya yolu en az 'results / kategori / model / benchmark / dosya.json' uzunluğunda olmalı
    if len(parts) < res_idx + 4:
        continue
        
    # 1. Kırılım: Kategori (base, instruct, reasoning)
    category = parts[res_idx + 1]
    
    # 2. Kırılım: Model adı veya "judgments" klasörü
    if parts[res_idx + 2] == "judgments":
        model_folder = parts[res_idx + 3]
        
        if "evaluated_by" in model_folder:
            splits = model_folder.split("evaluated_by", 1)
            base_model = splits[0].strip("_")
            judge_model = splits[1].strip("_")
        else:
            base_model = model_folder
            judge_model = "Unknown_Judge"
    else:
        model_folder = parts[res_idx + 2]
        base_model = model_folder
        judge_model = "" 
        
    # Dosya adından benchmark ismini çıkarma
    filename = parts[-1]
    benchmark = filename.replace("_summary.json", "")
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            flat_data = flatten_dict(raw_data)
            
            metrics = {}
            for key, val in flat_data.items():
                # Sadece sayısal değerleri, boolean'ları ve düz metinleri al
                if isinstance(val, (int, float, str, bool)):
                    metrics[key] = val
                    all_metrics.add(key)
                    
            row_data = {
                "Category": category,
                "Base Model": base_model,
                "Judge Model": judge_model,
                "Benchmark": benchmark
            }
            row_data.update(metrics)
            results.append(row_data)
            
    except Exception as e:
        print(f"Uyarı - Okunamayan/Bozuk JSON: {filepath} | Hata: {e}")

# Satır Sıralaması: Category -> Base Model -> Judge Model -> Benchmark
results.sort(key=lambda x: (x["Category"], x["Base Model"], x["Judge Model"], x["Benchmark"]))

# KOLON SIRALAMASI MANTIĞI
pass_k_cols = []
cons_cols = []
note_cols = []
other_cols = []

for m in all_metrics:
    if m.startswith("pass_at_k_"):
        pass_k_cols.append(m)
    elif m == "cons_at_k":
        cons_cols.append(m)
    elif m == "note":
        note_cols.append(m)
    else:
        other_cols.append(m)

# pass_at_k kolonlarını sonlarındaki sayısal değere göre (k değerine) küçükten büyüğe sırala
def get_k_value(col_name):
    try:
        return int(col_name.split('_')[-1])
    except ValueError:
        return 999999 # Sayı çıkarılamazsa en sona at

pass_k_cols.sort(key=get_k_value)

# Diğer isimsiz metrikleri kendi içinde alfabetik sırala
other_cols.sort()

# Nihai Kolon (Header) Sıralamasını Oluştur
headers = ["Category", "Base Model", "Judge Model", "Benchmark"] + pass_k_cols + cons_cols + other_cols + note_cols

output_path = "results/all_results.csv"

# Hedef klasörün var olduğundan emin ol
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Sonuçları CSV'ye yazdır (Eksik kolonlar boş bırakılacak şekilde DictWriter bunu otomatik çözer)
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"İşlem tamamlandı! Toplam {len(results)} adet değerlendirme başarıyla '{output_path}' dosyasına yazıldı.")
EOF