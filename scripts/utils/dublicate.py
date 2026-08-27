import os
import json
import re
from collections import defaultdict

def get_model_config(filepath):
    parts = os.path.normpath(filepath).split(os.sep)
    if len(parts) < 5: return None
        
    eval_type = parts[1]
    is_judgment = (parts[2] == 'judgments')
    if not is_judgment: return None
    
    model_dir = parts[3]
    dataset = parts[4]
    
    temperature, max_tokens = "", ""
    temp_match = re.search(r'_t([\d\.]+)', model_dir)
    if temp_match: temperature = temp_match.group(1)
        
    max_match = re.search(r'_max(\d+)', model_dir)
    if max_match: max_tokens = max_match.group(1)
    else:
        max_match_alt = re.search(r'_(\d+)$', model_dir)
        if max_match_alt: max_tokens = max_match_alt.group(1)

    if '_evaluated_by_' in model_dir:
        splits = model_dir.split('_evaluated_by_')
    else:
        splits = model_dir.split('evaluated_by')
    
    base_model = splits[0]
    judge_model = splits[1] if len(splits) > 1 else ""
    judge_model = re.sub(r'_\d+$', '', judge_model)
    judge_model = re.sub(r'_t[\d\.]+$', '', judge_model)
    
    return (eval_type, dataset, base_model, judge_model, temperature, max_tokens)

def check_data_quality(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pass_at_k = data.get('pass_at_k', {})
        cons_at_k = data.get('cons_at_k', None)
        
        if not isinstance(pass_at_k, dict):
            pass_at_k = {'1': pass_at_k}
            
        is_valid = False
        # pass@k içinde geçerli (None veya NA olmayan) sayısal bir metrik var mı?
        for v in pass_at_k.values():
            if v is not None and str(v).lower() not in ['nan', 'na', 'null', '', 'none']:
                is_valid = True
                
        # cons_at_k (örneğin CoT tutarlılığı) için geçerlilik kontrolü
        if cons_at_k is not None and str(cons_at_k).lower() not in ['nan', 'na', 'null', '', 'none']:
            is_valid = True
            
        return is_valid, pass_at_k, cons_at_k
    except Exception:
        return False, {}, None

# ---------------------------------------------------------
# 1. GENEL İSİMLENDİRME EĞİLİMİNİ BULMA
# ---------------------------------------------------------
with_underscore = 0
without_underscore = 0

for root, dirs, files in os.walk('results'):
    for d in dirs:
        if 'evaluated_by' in d:
            if '_evaluated_by_' in d:
                with_underscore += 1
            else:
                without_underscore += 1

print("-" * 70)
print("1. KLASÖR İSİMLENDİRME STANDARDI KONTROLÜ")
print("-" * 70)
print(f"Toplam '_evaluated_by_' içeren klasör sayısı: {with_underscore}")
print(f"Toplam sadece 'evaluated_by' içeren klasör sayısı: {without_underscore}")

# Daha fazla kullanılan formatı standart kabul ediyoruz
if with_underscore > without_underscore:
    standard_format = "_evaluated_by_" 
else:
    standard_format = "evaluated_by"

print(f"\n-> Sistem geneliyle DAHA UYUMLU olan format: '{standard_format}'")

# ---------------------------------------------------------
# 2. DUPLICATE İÇERİK VE NA/NULL KONTROLÜ
# ---------------------------------------------------------
config_tracker = defaultdict(list)
for root, dirs, files in os.walk('results'):
    for file in files:
        if file.endswith('summary.json'):
            filepath = os.path.join(root, file)
            config = get_model_config(filepath)
            if config:
                config_tracker[config].append(filepath)

duplicates = {k: v for k, v in config_tracker.items() if len(v) > 1}

print("\n" + "-" * 70)
print("2. ÇAKIŞAN DOSYALARIN İÇERİK VE BÜTÜNLÜK (NA/NULL) KONTROLÜ")
print("-" * 70)

for key, paths in duplicates.items():
    dataset_name = key[1]
    base_model = key[2]
    print(f"\n[İnceleniyor] Base: {base_model} | Dataset: {dataset_name}")
    
    for path in paths:
        # Yol standart formata uyuyor mu? Özel kontrol (sadece evaluated_by olanlar alt küme kalmasın diye split ile kontrol ediyoruz)
        if standard_format == "_evaluated_by_":
            is_standard = "_evaluated_by_" in path
        else:
            is_standard = "_evaluated_by_" not in path and "evaluated_by" in path
            
        format_tag = "[YOL: UYUMLU ]" if is_standard else "[YOL: UYUMSUZ]"
        
        is_valid, passes, cons = check_data_quality(path)
        data_tag = "[VERİ: SAĞLAM]" if is_valid else "[VERİ: BOŞ/NA]"
        
        print(f"  {format_tag} {data_tag} -> {path}")
        
        if is_valid:
            pass_str = " | ".join([f"p@{k}: {v}" for k, v in passes.items() if v is not None])
            print(f"       -> Örnek İçerik: {pass_str} | cons@k: {cons}")