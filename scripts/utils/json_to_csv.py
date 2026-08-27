import os
import json
import csv
import re

# Kaydedilecek CSV dosyasının yolu ve kolon isimleri
output_csv = os.path.join('results', 'summary_results.csv')
fieldnames = [
    'type', 'dataset', 'base_model', 'judge_model', 
    'pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'pass@32', 'pass@64', 
    'cons_at_k', 'temperature', 'max_tokens'
]

rows = []

for root, dirs, files in os.walk('results'):
    for file in files:
        if file.endswith('summary.json'):
            filepath = os.path.join(root, file)
            # Dosya yolunu işletim sistemine uygun şekilde parçalara ayır
            parts = os.path.normpath(filepath).split(os.sep)
            
            # results/type/[judgments]/model_dir/dataset/file.json yapısı beklendiği için
            if len(parts) < 5:
                continue
                
            eval_type = parts[1] # base, instruct veya reasoning
            is_judgment = (parts[2] == 'judgments')
            
            if is_judgment:
                model_dir = parts[3]
                dataset = parts[4]
            else:
                model_dir = parts[2]
                dataset = parts[3]
                
            base_model = ""
            judge_model = ""
            temperature = ""
            max_tokens = ""
            
            # Regex ile temperature (örn: _t0.6) ve max_tokens (örn: _max16384 veya _16384) bulma
            temp_match = re.search(r'_t([\d\.]+)', model_dir)
            if temp_match:
                temperature = temp_match.group(1)
                
            max_match = re.search(r'_max(\d+)', model_dir)
            if max_match:
                max_tokens = max_match.group(1)
            else:
                # _16384 gibi sadece sayı olan varyasyonları yakala
                max_match_alt = re.search(r'_(\d+)$', model_dir)
                if max_match_alt:
                    max_tokens = max_match_alt.group(1)

            # Model isimlendirmelerini ayrıştırma
            if is_judgment:
                if '_evaluated_by_' in model_dir:
                    splits = model_dir.split('_evaluated_by_')
                else:
                    splits = model_dir.split('evaluated_by')
                
                base_model = splits[0]
                judge_model = splits[1] if len(splits) > 1 else ""
                
                # Judge modelin sonundaki _16384 gibi token kalıntılarını temizle
                judge_model = re.sub(r'_\d+$', '', judge_model)
                judge_model = re.sub(r'_t[\d\.]+$', '', judge_model) # Eğer temp kaldıysa
            else:
                # Base model içinden temp ve max token kısımlarını temizleyip saf modeli bırak
                base_model = re.sub(r'_t[\d\.]+.*$', '', model_dir)
                base_model = re.sub(r'_max\d+$', '', base_model)

            # JSON dosyasını oku ve verileri çek
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Hata! Okunamayan dosya: {filepath} ({e})")
                continue
                
            pass_at_k = data.get('pass_at_k', {})
            
            # GÜVENLİK KONTROLÜ: Eğer pass_at_k bir dict değil de float/int ise çökmeyi engelle
            if not isinstance(pass_at_k, dict):
                pass_at_k = {'1': pass_at_k}
                
            cons_at_k = data.get('cons_at_k', 0.0)
            
            rows.append({
                'type': eval_type,
                'dataset': dataset,
                'base_model': base_model,
                'judge_model': judge_model,
                'pass@1': pass_at_k.get('1', ''),
                'pass@2': pass_at_k.get('2', ''),
                'pass@4': pass_at_k.get('4', ''),
                'pass@8': pass_at_k.get('8', ''),
                'pass@16': pass_at_k.get('16', ''),
                'pass@32': pass_at_k.get('32', ''),
                'pass@64': pass_at_k.get('64', ''),
                'cons_at_k': cons_at_k,
                'temperature': temperature,
                'max_tokens': max_tokens
            })

# CSV Dosyasına Yazma İşlemi
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Başarılı! Toplam {len(rows)} satır veri '{output_csv}' konumuna kaydedildi.")