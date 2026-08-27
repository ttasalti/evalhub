#!/bin/bash
#SBATCH --job-name=filter_json      # İşin adı
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err    # Hata mesajlarının yazılacağı dosya
#SBATCH --ntasks=1                  # Toplam görev sayısı
#SBATCH --cpus-per-task=1           # Bu işlem için 1 CPU çekirdeği fazlasıyla yeterli
#SBATCH --mem=4G                    # Bellek miktarı (Büyük JSONL dosyaları için 4GB yeterlidir)
#SBATCH --time=00:15:00             # Maksimum süre (15 dakika bu işlem için bolca yeterli)
#SBATCH --cpus-per-task 2

# 1. Adım: Çalışma dizinine geçiş yapın (Slurm varsayılan olarak işin gönderildiği dizinde çalışır ama garantiye alıyoruz)
cd $SLURM_SUBMIT_DIR
conda activate evalhub  # Eğer conda kullanıyorsanız, sanal ortamınızı aktif edin (Reponuzdaki README.md dosyasına göre 'evalhub' varsayılmıştır)

# 2. Adım: EvalHub için kurduğunuz sanal ortamı aktif edin
# (Reponuzdaki README.md dosyasına göre 'uv' kullanarak oluşturduğunuz varsayılmıştır)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Sanal ortam aktif edildi."
fi

# 3. Adım: Python betiğini çalıştırın
echo "JSONL filtreleme işlemi başlatılıyor..."

# Python yolu yeni klasör yapısına göre güncellendi
python scripts/cot_judge_pipeline/old_cot_organizer.py

echo "İşlem tamamlandı!"