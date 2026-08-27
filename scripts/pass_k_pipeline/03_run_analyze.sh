#!/bin/bash
#SBATCH --job-name=analyze_gen       # İşin adı
#SBATCH --output=logs/%x-%j.out      # Standart çıktı logları
#SBATCH -e logs/%x-%j.err            # Hata logları
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1           
#SBATCH --mem=4G                     # JSON işlemleri için 4GB yeterli
#SBATCH --time=00:15:00              # Maksimum süre

# Çalışma dizinine geçiş yap
cd $SLURM_SUBMIT_DIR

# Ortamı aktif et (Kendi yapına göre conda veya venv kullanabilirsin)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[Sistem] .venv sanal ortamı aktif edildi."
fi
# eval "$(conda shell.bash hook)"
# conda activate evalhub_env

# Analiz edilecek modelleri buraya ekleyebilirsin
MODELS=(
    "DeepSeek-R1-0528-Qwen3-8B"
    "Qwen3.5-35B-A3B"
)

# Analiz edilecek benchmarkları buraya ekleyebilirsin
BENCHMARKS=(
    "aime2026"
    #"aime2025"
)

echo "============================================================"
echo "📊 Üretim (Generation) istatistik analizleri başlatılıyor..."
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    for BENCH in "${BENCHMARKS[@]}"; do
        echo "------------------------------------------------------------"
        echo "🚀 İşleniyor -> Model: $MODEL | Benchmark: $BENCH"
        echo "------------------------------------------------------------"
        
        # Python betiğini ilgili argümanlarla çağır
        python scripts/pass_k_pipeline/03_analyze_stats.py \
            --model "$MODEL" \
            --benchmark "$BENCH"
            
    done
done

echo "============================================================"
echo "✅ Tüm analiz işlemleri başarıyla tamamlandı!"
echo "============================================================"