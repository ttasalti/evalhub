#!/bin/bash
# ==============================================================================
# Master Runner Script for EvalHub Judges
# ==============================================================================

ENV_FILE="scripts/judge_only.env"
ORCHESTRATOR_SCRIPT="scripts/judge_vllm3.sh" # Kendi yoluna göre teyit et

# Format: "BASE_PATH|JUDGE_MODELS"
# Sadece ANA ÇIKTI HAZIR olanlar eklendi ve aynı path'e sahip olanların yargıçları birleştirildi.
declare -a job_list=(
    "results/reasoning/Qwen3.5-9B_think-true_t0.6_max16384/aime2026_tr|google/gemma-4-26B-A4B-it"
)

PREV_JOB_ID=""

echo "================================================================"
echo "Eksik Judgement'lar Icin Otomatik Gonderim Basliyor..."
echo "================================================================"

for job_item in "${job_list[@]}"; do
    # String'i '|' karakterinden ikiye bölüyoruz
    base_path="${job_item%%|*}"
    judges="${job_item##*|}"

    echo -e "\n[Hazirlaniyor] Path: $base_path"
    echo "[Hazirlaniyor] Judges: $judges"
    
    if [[ -n "$PREV_JOB_ID" ]]; then
        echo "[Hazirlaniyor] Dependency (Bagimlilik): $PREV_JOB_ID bekleniyor..."
    fi

    # sed ile .env dosyasını dinamik olarak güncelliyoruz. 
    # (Ayraç olarak '/' yerine '|' kullanıyoruz ki dosya yolları (path) sed'i bozmasın)
    sed -i "s|^BASE_GENERATION_PATH=.*|BASE_GENERATION_PATH=\"$base_path\"|g" "$ENV_FILE"
    sed -i "s|^JUDGE_MODELS=.*|JUDGE_MODELS=\"$judges\"|g" "$ENV_FILE"
    sed -i "s|^WAIT_FOR_JOB_ID=.*|WAIT_FOR_JOB_ID=\"$PREV_JOB_ID\"|g" "$ENV_FILE"

    # Orchestrator scriptini çalıştırıp çıktıyı yakalıyoruz
    output=$(bash "$ORCHESTRATOR_SCRIPT")
    echo "$output"
    
    # Çıktının içinden Slurm'un verdiği yeni Job ID'yi yakalıyoruz (Örn: "Job ID: 123456")
    new_job_id=$(echo "$output" | grep "Job ID:" | awk '{print $NF}')
    
    if [[ -n "$new_job_id" ]]; then
        PREV_JOB_ID="$new_job_id"
    else
        echo "[UYARI] Job ID yakalanamadi! Bir sonraki is bagimsiz (dependency olmadan) gonderilecek."
        PREV_JOB_ID=""
    fi
    
    # Slurm sbatch mekanizmasını yormamak ve işlemleri güvene almak için ufak bir bekleme
    sleep 3
done

echo -e "\n================================================================"
echo "Tum Gorevler Slurm'e Basariyla Iletildi ve Birbirine Baglandi!"
echo "================================================================"

# Kuyruğu temiz bırakmak için env dosyasını sıfırlıyoruz (isteğe bağlı)
sed -i "s|^WAIT_FOR_JOB_ID=.*|WAIT_FOR_JOB_ID=\"\"|g" "$ENV_FILE"