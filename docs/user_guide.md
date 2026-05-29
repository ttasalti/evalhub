# Kullanıcı rehberi — sıfırdan tek başına çalışmak için

Bu rehber, depoyu klonlayıp **kimseye danışmadan** end-to-end bir CoT-Pass@K
deneyi koşmaya, sonuçları görselleştirmeye ve hata çıkınca tek başına debug
etmeye yetecek kadar yoldur. Komutlar gerçek; kopyala-yapıştır çalışır.

> Cluster: nscluster (8× A100-40GB) veya nsdl2 (8× H200), Slurm 22.05.

## 1. İlk kurulum (bir kez yapılır)

```bash
# 1) Repo'da olduğundan emin ol
cd ~/evalhub

# 2) Conda ortamını aktive et
source /opt/Anaconda-2021.05/etc/profile.d/conda.sh
conda activate evalhub_env
export PATH="$HOME/.conda/envs/evalhub_env/bin:$PATH"

# 3) HuggingFace token'ı (gated dataset varsa zorunlu)
#    HF web → Settings → Access Tokens → "read" scope ile yeni token üret.
cp scripts/secrets.env.example scripts/secrets.env
# scripts/secrets.env içinde HF_TOKEN="hf_..." satırını doldur (vim ile)
# Bu dosya gitignore'da; commit ETMEZ.
```

Doğrulama:

```bash
which evalhub                 # /user/home/.../evalhub_env/bin/evalhub
evalhub --help                # subcommands: gen, eval, cot, report, tasks, view
```

## 2. Tek run — üç farklı yol

### 2A. Sabit demo (her şey env'de)

```bash
sbatch scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
```

3 benchmark seri, ~30 dk. Hızlı smoke-test için ideal.

### 2B. Env'den SBATCH override (submit.sh wrapper)

```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env
```

`submit.sh` env içindeki `SLURM_GRES`, `SLURM_MEM`, `SLURM_NODELIST`,
`SLURM_TIME`, `SLURM_CPUS_PER_TASK`, `SLURM_JOB_NAME`, `SLURM_PARTITION`
değerlerini sbatch CLI override olarak geçirir.

### 2C. Dinamik — model + benchmark CLI'dan seç (önerilen sweep yöntemi)

Generic config + CLI args. Tek bir `base.env`'i N farklı model/benchmark
ile yeniden kullan; env file'ı düzenlemen gerekmez.

```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
    --model Qwen/Qwen3.5-0.8B-Base \
    --judge Qwen/Qwen3.5-0.8B \
    --benchmarks "aime2026 aime2026_tr aime2026_pt" \
    --n-samples 4 \
    --output-root results_demo
```

Tanınan CLI flag'leri:

| Flag | Set ettiği env var |
|---|---|
| `--model X` | `TARGET_MODEL` |
| `--judge X` | `JUDGE_MODEL` |
| `--benchmark X` | `BENCHMARK` |
| `--benchmarks "X Y"` | `BENCHMARKS` (plural; loop) |
| `--target-state X` | `TARGET_STATE` |
| `--judge-state X` | `JUDGE_STATE` |
| `--output-root DIR` | `OUTPUT_ROOT` |
| `--temperature N` | `TARGET_TEMPERATURE` |
| `--judge-temperature N` | `JUDGE_TEMPERATURE` |
| `--n-samples N` | `TARGET_N_SAMPLES` |
| `--judge-n-samples N` | `JUDGE_N_SAMPLES` |
| `--max-completion-tokens N` | `TARGET_MAX_COMPLETION_TOKENS` |
| `--set KEY=VAL` | arbitrary KEY=VAL (repeatable) |
| `-- ...` | sonrası sbatch'e direkt geçer |

Precedence: **CLI args > env file > pipeline_common.sh defaults**.

Nasıl çalışıyor (önemli detay): `submit.sh` CLI args'ı `.overrides_<ts>_<pid>.env`
adında geçici bir dosyaya yazar (env file'ın yanına, gitignore'da), sonra
sbatch'e `EVALHUB_OVERRIDES_FILE=<path>` env değişkeniyle gönderir.
Orchestrator'un `pipeline_load_env()`'i bu override dosyasını main env'den
SONRA source eder, böylece whitespace/comma içeren değerler de düzgün geçer.

## 3. Job'u izleme

```bash
# Kuyruktaki / çalışan job'larım
squeue -u $USER

# Job final durumu (RUNNING / COMPLETED / FAILED / OUT_OF_MEMORY)
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed

# Canlı stdout / stderr
tail -f logs/evalhub-e2e-<JOBID>.out
tail -f logs/evalhub-e2e-<JOBID>.err

# Canlı vLLM log'u (her benchmark için ayrı dosya)
tail -f logs/vllm_target_<JOBID>_aime2026.log
tail -f logs/vllm_judge_<JOBID>_aime2026.log
```

### Sonuçları nereye düşer (V2 layout — current)

V2 layout 11 ayırt edici parametreyi (target/judge × {model, state, temp,
max_tokens, n_samples} + benchmark) klasör adına gömer; aynı tuple = aynı
path (idempotent re-run), farklı tuple = farklı path (collision yok).

```
results_demo/
├── report.csv                              # tüm runlar — long-form CSV
├── plots/                                  # heatmap + pass@k + base vs CoT
└── Qwen3.5-0.8B-Base__state-base__t0.6__max16384__n64/
    ├── aime2026/
    │   ├── aime2026.jsonl                  # raw generations
    │   ├── aime2026_raw.jsonl              # ham LLM response
    │   ├── aime2026_results.jsonl          # task başına correct[] + per_task_counts
    │   ├── aime2026_summary.json           # Pass@K + Cons@K + total/true/false/cot_false_count
    │   └── aime2026_per_task.csv           # task_id, true, false, cot_false, pass@K..., ground_truth
    └── judged_by/Qwen3.5-0.8B__state-think__t0.6__max16384__n3/
        └── aime2026/
            ├── cot_judge_raw.jsonl         # judge ham LLM output (diagnostic)
            ├── cot_judge.jsonl             # yes/no extracted per generation
            ├── cot_judge_results.jsonl     # judge eval (per-gen pass@k)
            ├── aime2026_cot_judge_input.jsonl
            ├── aime2026_cot_majority.jsonl # majority vote per task
            ├── aime2026_cot_results.jsonl  # correct[] + per_task_counts (cot_false dahil)
            ├── aime2026_cot_stats.json     # global true/false/cot_false/invalid_count
            ├── aime2026_cot_summary.json   # CoT-Pass@K (majority-voted, real metric)
            └── aime2026_cot_per_task.csv   # task bazlı CSV (cot_false sütunlu)
```

**Excel'de aç → soru bazlı dökümü gör.** Her `*_per_task.csv` satırı bir
benchmark sorusu için: kaç generation doğru, kaç yanlış, kaç cot-veto,
pass@1...pass@k_max, ground truth, majority vote, consensus doğru mu.

Job COMPLETE olduktan sonra `results_demo/report.csv` + `results_demo/plots/`
otomatik üretilir (her tek-benchmark koşusu da kendi REPORT aşamasını çalıştırır).

**Eski layout uyumluluğu**: `evalhub report aggregate` eski
`<class>/judgments/<target>_evaluated_by_<judge>_<max>/<benchmark>_t<T>/`
dizinlerini de okumaya devam eder; hiçbir mevcut veri kaybolmaz.

## 4. Sonuçları görselleştirme — Dashboard

İnteraktif Streamlit dashboard:

```bash
# Dashboard'u localhost:8501'de başlat (uzun süre çalışır, Ctrl-C ile durdurulur)
evalhub report dashboard \
    --csv    results_demo/report.csv \
    --results-root results_demo \
    --port 8501
```

VSCode Remote SSH kullanıyorsan, sol kenar çubuğundaki **PORTS** sekmesi
otomatik forward yapacaktır. Yapmıyorsa **+ Add Port** → 8501 → tarayıcıdan
`http://localhost:8501`.

Dashboard sekmeler:
- **Overview** — KPI kartları + tüm CSV tablo
- **Pass@K** — her benchmark için çizgi grafiği (base + cot)
- **Language comparison** — aynı benchmark ailesini diller arası karşılaştır
- **Pass vs CoT** — judge'ın Δ etkisini görselleştirir (pivot tablo + grouped bar)
- **Heatmap** — model × benchmark Pass@K ısı haritası
- **CoT veto** — judge veto oranı bar grafiği
- **Drill-down** — tek run'ın JSONL kayıtlarına bak

Statik plot'lar (PNG + PDF, paper için) zaten `results_demo/plots/` altında:
- `pass_at_k__<model>__<benchmark>.{png,pdf}` (her benchmark için)
- `base_vs_cot_pass_at_1.{png,pdf}`
- `pass_at_1_heatmap.{png,pdf}`
- `cot_veto_rate.{png,pdf}`

## 5. Çok modelli / çok benchmark'lı / çok sıcaklıklı DAG

İki yol var:

### 5A. Env file'da lists (sabit sweep tanımı)

```bash
# scripts/configs/my_sweep.env içeriği:
#   TARGET_MODELS="Qwen/Qwen3.5-0.8B-Base meta-llama/Llama-3.1-8B"
#   BENCHMARKS="aime2026 aime2026_tr math500"
#   TARGET_TEMPERATURES="0.6 0.9"
#   JUDGE_MODEL="Qwen/Qwen3.5-0.8B"
#   SLURM_GRES="gpu:nvidia_a100-pcie-40gb:1"
#   SLURM_MEM="30G" ...

scripts/orchestrate.sh scripts/configs/my_sweep.env sequential
# veya:
scripts/orchestrate.sh scripts/configs/my_sweep.env parallel
```

### 5B. CLI ile dinamik sweep (önerilen, env dosyası dokunulmaz)

```bash
scripts/orchestrate.sh scripts/configs/base.env sequential \
    --models "Qwen/Qwen3.5-0.8B-Base meta-llama/Llama-3.1-8B" \
    --benchmarks "aime2026 aime2026_tr math500" \
    --temps "0.6 0.9" \
    --judge Qwen/Qwen3.5-0.8B
```

CLI flag'leri (submit.sh ile aynı precedence):
- `--models "A B"` → `TARGET_MODELS`
- `--benchmarks "X Y"` → `BENCHMARKS`
- `--temps "0.6 0.9"` → `TARGET_TEMPERATURES`
- `--judge X` → `JUDGE_MODEL`
- `--target-state X`, `--judge-state X`, `--output-root DIR`, `--set KEY=VAL`

`sequential` mode: her (model, benchmark, temp) sırayla bekler — queue dostu,
yavaş.

`parallel` mode: tüm base job'ları aynı anda submit edilir, judge'lar kendi
base'lerini bekler, son report job tüm judge'ları bekler — hızlı ama N adet
GPU rezerve eder.

Submit edilen tüm job'ları görmek için:

```bash
squeue -u $USER -o '%.10i %.20j %.8T %.10M %.6D %R'
```

## 6. Hata çıkarsa — debug rehberi

### Job FAILED, MaxRSS yüksek (OOM-kill)

vLLM **V1 multi-process** açık kalmış olabilir. Kontrol:

```bash
grep VLLM_ENABLE_V1_MULTIPROCESSING scripts/lib/pipeline_common.sh
# Beklenen: VLLM_ENABLE_V1_MULTIPROCESSING:-0  (yani off, default)
```

### Job FAILED ama `.err` BOŞ

Bash silent exit. En sık sebep: `pipeline_common.sh` içinde bir helper
function `[[ test ]] && cmd` ile bitiyorsa, test false olunca set-e tetikler
ve script sessizce ölür. Bilinen iki örnek (`build_gen_args`, çıkış return 0)
zaten patch'lendi; benzeri yeni bir helper eklersen sonuna `return 0` koy.

`.out`'ta son satır vLLM healthy ise:

```bash
tail -5 logs/evalhub-e2e-<JOBID>.out
# Sonra elle bash trace:
PROJECT_ROOT=$PWD bash -x scripts/run_end_to_end.sh scripts/configs/<config>.env 2>&1 | head -200
```

### vLLM died, real Python traceback gerek

`start_vllm` öldükten sonra otomatik olarak vLLM log'unun son 200 satırını
`.err`'e basar. Eksikse:

```bash
tail -200 logs/vllm_target_<JOBID>_<benchmark>.log
```

### `evalhub gen` AuthenticationError

`HF_TOKEN` boş ya da yanlış. `scripts/secrets.env`'i doldur (bölüm 1).

### Streamlit dashboard çalışmıyor

```bash
# Bağımlılıkları kontrol et
python -c "import streamlit, plotly; print(streamlit.__version__, plotly.__version__)"

# Yoksa kur (vllm/torch etkilenmez)
pip install streamlit plotly
```

### Submit ettiğim job submit.sh ile beklenmedik node'a gitti

`scripts/submit.sh` env'den `SLURM_NODELIST` okur, header'ı override eder.
İstenmiyorsa env'den sil veya direkt `sbatch run_end_to_end.sh ...` ile
submit et.

## 7. Yeni model ekleme (config dosyası)

```bash
# 1) Yeni env config oluştur
cp scripts/configs/qwen_0.8b_demo.env scripts/configs/<yeni>.env
# Düzenle: TARGET_MODEL, JUDGE_MODEL, TARGET_STATE (base|non-think|think),
# OUTPUT_ROOT, SLURM_*

# 2) Model template var mı kontrol et
ls scripts/templates/
# qwen3.5-base.jinja, qwen3.5-think.jinja, gemma4-*.jinja, ministral3-*.jinja
# Yoksa modelin family ismine göre yeni .jinja eklemen + evalhub/utils/model_state.py
# MODEL_FAMILIES'e kayıt eklemen gerek.

# 3) Submit
sbatch scripts/run_end_to_end.sh scripts/configs/<yeni>.env
```

## 8. Yeni benchmark ekleme (tam pipeline)

Mevcut bir benchmark dosyasını referans al (`evalhub/benchmarks/math/aime2026/__init__.py`):

```python
from evalhub.benchmarks.math.base import MathDataset
from evalhub.benchmarks.registry import register_dataset

NAME = "<benchmark_name>"
HUB = "owner/dataset"          # HF Hub ID veya None + local parquet path

@register_dataset((NAME, HUB, True))   # True = evaluable
class <Name>Dataset(MathDataset):
    def load_tasks(self) -> None:
        # ...
        # her item için self.add_task(Task(task_id=..., prompt=...))
        # her ground truth için self.add_groundtruth(GroundTruth(task_id=..., answer=...))
        pass
```

Sonra modül import edilir hale getir:

```bash
# evalhub/benchmarks/math/__init__.py içine ekle:
from .<benchmark_name> import *
```

Test:

```bash
evalhub tasks | grep <benchmark_name>      # listede görünmeli
```

Çalışan örnekleri: `evalhub/benchmarks/math/aime2026_tr/`, `aime2026_pt/`,
`math500/`, `gsm8k/`.

### CSV'den yerel benchmark — `tubitak_math2026` örneği

`evalhub/benchmarks/math/tubitak_math2026/` Türkçe matematik olimpiyatı için
CSV-okuyan bir benchmark — HF Hub yok, sadece `tubitak_math2026.csv` (32 soru,
integer + LaTeX karışık cevaplar) ve loader. CSV'yi düzenledikten sonra
`rm -rf ~/.cache/evalhub/` ile cache invalidate edilir.

Pass@K:
```bash
evalhub gen --model hosted_vllm/Qwen/Qwen3.5-0.8B-Base --tasks tubitak_math2026 \
    --temperature 0.6 --n-samples 8 --output-dir results/tubitak/
evalhub eval --tasks tubitak_math2026 \
    --solutions results/tubitak/tubitak_math2026.jsonl --output-dir results/tubitak/
```

CoT-Pass@K (Türkçe judge prompt zorunlu — `JUDGE_TASK=cot_judge_tr`):
```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/tubitak_math2026.env \
    --model Qwen/Qwen3.5-0.8B-Base --judge Qwen/Qwen3.5-0.8B
```

## 9. Yararlı tek seferlik komutlar

```bash
# Sonuçlarımı kaybetmek için (clean slate)
rm -rf results_demo logs/

# Tüm job'larımı toplu cancel
squeue -u $USER -h -o '%i' | xargs -r scancel

# Streamlit'i background'da bırakmadan unutmak için bul + durdur
pgrep -af streamlit
pkill -f 'streamlit.*dashboard'

# Eski log'ları arşivle
mkdir -p logs/archive && mv logs/*.log logs/*.err logs/*.out logs/archive/ 2>/dev/null

# Mevcut conda env'in versiyonlarını yedekle (debug için)
pip freeze > /tmp/evalhub_env_freeze.txt
```

## 10. Tek başına sorun çözme algoritması

```
Job FAILED  ─→  sacct'a bak (State, ExitCode, MaxRSS)
              │
              ├── OUT_OF_MEMORY    → mem yetersiz veya vLLM V1 mp açık
              ├── State=FAILED     → .err'i oku, traceback ara
              │                     │
              │                     ├── boş    → bash silent exit; bash -x ile çalıştır
              │                     ├── HF auth → secrets.env boş, doldur
              │                     ├── vLLM died → vllm_*.log tail, traceback'i oku
              │                     └── Python  → traceback'i oku, fix
              │
              └── CANCELLED       → time limit aşıldı (SLURM_TIME artır)
                                    veya elle scancel'lendi

Plot/CSV eksik  ─→ report stage çalıştı mı?
                 │
                 ├── Çalışmadı → run_report.sh elle çalıştır:
                 │              `evalhub report aggregate ... && evalhub report plot ...`
                 │
                 └── Çalıştı ama eksik → scan.py debug log'ları açık tut:
                                        Python -c "import logging; logging.basicConfig(level='DEBUG')"
                                        ile aggregate'i tekrar çalıştır

Dashboard açılmıyor → streamlit/plotly kur, port forwarding aç
```

## 11. Hangi dosyanın ne yaptığı (kuş bakışı)

| Dosya | Görev |
|---|---|
| `scripts/run_end_to_end.sh` | Tek job, 3 aşama (base + judge + cot finalize) + report |
| `scripts/run_eval_only.sh` | Sadece base gen + eval |
| `scripts/run_judge_only.sh` | Sadece judge gen + eval + cot finalize (var olan base'i kullanır) |
| `scripts/run_report.sh` | Sadece `evalhub report aggregate + plot` (DAG'ın tail node'u) |
| `scripts/submit.sh` | env'den SLURM_* okur, sbatch CLI override geçirir |
| `scripts/orchestrate.sh` | Multi-model × multi-benchmark × multi-temp DAG submitter |
| `scripts/lib/pipeline_common.sh` | Shared shell helpers (vLLM lifecycle, env loading, vs.) |
| `scripts/templates/` | vLLM `--chat-template` jinja dosyaları |
| `scripts/configs/` | Per-config env dosyaları (model + sample + Slurm knob'ları) |
| `scripts/secrets.env.example` | Token şablonu (kopyalanır, doldurulur, gitignore'da) |

| Python modülü | Görev |
|---|---|
| `evalhub/cli.py` | Typer CLI giriş noktası (`evalhub gen/eval/cot/report/tasks/view`) |
| `evalhub/generator/` | Generation orchestration (litellm + hosted_vllm) |
| `evalhub/benchmarks/` | Benchmark family'leri (math, code, general, multilingual, cot, logic) |
| `evalhub/benchmarks/math/base.py` | Genel math dataset + per_task_counts ekleyen `evaluate()` |
| `evalhub/benchmarks/math/verifier/rllm.py` | Cevap karşılaştırma (decimal, fraction, sympy) |
| `evalhub/cot/extract.py` | Base-correct generation'ları judge input'a çıkarır |
| `evalhub/cot/aggregate.py` | Judge verdict'lerini majority vote'a indirir |
| `evalhub/cot/metrics.py` | CoT veto'yu base'e uygular, per_task_counts üretir |
| `evalhub/report/scan.py` | Sonuç dizinlerini walk eder, summary'leri parse eder |
| `evalhub/report/aggregate.py` | CSV üretir (long form) |
| `evalhub/report/plots.py` | matplotlib statik plot seti (paper için) |
| `evalhub/report/dashboard.py` | Streamlit interaktif dashboard |
| `evalhub/utils/model_state.py` | Model family + state → chat template eşleştirmesi |

## 12. Yardım gerektiğinde

- `evalhub --help` ve her subcommand'in `--help`'i
- `docs/onboarding.md` — proje genel tanıtımı
- `docs/reporting.md` — reporting subsystem detayı
- `docs/scripts.md` — scripts kataloğu
- `docs/evalhub.md` — Python modül kataloğu
