# scripts/ kataloğu

Her shell script ne yapar, hangi env değişkenlerini okur, hangi artifact'leri
üretir. Sıra: en yaygın kullanılandan en spesifik olana.

## Orkestrasyon

### `scripts/run_end_to_end.sh`

**Amaç:** Tek Slurm job içinde tek model + (BENCHMARKS plural varsa) N benchmark için
base eval + judge + cot finalize + report stage'lerini sırayla yürütür.

**Submit:**
```bash
sbatch scripts/run_end_to_end.sh scripts/configs/<config>.env
# veya
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/<config>.env
```

**SBATCH header (default):** `--gres=gpu:nvidia_a100-pcie-40gb:1`,
`--cpus-per-task=8`, `--mem=30G`, `--time=12:00:00`, `--nodelist=nscluster`.
`submit.sh` ile env'den override edilebilir.

**Akış:**
1. `pipeline_load_env "$1"` → env + secrets dosyalarını source eder
2. `apply_legacy_env_aliases`, `apply_target_defaults`, `apply_judge_defaults`, `apply_common_defaults`
3. `BENCHMARKS` plural varsa kendisini bash ile re-invoke eder (her benchmark için ayrı `BENCHMARK` set'i)
4. Tüm benchmark'ler bittikten sonra `evalhub report aggregate` + `plot`
5. Stage 1 (per benchmark): `start_vllm(target) → pipeline_run_target_gen_eval → stop_vllm`
6. Stage 2 (per benchmark): `evalhub cot extract → start_vllm(judge) → pipeline_run_judge_gen_eval → stop_vllm`
7. Stage 3 (per benchmark): `evalhub cot finalize`

**Üretilen artifact'ler:** Bölüm 3, kullanıcı rehberi.

### `scripts/run_eval_only.sh`

**Amaç:** Sadece base generation + base evaluation (judge yok).

**Use case:** Bir model'i hızlıca smoke-test etmek; DAG'da base stage'i ayrı submit etmek.

**Akış:** Stage 1 only. Output: `*_raw.jsonl`, `*_results.jsonl`, `*_summary.json`.

### `scripts/run_judge_only.sh`

**Amaç:** Var olan base sonuçlarını alır, judge'ı çalıştırır, cot finalize üretir.

**Required env:** `BASE_RESULTS_DIR` veya `BASE_RESULTS_FILE` + `BASE_RAW_FILE` —
ya bir parent dir göster, ya da explicit iki dosya path'i ver.

**Use case:** Aynı base sonuçları farklı judge model'leri ile karşılaştırmak.

### `scripts/run_report.sh`

**Amaç:** Sadece `evalhub report aggregate` + `evalhub report plot` çalıştırır.

**Use case:** DAG'ın tail node'u olarak (orchestrate.sh tarafından kullanılır), veya elle eski sonuçların report'unu yenilemek için.

**SBATCH:** `--cpus-per-task=2`, `--mem=8G`, `--time=00:30:00` (light job).

### `scripts/submit.sh`

**Amaç:** Env file içindeki SLURM_* knob'larını sbatch CLI override olarak verir; orkestrator script'in kendi #SBATCH header'ı fallback.

**Honored Slurm env vars:** `SLURM_JOB_NAME`, `SLURM_GRES`, `SLURM_CPUS_PER_TASK`,
`SLURM_MEM`, `SLURM_TIME`, `SLURM_NODELIST`, `SLURM_PARTITION`,
`SLURM_EXTRA_ARGS` (whitespace-separated passthrough).

**CLI overrides** (CLI > env > defaults):

| Flag | Sets |
|---|---|
| `--model X` | `TARGET_MODEL` |
| `--judge X` | `JUDGE_MODEL` |
| `--benchmark X` | `BENCHMARK` |
| `--benchmarks "X Y"` | `BENCHMARKS` (plural, loops) |
| `--target-state X` | `TARGET_STATE` (base/non-think/think) |
| `--judge-state X` | `JUDGE_STATE` |
| `--output-root DIR` | `OUTPUT_ROOT` |
| `--temperature N` | `TARGET_TEMPERATURE` |
| `--judge-temperature N` | `JUDGE_TEMPERATURE` |
| `--n-samples N` | `TARGET_N_SAMPLES` |
| `--judge-n-samples N` | `JUDGE_N_SAMPLES` |
| `--max-completion-tokens N` | `TARGET_MAX_COMPLETION_TOKENS` |
| `--set KEY=VAL` | arbitrary KEY=VAL (repeatable) |
| `-- ...` | everything after passed verbatim to `sbatch` |

CLI override mechanism: submit.sh writes a temp file next to the env file
(`.overrides_<timestamp>_<pid>.env`, gitignored), then passes
`EVALHUB_OVERRIDES_FILE=<path>` via sbatch `--export`. The orchestrator's
`pipeline_load_env()` sources the override file LAST so values win over the
main env. This avoids the whitespace/comma corruption problem of stuffing
multiple `KEY=VAL` pairs into `sbatch --export`.

```bash
# Baked env:
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/qwen_0.8b_demo.env

# Dynamic:
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
    --model Qwen/Qwen3.5-0.8B-Base \
    --judge Qwen/Qwen3.5-0.8B \
    --benchmarks "aime2026 aime2026_tr aime2026_pt" \
    --n-samples 4
```

### `scripts/orchestrate.sh`

**Amaç:** Multi-model × multi-benchmark × multi-temp DAG submitter.

**Required env:** `TARGET_MODELS` (plural), `BENCHMARKS` (plural),
`JUDGE_MODEL`. Singular fallback'leri var (TARGET_MODEL, BENCHMARK).

**Use case:** Bir gecede 5 model × 10 benchmark sweep'i çalıştırmak.

**Mode toggle (2. arg):**
- `sequential` (default): her kombinasyon önceki bitince başlar (queue-friendly)
- `parallel`: tüm base'ler aynı anda submit, judge'lar kendi base'lerini bekler, report tüm judge'ları bekler

**CLI overrides** (same precedence as submit.sh):

| Flag | Sets |
|---|---|
| `--models "A B"` | `TARGET_MODELS` |
| `--benchmarks "X Y"` | `BENCHMARKS` |
| `--temps "0.6 0.9"` | `TARGET_TEMPERATURES` |
| `--judge X` | `JUDGE_MODEL` |
| `--target-state X` | `TARGET_STATE` |
| `--judge-state X` | `JUDGE_STATE` |
| `--output-root DIR` | `OUTPUT_ROOT` |
| `--set KEY=VAL` | arbitrary |

```bash
# Lists from env:
scripts/orchestrate.sh scripts/configs/my_sweep.env sequential

# Lists from CLI:
scripts/orchestrate.sh scripts/configs/base.env parallel \
    --models "Qwen/Qwen3.5-0.8B-Base meta-llama/Llama-3.1-8B" \
    --benchmarks "aime2026 math500" \
    --temps "0.6 0.9" \
    --judge Qwen/Qwen3.5-0.8B
```

## Kütüphane

### `scripts/lib/pipeline_common.sh`

**Source'lanır, çalıştırılmaz.** Üç orkestratörün ortak helper'larını barındırır.

**Public function'lar (özet):**

| Function | Görev |
|---|---|
| `pipeline_log MSG` | Timestamp'li log mesajı |
| `pipeline_die MSG` | Log + exit 1 |
| `pipeline_load_env FILE` | env file source + optional secrets.env source |
| `require_env VAR1 VAR2 ...` | Eksik varsa pipeline_die |
| `pipeline_init_paths` | OUTPUT_ROOT'u normalize eder, dizin yaratır |
| `apply_legacy_env_aliases` | Legacy BASE_*/PORT/etc.'ı yeni isimlere map'ler, soft default'ları (PYTORCH_CUDA_ALLOC_CONF, VLLM_USE_TRITON_FLASH_ATTN, VLLM_ENABLE_V1_MULTIPROCESSING, PYTHONUNBUFFERED) export eder |
| `apply_target_defaults` / `apply_judge_defaults` / `apply_common_defaults` | Tüm TARGET_*/JUDGE_* knob'larına default değer atar |
| `detect_model_class MODEL` | `base` veya `instruct` döndürür |
| `target_clean_name MODEL` | Output dir adı için temizlenmiş model adı |
| `resolve_template MODEL STATE` | Chat template .jinja path'ini Python ile resolve eder |
| `start_vllm MODEL PORT TP STATE LOG_FILE` | vLLM background'da başlatır, healthy olana kadar bekler, SERVER_PID set eder, ölürse log dump |
| `stop_vllm` | SERVER_PID'i kill eder |
| `pipeline_register_cleanup` | trap EXIT stop_vllm |
| `build_gen_args ROLE ARRAY` | TARGET_/JUDGE_ knob'larından `evalhub gen --flag VAL` args üretir, sonu `return 0` |
| `compose_target_dir BENCHMARK` | V2 layout: `${OUTPUT_ROOT}/${clean}__state-${TARGET_STATE}__t${T}__max${N}__n${TARGET_N_SAMPLES}/${benchmark}` |
| `compose_judge_dir BENCHMARK` | V2 nested: `<target_root>/judged_by/${judge_clean}__state-${JUDGE_STATE}__t${JT}__max${JN}__n${JUDGE_N_SAMPLES}/${benchmark}` |
| `pipeline_run_target_gen_eval DIR BM` | `evalhub gen` + `evalhub eval` |
| `pipeline_run_judge_gen_eval DIR INPUT` | `evalhub gen` + `evalhub eval` for judge. Return: global `JUDGE_SOLUTIONS_OUT` |
| `pipeline_write_empty_cot_summary DIR BM` | Boş cot_summary.json yaz (0 base-correct durumu için stub) |

**Soft default'lar:** `VLLM_ENABLE_V1_MULTIPROCESSING=0` (host RAM ikilenmesini engeller),
`PYTHONUNBUFFERED=1` (silent crash buffer'ı engeller), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`VLLM_USE_TRITON_FLASH_ATTN=0`. Her birini env file'dan override edebilirsin.

## Config dosyaları

### `scripts/configs/*.env`

Her config bir (model, benchmark seti, hiperparametre) kombinasyonu tanımlar.
Yapısı:

```bash
# Slurm
SLURM_JOB_NAME=, SLURM_GRES=, SLURM_CPUS_PER_TASK=, SLURM_MEM=, SLURM_TIME=, SLURM_NODELIST=

# Models + benchmarks
TARGET_MODEL=,    JUDGE_MODEL=
BENCHMARKS=,      OUTPUT_ROOT=
TARGET_STATE=,    JUDGE_STATE=                # base | non-think | think

# Sampling
TARGET_TEMPERATURE=, TARGET_TOP_P=, TARGET_N_SAMPLES=, TARGET_MAX_COMPLETION_TOKENS=, ...
JUDGE_TEMPERATURE=,  JUDGE_TOP_P=,  JUDGE_N_SAMPLES=,  JUDGE_MAX_COMPLETION_TOKENS=, ...

# Optional vLLM CLI knobs (boş = vLLM default)
TARGET_GPU_MEMORY_UTILIZATION=, TARGET_MAX_MODEL_LEN=, TARGET_ENFORCE_EAGER=,
TARGET_SWAP_SPACE=, TARGET_DTYPE=, TARGET_KV_CACHE_DTYPE=, TARGET_VLLM_EXTRA_ARGS=
(JUDGE_* counterparts)

# Ports
TARGET_PORT=30000, JUDGE_PORT=30001
```

Mevcut örnekler:
- `scripts/configs/base.env` — generic (model/benchmark CLI'dan gelir)
- `scripts/configs/qwen_0.8b_demo.env` — Qwen 0.8B + AIME demo (model + benchmark baked)

### `scripts/secrets.env.example` / `scripts/secrets.env`

`.example` repo'da, gerçek dosya gitignore'da. `pipeline_load_env` env file'ın
yanında veya `scripts/`'ta `secrets.env` varsa onu da source eder. HF_TOKEN
gibi secret'ları orada tut.

## Template'ler

### `scripts/templates/*.jinja`

vLLM `--chat-template` ile geçirilen Jinja şablonları. Üç tip × üç family:

- `qwen3.5-base.jinja`, `qwen3.5-no-think.jinja`, `qwen3.5-think.jinja`
- `gemma4-base.jinja`, `gemma4-no-think.jinja`, `gemma4-think.jinja`
- `ministral3-base.jinja`, `ministral3-instruct.jinja`, `ministral3-reasoning.jinja`

Yeni bir model family eklemek için: jinja yaz + `evalhub/utils/model_state.py:MODEL_FAMILIES`
listesine ekle.

## Komut hattı eşdeğerleri

| Yapılmak istenen | Hızlı yol | DAG yolu |
|---|---|---|
| Tek model, 1 benchmark | `sbatch run_end_to_end.sh <cfg>` | — |
| Tek model, N benchmark | aynı (BENCHMARKS plural set) | — |
| N model, M benchmark | — | `orchestrate.sh <cfg> sequential\|parallel` |
| Sadece judge yeniden | — | `sbatch run_judge_only.sh <cfg>` (BASE_RESULTS_DIR set) |
| Sadece report yeniden | — | `sbatch run_report.sh <cfg>` (veya elle aggregate+plot) |
