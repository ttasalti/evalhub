# evalhub/ Python kataloğu

CLI ve kütüphane Python modüllerinin kuş bakışı görünümü.

## CLI giriş noktası

### `evalhub/cli.py`

Typer'a dayalı CLI. Komutlar:

| Komut | Görev |
|---|---|
| `evalhub gen` | Generation: model + tasks + sampling knobs → `*_raw.jsonl` |
| `evalhub eval` | Evaluation: solutions + tasks → `*_results.jsonl` + `*_summary.json` |
| `evalhub tasks` | Kayıtlı benchmark'ları listele |
| `evalhub view` | Bir result JSONL'i terminal'de zengin formatta gör |
| `evalhub cot extract` | Base-correct generation'ları judge input'a çıkar |
| `evalhub cot aggregate` | Judge verdict'lerini majority vote → `*_cot_majority.jsonl` |
| `evalhub cot metrics` | CoT veto'yu uygula → `*_cot_results.jsonl` + `*_cot_summary.json` + `*_cot_stats.json` |
| `evalhub cot finalize` | extract + (judge ayrı çalışmış) + aggregate + metrics tek seferde |
| `evalhub report aggregate` | OUTPUT_ROOT'u walk et, long-form CSV üret |
| `evalhub report plot` | CSV → 4 plot tipi (matplotlib + seaborn) PNG + PDF |
| `evalhub report dashboard` | CSV → Streamlit + Plotly interaktif dashboard |

Bütün argümanlar `--help` ile keşfedilebilir: `evalhub gen --help` vb.

## Generation + evaluation

### `evalhub/generator/`

Generation orchestration. `litellm` üzerinden hosted_vllm/<model> endpoint'ine
async olarak istek atar. `num-workers` paralel request sayısı; `timeout`
istek başına saniye.

### `evalhub/evaluator/`

Evaluation entry point. `evalhub eval` çağırınca her benchmark'in
`MathDataset.evaluate()` (veya muadili) tetiklenir.

## Benchmarks

### `evalhub/benchmarks/registry.py`

`@register_dataset((name, hub_id_or_none, evaluable))` decorator. Tüm
benchmark sınıfları kendini buraya kaydeder; `DATASET_MAP[name] → Class`.

### `evalhub/benchmarks/base.py`

`Dataset` ve `Task`, `GroundTruth` dataclass'ları. Her dataset
`load_tasks()`, `format_prompt(item)`, `extract_solution(task_id, response)`,
`check_correct(extracted, ground_truth, task_id)` implement eder.

### `evalhub/benchmarks/math/base.py`

`MathDataset` — math benchmark'larının ortak ata sınıfı. `evaluate()`
metodu solutions'ı yükler, her task için `is_correct[]` array'i hesaplar,
Pass@K + majority vote + **`per_task_counts`** (yeni: true/false/cot_false/invalid_format)
üretir.

### `evalhub/benchmarks/math/verifier/`

Cevap karşılaştırma. İki ayrı verifier:

- `rllm.py` — `extract_boxed_answer`, `grade_answer` (mathd + sympy fallback). Decimal, fraction,
  sympy expr destekli.
- `dapo.py` — alternatif normalize'lı string match grader.

`__init__.py`'deki `grade_answer()` üçünü sırayla dener: `grade_answer_mathd` → `grade_answer_sympy` → `verify_dapo`. İlk true sonuç döner.

### Bilinen math benchmark'lar

- `aime2026`, `aime2026_tr`, `aime2026_pt` (HF Hub veya local parquet)
- `aime2024`, `aime2025`
- `math500`, `hendrycks_math`
- `gsm8k` (decimal cevap desteği zaten var)
- Logic: `autologi`, `zebralogic` (math değil ama benzer pipeline)

### Diğer family'ler

- `evalhub/benchmarks/code/livecodebench/` — code gen + execution-based eval
- `evalhub/benchmarks/general/mmlu_redux/`, `ceval/` — MCQA
- `evalhub/benchmarks/multilingual/...` — birden çok dilli math/MCQA

## CoT-Pass@K subsystem

### `evalhub/cot/ids.py`

`encode(task_id, gen_idx) → "task_id__gen{idx}"` formatı.

### `evalhub/cot/extract.py`

Base eval'in `_results.jsonl` + `_raw.jsonl`'ından `correct=True` olan
generation'ları çıkarır → `*_cot_judge_input.jsonl`. Her record:
`task_id`, `original_task_id`, `generation_idx`, `ground_truth`,
`generated_answer`, `raw_response`.

### `evalhub/benchmarks/cot/judge.py`

`CoTJudgeDataset` (`MathDataset`'ın alt sınıfı). `cot_judge`, `cot_judge_tr`,
`cot_judge_pt` benchmark'larını register eder. `load_tasks()` judge-input
JSONL'i okur, her generation için orijinal soruyu DATASET_MAP üzerinden bulup
judge prompt'unu formatlar.

`_lookup_original_question` (line 88-116): `original_task_id.split("/")[0]`'i
DATASET_MAP key'i olarak kullanır. Dash→underscore normalize eder
(`AIME2026-PT/7 → aime2026_pt`).

### `evalhub/cot/aggregate.py`

Judge'ın `*_results.jsonl`'ından her generation için `yes_count`,
`no_count`, `invalid_count`, `majority_correct = yes > no` → `*_cot_majority.jsonl`.

### `evalhub/cot/metrics.py`

`apply_cot_metrics(base_results, majority, output_results, summary, stats)`:
base record'ları okur, base-correct olan generation'ları judge majority false
ise `"cot_false"` etiketler. Per-task Pass@K, Cons@K, **per_task_counts**
hesaplar; global stats üretir.

### `evalhub/cot/pipeline.py`

`finalize_cot_pipeline(base_results, base_raw, judge_solutions, output_dir, benchmark)`:
extract + aggregate + metrics'i tek bir fonksiyonda zincirler. `evalhub cot finalize`
buradan çağrılır.

## Reporting subsystem

### `evalhub/report/scan.py`

`scan_results(results_root)` → `list[RunRecord]`. `rglob("*_summary.json")`
ile tüm summary dosyalarını bulur, parent dir name'i regex ile parse eder.

İki regex çifti:
- `_BASE_DIR_RE`: `<model>_state-<state>_t<T>_max<N>` (yeni canonical)
- `_LEGACY_BASE_DIR_RE`: `<model>_t<T>_max<N>` (state'siz, mevcut layout)
- `_JUDGE_DIR_RE`: `<target>_state-<state>_judged_by_<judge>_state-<state>_t<T>_max<N>` (canonical)
- `_LEGACY_JUDGE_DIR_RE`: `<target>_evaluated_by_<judge>_<max>` (mevcut layout, temp child dir'da)
- `_LEGACY_BENCHMARK_TEMP_RE`: `<benchmark>_t<T>` (child dir formatı)

### `evalhub/report/aggregate.py`

`build_dataframe(records)` → pandas DataFrame (long form). `LONG_COLUMNS`
canonical sütun sıralaması. `write_csv(df, output)` orjson'la performant yazar.

### `evalhub/report/plots.py`

Dört statik plot fonksiyonu (`matplotlib` + `seaborn`):

- `plot_pass_at_k_curves` — Pass@K vs K, her benchmark için ayrı dosya
- `plot_base_vs_cot_bars` — Pass@1 grouped bar, base vs cot
- `plot_pass1_heatmap` — model × benchmark Pass@1 ısı haritası
- `plot_cot_veto_rate` — `cot_false_count / total_generations` bar grafiği

`render_all(df, output_dir, formats)` hepsini sırayla çağırır.

### `evalhub/report/dashboard.py`

Streamlit + Plotly. 7 tab:

| Tab | Görsel |
|---|---|
| Overview | KPI kartları + DataFrame tablo |
| Pass@K | `px.line` her benchmark faceting, base vs cot dash |
| Language comparison | `px.line` benchmark family seçimi, dil rengi, judge facet |
| Pass vs CoT | pivot DataFrame + `px.bar(delta)` grouped bar |
| Heatmap | `px.imshow` model × benchmark, K seçilebilir |
| CoT veto | `px.bar` veto rate |
| Drill-down | JSONL viewer (run seç → file seç → 200 row göster) |

Sidebar filtreleri: eval_type, model, state, benchmark, judge_model, judge_state.

Türetilmiş sütun: `benchmark_family` (`_tr/_pt/...` suffix'i strip eder).

## Utils

### `evalhub/utils/model_state.py`

`MODEL_FAMILIES` listesi: her family (qwen, gemma, ministral) için
`(name, aliases, base_template, non_think_template, think_template)`.
`resolve_template_path(model, state)` → `.jinja` mutlak path.

CLI: `python -m evalhub.utils.model_state --model X --state base --allow-missing`
→ pipeline_common.sh:resolve_template() tarafından çağrılır.

### `evalhub/utils/metrics.py`

`compute_pass_at_k(n, c, k)` — `1 - C(n-c, k) / C(n, k)` formülü.
`get_majority_vote(solutions)` — Counter.most_common(1).

### `evalhub/utils/logger.py`

Rich + logging entegre logger. Tüm modüller burayı kullanır.

### `evalhub/utils/pbar.py`

`get_progress_bar()` Rich progress bar factory.

## Test'ler

| Test | Kapsam |
|---|---|
| `tests/cot/test_extract.py` | extract.py: doğru generation'lar filtreleniyor mu |
| `tests/cot/test_aggregate.py` | aggregate.py: majority vote doğru sayılıyor mu |
| `tests/cot/test_per_task_counts.py` | metrics.py per_task_counts üretiyor mu |
| `tests/cot/test_ids.py` | encode/decode round-trip |
| `tests/cot/test_pipeline.py` | finalize end-to-end |
| `tests/cot/test_cli_integration.py` | `evalhub cot finalize` CLI smoke |
| `tests/report/test_aggregate.py` | report aggregate CSV doğru sütunlar |

```bash
# Tüm testleri çalıştır:
python -m pytest

# Belirli bir dosya:
python -m pytest tests/cot/test_per_task_counts.py -v
```
