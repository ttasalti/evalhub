# `tubitak_math2026`

TÜBİTAK Matematik Olimpiyatı 2026 — 32 soruluk Türkçe matematik olimpiyat benchmark'ı.

## Veri formatı

`tubitak_math2026.csv` — UTF-8, üç kolon:

| Kolon | İçerik |
|---|---|
| `Question_Number` | 1–32 |
| `Question_Text` | Türkçe soru metni (LaTeX inline math içerebilir, `$...$` ile) |
| `Answer` | Doğru cevap. `$...$` wrapper opsiyonel; loader otomatik strip eder. |

Cevaplar integer + LaTeX karışık: `$2$`, `$\frac{52}{5}$`, `$105^\circ$`, `$3\sqrt{10}$`, `$8^5$`, `$-\frac{49}{8}$` vb. Grading [`evalhub/benchmarks/math/verifier/`](../verifier/) zincirinden geçer (sympy + mathd + dapo), bu form'ların hepsi handle edilir.

## CSV güncelleme

Soru/cevap düzenledikten sonra evalhub'un task cache'i geçersiz olur:

```bash
rm -rf ~/.cache/evalhub/
```

Bir sonraki `evalhub gen` veya `evalhub eval` CSV'yi tekrar okur.

## Pass@K kullanımı

```bash
evalhub gen --model hosted_vllm/Qwen/Qwen3.5-0.8B-Base --tasks tubitak_math2026 \
    --temperature 0.6 --n-samples 8 --output-dir results/tubitak/
evalhub eval --tasks tubitak_math2026 \
    --solutions results/tubitak/tubitak_math2026.jsonl --output-dir results/tubitak/
```

## CoT-Pass@K kullanımı

Türkçe judge prompt (`cot_judge_tr`) zorunlu:

```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/tubitak_math2026.env \
    --model X --judge Y
# JUDGE_TASK=cot_judge_tr config dosyasında set edilmiş durumda
```

Veya `base.env` ile dinamik:

```bash
scripts/submit.sh scripts/run_end_to_end.sh scripts/configs/base.env \
    --model X --judge Y --benchmark tubitak_math2026 \
    --set JUDGE_TASK=cot_judge_tr
```
