"""Shared helpers for the error-injection study.

Two concerns live here:

1. Boxed-answer / number utilities and the **wrong-number menu**
   (pm12 / digit_swap / single_digit) with the post-checks and
   fallback-with-note behaviour.
2. The four **error-injection primitives** (type1..type4) that operate on a
   solution's full text.

Correctness grading and question/GT loading are done through evalhub itself
(imported by the caller) so this module has no heavy dependencies and is unit
testable on its own.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from error_study import config

_INT_RE = re.compile(r"\d+")


# boxed helpers
def find_last_boxed_span(text: str) -> tuple[int, int, str] | None:
    r"""Return (start, end, inner) of the LAST ``\boxed{...}`` (or ``\fbox``).

    ``start`` indexes the backslash of ``\boxed``; ``end`` is one past the
    closing brace; ``inner`` is the brace content. ``None`` if no box.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (idx, j + 1, text[i + 1 : j])
    return None


def _normalize(s: str) -> str:
    return s.strip().replace("\\text", "").replace("\\!", "").replace("\\,", "").replace("$", "").replace(" ", "")


def set_last_boxed_inner(text: str, new_inner: str) -> str | None:
    r"""Replace the inner content of the last ``\boxed{...}`` with ``new_inner``."""
    span = find_last_boxed_span(text)
    if span is None:
        return None
    start, end, _ = span
    return text[:start] + "\\boxed{" + new_inner + "}" + text[end:]


# number extraction
def int_tokens(text: str) -> list[tuple[int, int, int, str]]:
    """All maximal digit-runs as (value, start, end, raw)."""
    return [(int(m.group()), m.start(), m.end(), m.group()) for m in _INT_RE.finditer(text)]


def question_numbers(question_text: str) -> set[int]:
    """Set of integer values that appear in the question (forbidden targets)."""
    return {int(m.group()) for m in _INT_RE.finditer(question_text)}


def is_label_number(text: str, st: int, en: int) -> bool:
    """True if the digit-run at [st, en) is a non-computational label rather than a
    real intermediate value, an ordinal (2nd, 3rd, 21st, 4th) or a subscript /
    variable index (s_1, z_{2}). Such tokens make awkward intermediate errors
    ("2nd quadrant" -> "7nd quadrant"), so type 2 skips them."""
    if text[en : en + 2].lower() in ("st", "nd", "rd", "th"):
        return True
    prev = text[st - 1] if st > 0 else ""
    if prev == "_":  # s_1 , z_12
        return True
    if prev == "{" and st >= 2 and text[st - 2] == "_":  # s_{1}
        return True
    return False


def has_embedded_occurrence(text: str, value_str: str) -> bool:
    """True if ``value_str`` (a digit string) ever appears as part of a LARGER
    number (a digit on either side), the type-3 'answer inside 125' skip case."""
    for m in re.finditer(re.escape(value_str), text):
        left = text[m.start() - 1] if m.start() > 0 else ""
        right = text[m.end()] if m.end() < len(text) else ""
        if left.isdigit() or right.isdigit():
            return True
    return False


def boundary_safe_replace_int(text: str, value_str: str, repl: str) -> tuple[str, int, int]:
    """Replace every occurrence of the integer ``value_str`` that is NOT part of a
    larger number. Returns (new_text, n_replaced, first_replacement_offset) where
    the offset is the position in ``new_text`` of the FIRST actual replacement
    (-1 if none), so callers record a true change site, not a coincidental
    pre-existing occurrence of the replacement string."""
    out = []
    n = 0
    first_off = -1
    cur = 0  # length of output built so far
    i = 0
    value_len = len(value_str)
    while i < len(text):
        if text.startswith(value_str, i):
            left = text[i - 1] if i > 0 else ""
            right = text[i + value_len] if i + value_len < len(text) else ""
            if not left.isdigit() and not right.isdigit():
                if first_off < 0:
                    first_off = cur
                out.append(repl)
                cur += len(repl)
                i += value_len
                n += 1
                continue
        out.append(text[i])
        cur += 1
        i += 1
    return "".join(out), n, first_off


# answer-type classification
def classify_answer_type(gt: str) -> str:
    a = gt.strip()
    if re.fullmatch(r"-?\d+", a):
        return "int"
    if re.fullmatch(r"-?\d+[.,]\d+", a):
        return "decimal"
    if re.fullmatch(r"[A-Ea-e]", a):
        return "mc"
    if "\\frac" in a or re.search(r"\d+\s*/\s*\d+", a):
        return "frac"
    if "\\sqrt" in a:
        return "sqrt"
    if a.endswith("%") or "\\%" in a:
        return "percent"
    if re.search(r"[a-zA-Z]", a) or "^" in a or "\\" in a:
        return "expr"
    return "other"


# wrong-number menu (integer primitive)
def _pm12_candidates(n: int, rng: random.Random) -> list[int]:
    cands = [n - 1, n + 1, n - 2, n + 2]
    rng.shuffle(cands)
    return [c for c in cands if c != n]


def _digit_swap_candidates(n: int, rng: random.Random) -> list[int]:
    """Transpose two distinct-valued digit positions (2-digit => reverse, e.g.
    47->74). No leading zero, result != n. When config.DIGIT_SWAP_STRICT_2DIGIT is
    True, only 2-digit numbers qualify (literal spec reading) and 3+ digit numbers
    fall through to the next menu category; when False (default), any-length
    transposition is allowed."""
    s = str(abs(n))
    if config.DIGIT_SWAP_STRICT_2DIGIT:
        if len(s) != 2:
            return []
    elif len(s) < 2:
        return []
    out: set[int] = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[i] == s[j]:
                continue
            lst = list(s)
            lst[i], lst[j] = lst[j], lst[i]
            if lst[0] == "0":
                continue
            v = int("".join(lst))
            if v != abs(n):
                out.add(v)
    res = [(-v if n < 0 else v) for v in out]
    rng.shuffle(res)
    return res


def _single_digit_candidates(n: int, rng: random.Random) -> list[int]:
    """Change exactly one digit to a different digit (e.g. 392->382). Keep length,
    no leading zero for multi-digit."""
    s = str(abs(n))
    out: set[int] = set()
    for i in range(len(s)):
        for d in "0123456789":
            if d == s[i]:
                continue
            lst = list(s)
            lst[i] = d
            if len(s) > 1 and lst[0] == "0":
                continue
            v = int("".join(lst))
            if v != abs(n):
                out.add(v)
    res = [(-v if n < 0 else v) for v in out]
    rng.shuffle(res)
    return res


_GEN = {
    "pm12": _pm12_candidates,
    "digit_swap": _digit_swap_candidates,
    "single_digit": _single_digit_candidates,
}


def corrupt_integer(
    n: int,
    primary: str,
    check: Callable[[int], bool],
    rng: random.Random,
) -> tuple[int, str, bool] | None:
    """Produce a wrong integer for ``n`` using the menu.

    Tries ``primary`` category first; on infeasibility falls back through
    ``MENU_ORDER``. Returns (new_value, category_used, is_fallback) or None if no
    category yields a value passing ``check``.
    """
    order = [primary] + [c for c in config.MENU_ORDER if c != primary]
    for k, cat in enumerate(order):
        for cand in _GEN[cat](n, rng):
            if check(cand):
                return cand, cat, (k > 0)
    return None


def corrupt_answer_value(
    gt: str,
    answer_type: str,
    is_aime: bool,
    qnums: set[int],
    primary: str,
    rng: random.Random,
) -> dict | None:
    """Compute ONE wrong-answer value (shared by type1 & type3).

    Returns dict(wrong_str, old_token, new_token, category, fallback) or None
    (=> caller skips types 1&3 with a note). Integer answers use the whole value;
    other numeric types corrupt one digit-token inside the expression, keeping the
    type (e.g. \\frac{52}{5} -> \\frac{53}{5}).
    """
    if answer_type == "int":
        n = int(gt)

        def chk(c: int) -> bool:
            if c == n or c in qnums:
                return False
            if n >= 0 and c < 0:  # forbid negative wrong-numbers only for non-negative answers
                return False
            if is_aime and not (config.AIME_MIN <= c <= config.AIME_MAX):
                return False
            return True

        r = corrupt_integer(n, primary, chk, rng)
        if r is None:
            return None
        new, cat, fb = r
        return {"wrong_str": str(new), "old_token": str(n), "new_token": str(new), "category": cat, "fallback": fb}

    # non-integer numeric: corrupt one integer token inside the expression
    toks = [(m.group(), m.start(), m.end()) for m in _INT_RE.finditer(gt)]
    if not toks:
        return None  # pure symbolic / MC -> skip types 1&3
    order = list(range(len(toks)))
    rng.shuffle(order)
    for ti in order:
        raw, st, en = toks[ti]
        n = int(raw)

        def chk(c: int, _n=n) -> bool:
            return c != _n and c not in qnums and c >= 0

        r = corrupt_integer(n, primary, chk, rng)
        if r is None:
            continue
        new, cat, fb = r
        wrong = gt[:st] + str(new) + gt[en:]
        if wrong == gt:
            continue
        return {"wrong_str": wrong, "old_token": raw, "new_token": str(new), "category": cat, "fallback": fb}
    return None


# error-injection primitives  (each returns a dict describing the change)
def apply_type1(content: str, gt: str, w: dict) -> dict | None:
    r"""Type 1, change ONLY the number inside the final \boxed{}."""
    span = find_last_boxed_span(content)
    if span is None:
        return {"skip_reason": "no_boxed_in_solution"}
    start, end, inner = span
    if _normalize(inner) == _normalize(gt):
        new_inner = w["wrong_str"]
    elif w["old_token"] in inner:
        new_inner = inner.replace(w["old_token"], w["new_token"], 1)
    elif gt in inner:
        new_inner = inner.replace(gt, w["wrong_str"], 1)
    else:
        new_inner = w["wrong_str"]
    new_content = content[:start] + "\\boxed{" + new_inner + "}" + content[end:]
    return {
        "corrupted_solution": new_content,
        "old_number": w["old_token"],
        "new_number": w["new_token"],
        "change_location_char": start,
        "change_location_pct": round(100 * start / max(1, len(content)), 1),
    }


def apply_type3(content: str, gt: str, answer_type: str, w: dict, qnums: set[int]) -> dict | None:
    r"""Type 3, hidden consistent: change EVERY occurrence of the real answer
    (boxed + body) to the SAME wrong value as type1."""
    if answer_type == "int":
        n = int(gt)
        if n in qnums:
            return {"skip_reason": "answer_equals_question_number"}
        if has_embedded_occurrence(content, gt):
            return {"skip_reason": "answer_inside_larger_number"}
        new_content, n_occ, first = boundary_safe_replace_int(content, gt, w["wrong_str"])
    else:
        if gt not in content:
            return {"skip_reason": "answer_string_not_found"}
        n_occ = content.count(gt)
        new_content = content.replace(gt, w["wrong_str"])
        first = content.find(gt)  # first occurrence == first replacement site (prefix unchanged)
    if n_occ == 0:
        return {"skip_reason": "answer_string_not_found"}
    return {
        "corrupted_solution": new_content,
        "old_number": w["old_token"],
        "new_number": w["new_token"],
        "change_location_char": first,
        "change_location_pct": round(100 * first / max(1, len(content)), 1),
        "num_occurrences_replaced": n_occ,
    }


def apply_type2(
    content: str,
    gt: str,
    answer_type: str,
    qnums: set[int],
    primary: str,
    rng: random.Random,
) -> dict | None:
    r"""Type 2, intermediate slip: in the 40–70% region change an existing
    number that is not in the question, not the answer, and not 0/1."""
    length = len(content)
    lo = int(config.TYPE2_REGION[0] * length)
    hi = int(config.TYPE2_REGION[1] * length)
    # answer's own numeric tokens (forbidden as the intermediate target)
    answer_ints = {int(m.group()) for m in _INT_RE.finditer(gt)}
    if answer_type == "int":
        answer_ints.add(int(gt))
    # tokenize the FULL text and keep only whole tokens fully inside the region,
    # so a digit-run straddling the 40%/70% boundary is never partially captured.
    cands = [
        (val, st, en, raw)
        for (val, st, en, raw) in int_tokens(content)
        if st >= lo
        and en <= hi
        and val not in qnums
        and val not in answer_ints
        and val not in (0, 1)
        and not is_label_number(content, st, en)
    ]
    if not cands:
        return {"skip_reason": "no_intermediate_number_in_region"}
    rng.shuffle(cands)
    for val, abs_st, abs_en, raw in cands:

        def chk(c: int, _v=val) -> bool:
            return c != _v and c not in qnums and c not in answer_ints and c not in (0, 1) and c >= 0

        r = corrupt_integer(val, primary, chk, rng)
        if r is None:
            continue
        new, cat, fb = r
        new_content = content[:abs_st] + str(new) + content[abs_en:]
        return {
            "corrupted_solution": new_content,
            "old_number": raw,
            "new_number": str(new),
            "change_location_char": abs_st,
            "change_location_pct": round(100 * abs_st / max(1, length), 1),
            "intermediate_menu": cat + ("(fallback)" if fb else ""),
            "wrong_number_menu": cat + ("(fallback)" if fb else ""),
        }
    return {"skip_reason": "no_valid_wrong_number_for_intermediate"}


def apply_type4(content: str) -> dict | None:
    r"""Type 4, truncate the last 25% but keep the final \boxed{} and its number."""
    length = len(content)
    cut = int(config.TYPE4_KEEP_FRACTION * length)
    span = find_last_boxed_span(content)
    if span is None:
        return {"skip_reason": "no_boxed_in_solution"}
    b_start, b_end, _ = span
    if b_end > cut:
        # any part of the boxed falls in the removed tail -> keep text up to the
        # box start (drop any sliced fragment) and re-append the full boxed.
        truncated = content[: min(cut, b_start)].rstrip() + "\n" + content[b_start:b_end]
    else:
        # the boxed already survives fully inside the kept 75%
        truncated = content[:cut]
    return {
        "corrupted_solution": truncated,
        "change_location_char": cut,
        "change_location_pct": round(100 * config.TYPE4_KEEP_FRACTION, 1),
    }
