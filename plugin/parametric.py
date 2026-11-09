"""Esercizi parametrici.

Un esercizio parametrico è descritto da:
  * random_params : parametri generati a caso, con un intervallo diverso per ogni livello
                    di difficoltà (easy / medium / hard);
  * formulas      : parametri derivati, calcolati in ordine con una formula (expr) e con una
                    versione testuale (text) riutilizzabile nei testi ({formula_<nome>});
  * constraints   : condizioni da rispettare (se non valgono si rigenera);
  * problem_template, intermediate_steps, correct_solution : testi con segnaposto {nome}.

Il modulo contiene anche la verifica delle formule scritte dallo studente (check_formula_answer).

Tutti i calcoli usano Fraction, quindi radici frazionarie e radici quadrate esatte non
hanno errori di arrotondamento. Le formule sono valutate con un interprete AST ristretto
(niente eval).
"""
import ast
import json
import math
import random
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional

LEVELS = ("easy", "medium", "hard")
MAX_ATTEMPTS = 2000


class ExpressionError(Exception):
    """Errore nella valutazione di una formula o di un vincolo."""


# ---------------------------------------------------------------------------
# Numeri
# ---------------------------------------------------------------------------
def to_fraction(value: Any) -> Optional[Fraction]:
    """Converte numeri e stringhe ('3', '-1/2', '0,5', '−4') in Fraction; None se impossibile."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, int):
        return Fraction(value)
    text = str(value).strip().replace("−", "-").replace(",", ".").replace(" ", "")
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def fmt(val: Any) -> str:
    """Formatta un numero: interi senza decimali, frazioni come 'p/q'."""
    if isinstance(val, bool):
        return str(int(val))
    if isinstance(val, Fraction):
        if val.denominator == 1:
            return str(val.numerator)
        return f"{val.numerator}/{val.denominator}"
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val)
    return str(val)


def normalize_level(difficulty: str) -> str:
    d = (difficulty or "easy").strip().lower()
    if d in ("medio", "media", "medium", "intermedio"):
        return "medium"
    if d in ("difficile", "hard", "avanzato"):
        return "hard"
    return "easy"


# ---------------------------------------------------------------------------
# Valutazione sicura di formule e vincoli
# ---------------------------------------------------------------------------
def _exact_sqrt(x: Fraction) -> Fraction:
    if x < 0:
        raise ExpressionError("radice di un numero negativo")
    n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
    if n * n != x.numerator or d * d != x.denominator:
        raise ExpressionError(f"radice non esatta di {fmt(x)}")
    return Fraction(n, d)


def _as_int(x: Fraction) -> int:
    if x.denominator != 1:
        raise ExpressionError("serve un numero intero")
    return x.numerator


def _lcm(*args: Fraction) -> Fraction:
    out = 1
    for a in args:
        v = abs(_as_int(a))
        out = out * v // math.gcd(out, v) if v else 0
    return Fraction(out)


_FUNCS = {
    "abs": lambda x: abs(x),
    "min": lambda *a: min(a),
    "max": lambda *a: max(a),
    "sqrt": _exact_sqrt,
    "num": lambda x: Fraction(x.numerator),
    "den": lambda x: Fraction(x.denominator),
    "gcd": lambda *a: Fraction(math.gcd(*[_as_int(x) for x in a])),
    "lcm": _lcm,
}


def _div(a: Fraction, b: Fraction) -> Fraction:
    if b == 0:
        raise ExpressionError("divisione per zero")
    return a / b


def _mod(a: Fraction, b: Fraction) -> Fraction:
    if b == 0:
        raise ExpressionError("modulo zero")
    return a % b


def _pow(a: Fraction, b: Fraction) -> Fraction:
    if b.denominator != 1 or abs(b.numerator) > 12:
        raise ExpressionError("esponente non ammesso")
    if a == 0 and b < 0:
        raise ExpressionError("divisione per zero")
    return a ** b.numerator


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: _div,
    ast.Mod: _mod,
    ast.Pow: _pow,
}
_CMPOPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _ev(node: ast.AST, ctx: Dict[str, Fraction]) -> Any:
    if isinstance(node, ast.Expression):
        return _ev(node.body, ctx)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        raise ExpressionError("costante non ammessa")
    if isinstance(node, ast.Name):
        if node.id not in ctx:
            raise ExpressionError(f"variabile sconosciuta: {node.id}")
        return ctx[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, ctx)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Not):
            return not v
        raise ExpressionError("operatore unario non ammesso")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError("operatore non ammesso")
        return op(_ev(node.left, ctx), _ev(node.right, ctx))
    if isinstance(node, ast.Compare):
        left = _ev(node.left, ctx)
        for op, comp in zip(node.ops, node.comparators):
            right = _ev(comp, ctx)
            fn = _CMPOPS.get(type(op))
            if fn is None or not fn(left, right):
                if fn is None:
                    raise ExpressionError("confronto non ammesso")
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v, ctx) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise ExpressionError("funzione non ammessa")
        return _FUNCS[node.func.id](*[_ev(a, ctx) for a in node.args])
    raise ExpressionError("espressione non ammessa")


def evaluate_expression(expr: Any, context: Dict[str, Any]) -> Any:
    """Valuta una formula (es. '-a*(x1+x2)') con i valori in context; restituisce Fraction (o bool)."""
    ctx = {}
    for k, v in context.items():
        f = to_fraction(v)
        if f is not None:
            ctx[k] = f
    try:
        tree = ast.parse(str(expr).strip().replace("^", "**"), mode="eval")
        return _ev(tree, ctx)
    except ExpressionError:
        raise
    except SyntaxError as e:
        raise ExpressionError(f"formula non valida '{expr}': {e}")
    except (ValueError, ZeroDivisionError, TypeError, OverflowError) as e:
        raise ExpressionError(f"impossibile valutare '{expr}': {e}")


# ---------------------------------------------------------------------------
# Testi con segnaposto
# ---------------------------------------------------------------------------
_PLACEHOLDER = re.compile(r"\{(\w+)(?::([^{}]+))?\}")


def _format_value(v: Any, spec: str) -> str:
    f = to_fraction(v)
    if f is None:
        return fmt(v)
    neg = f < 0
    mag = fmt(abs(f))
    if spec == "abs":
        return mag
    if spec == "paren":
        return f"({fmt(f)})" if neg else fmt(f)
    if spec == "signed":
        if f == 0:
            return ""
        return f" - {mag}" if neg else f" + {mag}"
    if spec.startswith("lead_"):          # primo termine: 'x²', '-x²', '3x²'
        suf = spec[5:]
        if f == 1:
            return suf
        if f == -1:
            return "-" + suf
        return fmt(f) + suf
    if spec.startswith("term_"):          # termine successivo: ' + 3x', ' - x', '' se zero
        suf = spec[5:]
        if f == 0:
            return ""
        sign = " - " if neg else " + "
        return sign + ("" if abs(f) == 1 else mag) + suf
    return fmt(f)


def render(template: str, context: Dict[str, Any]) -> str:
    """Sostituisce {nome} e {nome:formato} nel testo. Formati: signed, abs, paren, lead_<s>, term_<s>.
    I segnaposto sconosciuti restano invariati."""
    if not template:
        return ""

    def sub(m: "re.Match[str]") -> str:
        name, spec = m.group(1), m.group(2)
        if name not in context:
            return m.group(0)
        v = context[name]
        return _format_value(v, spec) if spec else fmt(v)

    return _PLACEHOLDER.sub(sub, str(template))


# ---------------------------------------------------------------------------
# Generazione
# ---------------------------------------------------------------------------
def _load_json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value if value is not None else default


def _spec_for_level(spec: Any, level: str) -> Any:
    """Se lo spec ha una variante per livello la sceglie (con ripiego sul livello più vicino)."""
    if isinstance(spec, dict) and any(l in spec for l in LEVELS):
        order = list(reversed(LEVELS[:LEVELS.index(level) + 1])) + list(LEVELS)
        for l in order:
            if l in spec:
                return spec[l]
    return spec


def _draw(spec: Any) -> Fraction:
    """Estrae un valore: costante, [min, max], {min,max,denominators,values,exclude}."""
    if isinstance(spec, (int, float, str)):
        f = to_fraction(spec)
        if f is None:
            raise ExpressionError(f"valore non valido: {spec}")
        return f
    if isinstance(spec, list):
        spec = {"min": spec[0], "max": spec[1]} if len(spec) >= 2 else {"values": spec}
    if not isinstance(spec, dict):
        raise ExpressionError(f"specifica non valida: {spec}")
    exclude = [to_fraction(e) for e in spec.get("exclude", [])]
    for _ in range(200):
        if "values" in spec:
            v = to_fraction(random.choice(spec["values"]))
        else:
            lo, hi = to_fraction(spec.get("min", 0)), to_fraction(spec.get("max", 10))
            den = int(random.choice(spec.get("denominators", [1])))
            n = random.randint(math.ceil(lo * den), math.floor(hi * den))
            v = Fraction(n, den)
        if v is not None and v not in exclude:
            return v
    raise ExpressionError("impossibile estrarre un valore valido")


def _constraints_for(constraints: Any, level: str) -> List[str]:
    """Accetta una lista (valida per tutti) o un dict {'all': [...], 'easy': [...], ...}."""
    if isinstance(constraints, list):
        return [str(c) for c in constraints]
    if isinstance(constraints, dict):
        out = [str(c) for c in constraints.get("all", [])]
        out += [str(c) for c in constraints.get(level, [])]
        return out
    return []


def _formulas_of(ex: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Normalizza 'formulas' (e il vecchio 'derived_params') in {nome: {expr, text, text_values}}."""
    raw: Dict[str, Any] = {}
    raw.update(_load_json(ex.get("derived_params"), {}) or {})
    raw.update(_load_json(ex.get("formulas"), {}) or {})
    out: Dict[str, Dict[str, str]] = {}
    for name, f in raw.items():
        if isinstance(f, dict):
            out[name] = {"expr": str(f.get("expr", "")), "text": str(f.get("text", "")),
                         "text_values": str(f.get("text_values", ""))}
        else:
            out[name] = {"expr": str(f), "text": "", "text_values": ""}
    return out


def _generate_values(ex: Dict[str, Any], level: str) -> Dict[str, Fraction]:
    """Estrae i parametri casuali, calcola i derivati e verifica i vincoli (rigenera finché validi)."""
    random_params = _load_json(ex.get("random_params"), {}) or {}
    formulas = _formulas_of(ex)
    constraints = _constraints_for(_load_json(ex.get("constraints"), []), level)
    last_error = "vincoli non soddisfatti"
    for _ in range(MAX_ATTEMPTS):
        try:
            ctx: Dict[str, Fraction] = {}
            deferred: Dict[str, Any] = {}
            for name, spec in random_params.items():
                s = _spec_for_level(spec, level)
                if isinstance(s, dict) and "times" in s:
                    deferred[name] = s       # es. a moltiplicato per i denominatori delle radici
                else:
                    ctx[name] = _draw(s)
            for name, s in deferred.items():
                ctx[name] = _draw(s) * evaluate_expression(s["times"], ctx)
            for name, f in formulas.items():
                ctx[name] = evaluate_expression(f["expr"], ctx)
            if all(evaluate_expression(c, ctx) for c in constraints):
                return ctx
        except ExpressionError as e:
            last_error = str(e)
    raise ExpressionError(f"nessun valore valido dopo {MAX_ATTEMPTS} tentativi ({last_error})")


def _sympy_checks(item: Dict[str, Any], values: Dict[str, Fraction]) -> List[str]:
    """Controlli indipendenti con SymPy (identità da verificare, equazioni da risolvere).
    Non blocca mai la generazione: se SymPy non è disponibile o fallisce, restituisce []."""
    if not (item.get("checks") or item.get("solve_checks")):
        return []
    try:
        from .core import run_checks
        return run_checks(item, values)
    except Exception:
        return []


def materialize(record: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
    """Genera un'istanza concreta dell'esercizio per il livello richiesto."""
    level = normalize_level(difficulty)
    ex = dict(record)
    values = _generate_values(ex, level)
    formulas = _formulas_of(ex)

    context: Dict[str, Any] = dict(values)
    for name, f in formulas.items():          # testi delle formule: {formula_<nome>}
        context[f"formula_{name}"] = render(f["text"], context) if f["text"] else name
    for name, f in formulas.items():          # formule con i valori sostituiti
        if f["text_values"]:
            context[f"formula_{name}_values"] = render(f["text_values"], context)

    ex["parameters"] = {k: fmt(v) for k, v in values.items()}
    ex["checks_failed"] = _sympy_checks(ex, values)
    ex["difficulty"] = level
    ex["problem_statement"] = render(str(ex.get("problem_template") or ex.get("problem_statement") or ""), context)
    ex["correct_solution"] = render(str(ex.get("correct_solution", "")), context)

    steps = []
    for st in _load_json(ex.get("intermediate_steps"), []) or []:
        if not isinstance(st, dict):
            continue
        exprs = st.get("values") if isinstance(st.get("values"), list) else [st.get("value", "")]
        expected = [fmt(evaluate_expression(e, values)) for e in exprs if str(e).strip()]
        hints_tpl = st.get("hints") if isinstance(st.get("hints"), list) else ([st["hint"]] if st.get("hint") else [])
        hints = [render(str(h), context) for h in hints_tpl] or ["Ricontrolla i calcoli."]
        steps.append({
            "description": render(str(st.get("description", "")), context),
            "target": str(exprs[0]).strip() if exprs else "",
            "label": render(str(st.get("label") or ""), context) or (str(exprs[0]).strip() if exprs else ""),
            "value": expected[0] if expected else "",
            "values": expected,
            "unordered": bool(st.get("unordered", False)),
            "hint": hints[0],
            "hints": hints,
        })
    ex["intermediate_steps"] = steps
    ex["mode"] = f"PARAMETRIC_{level.upper()}"
    return ex


# ---------------------------------------------------------------------------
# Risposte dello studente scritte come espressioni ("(-12)^2", "sqrt(144)", "5/2", "2,5")
# ---------------------------------------------------------------------------
def normalize_math_text(text: Any) -> str:
    """Rende leggibile alle formule un testo scritto da una persona (segni unicode, virgola decimale, ...)."""
    t = str(text).strip()
    for a, b in (("\u2212", "-"), ("\u00b7", "*"), ("\u00d7", "*"), ("\u00f7", "/"),
                 ("\u00b2", "**2"), ("\u00b3", "**3")):
        t = t.replace(a, b)
    t = re.sub(r"\u221a\s*\(", "sqrt(", t)
    t = re.sub(r"\u221a\s*(\d+)", r"sqrt(\1)", t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)          # 2,5 -> 2.5
    t = re.sub(r"\)\s*\(", ")*(", t)               # (x-2)(x-4) -> (x-2)*(x-4)
    t = re.sub(r"(\d)\s*\(", r"\1*(", t)            # 2(3+4) -> 2*(3+4)
    t = re.sub(r"(\d)\s*([A-Za-z])", r"\1*\2", t)   # 2x -> 2*x
    return t


def eval_student_number(text: Any) -> Optional[Fraction]:
    """Valuta un'espressione numerica dello studente (senza SymPy). None se non è un numero esatto."""
    t = normalize_math_text(text)
    if not t or len(t) > 120:
        return None
    try:
        v = evaluate_expression(t, {})
    except ExpressionError:
        return None
    return v if isinstance(v, Fraction) else None


# ---------------------------------------------------------------------------
# Formule scritte dallo studente: "b^2 = (-4)^2 = 16", "Δ = b² − 4ac", "x = (-b + √Δ)/(2a)".
# La formula è giusta se, con i valori di QUALSIASI istanza dello stesso esercizio (quella
# attuale e altre generate a caso), dà lo stesso risultato della formula corretta.
# Tutto con Fraction e con l'interprete AST ristretto: nessun eval.
# ---------------------------------------------------------------------------
_FORMULA_FUNCS = ("sqrt", "abs", "min", "max")
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z_]\w*|\*\*|[-+*/()%]|\S")
_WORD_SEP_RE = re.compile(r"\b(?:fa|fanno|vale|valgono|risulta|viene|uguale a|e'|\u00e8)\b", re.I)
_SUB = {ord(c): str(i) for i, c in enumerate("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")}


def _formula_text(text: Any) -> str:
    """Testo scritto da una persona -> testo con soli simboli ASCII (potenze, radici, Δ, pedici...)."""
    t = str(text or "").translate(_SUB)
    for a, b in (("\u2212", "-"), ("\u2013", "-"), ("\u00b7", "*"), ("\u00d7", "*"), ("\u22c5", "*"),
                 ("\u00f7", "/"), ("\u00b2", "**2"), ("\u00b3", "**3"), ("\u00bd", "(1/2)"), ("^", "**")):
        t = t.replace(a, b)
    t = re.sub(r"[\u0394\u03b4](?=[A-Za-z])", "d", t)         # ΔEc -> dEc
    t = t.replace("\u0394", "delta").replace("\u03b4", "delta")
    t = re.sub(r"\u221a\s*\(", "sqrt(", t)                    # √( -> sqrt(
    t = re.sub(r"\u221a\s*([A-Za-z_]\w*|\d+(?:[.,]\d+)?)", r"sqrt(\1)", t)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)                     # 2,5 -> 2.5
    return t


def _split_known(token: str, names) -> Optional[List[str]]:
    """'ac' -> ['a', 'c'] se a e c sono nomi noti (moltiplicazione implicita); None se impossibile."""
    if token in names:
        return [token]
    if len(token) > 14:
        return None
    for i in range(len(token) - 1, 0, -1):
        if token[:i] in names:
            rest = _split_known(token[i:], names)
            if rest is not None:
                return [token[:i]] + rest
    return None


def _to_expression(text: str, names) -> tuple:
    """Converte una formula dello studente in testo valutabile e lista dei nomi usati.
    Aggiunge le moltiplicazioni implicite (4ac -> 4*a*c, 2(x+1) -> 2*(x+1))."""
    vocab = set(names) | set(_FORMULA_FUNCS)
    toks = []
    for tok in _TOKEN_RE.findall(_formula_text(text)):
        if tok[0].isdigit():
            toks.append(("num", tok))
        elif tok[0].isalpha() or tok[0] == "_":
            parts = _split_known(tok, vocab)
            if parts is None:
                raise ExpressionError(f"simbolo sconosciuto: {tok}")
            toks.extend(("func" if p in _FORMULA_FUNCS else "name", p) for p in parts)
        elif tok in ("(", ")"):
            toks.append((tok, tok))
        elif tok in ("**", "+", "-", "*", "/", "%"):
            toks.append(("op", tok))
        else:
            raise ExpressionError(f"carattere non ammesso: {tok}")
    if not toks:
        raise ExpressionError("formula vuota")
    out, used, prev = [], [], None
    for kind, tok in toks:
        if prev is not None:
            pk = prev[0]
            if pk in ("num", "name", ")") and kind in ("num", "name", "func", "("):
                if pk == "num" and kind == "num":
                    raise ExpressionError("due numeri di seguito")
                out.append("*")
        if kind == "name":
            used.append(tok)
        out.append(tok)
        prev = (kind, tok)
    return "".join(out), used


def split_student_answer(text: Any) -> List[str]:
    """'Il passaggio 1 fa b^2 = (-4)^2 = 16' -> ['b^2', '(-4)^2', '16']."""
    t = re.sub(r"\bpassagg\w*\s*(?:n\.?\s*|numero\s*)?\d+", " ", str(text or ""), flags=re.I)
    words = list(_WORD_SEP_RE.finditer(t))
    if words:
        t = t[words[-1].end():]
    elif ":" in t:
        t = t.split(":", 1)[1]
    t = t.replace("==", "=")
    return [p.strip(" \t.;!?") for p in t.split("=") if p.strip(" \t.;!?")]


def answer_label(text: Any) -> str:
    """Etichetta iniziale di una risposta ("Δ = b² - 4ac = 4" -> "delta"); vuota se non c'è un'uguaglianza."""
    parts = split_student_answer(text)
    return _formula_text(parts[0]).strip() if len(parts) > 1 else ""


_ARTICLES = {"il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una", "di", "del", "della", "dello", "dei", "delle", "e"}


def _claim_words(text: Any) -> str:
    """Parole in minuscolo senza articoli e punteggiatura ('Il discriminante Δ' -> 'discriminante delta')."""
    t = _formula_text(text).lower()
    return " ".join(w for w in re.findall(r"[a-z0-9\u00e0-\u00fc]+", t) if w not in _ARTICLES)


def answer_claim(text: Any) -> str:
    """Come lo studente chiama la grandezza che sta calcolando, se lo scrive a parole:
    'il discriminante è b^2 = 25' -> 'discriminante'. Stringa vuota se non lo dice."""
    t = re.sub(r"\bpassagg\w*\s*(?:n\.?\s*|numero\s*)?\d+", " ", str(text or ""), flags=re.I)
    words = list(_WORD_SEP_RE.finditer(t))
    if words:
        prefix = t[:words[-1].start()]
    elif ":" in t:
        prefix = t.split(":", 1)[0]
    else:
        return ""
    return _claim_words(prefix)


def match_claim(claim: str, steps: List[Dict[str, Any]]) -> int:
    """Numero del passaggio (1, 2, ...) il cui nome corrisponde a 'claim'; 0 se nessuno o se è ambiguo."""
    c = _claim_words(claim)
    if len(c) < 3:
        return 0
    hits = []
    for i, st in enumerate(steps, 1):
        names = {_claim_words(st.get("label", "")), _claim_words(st.get("target", ""))}
        if any(n and (c == n or (len(c) >= 4 and c in n.split()) or (len(n) >= 4 and n in c.split())) for n in names):
            hits.append(i)
    return hits[0] if len(hits) == 1 else 0


def _env_of(ex: Dict[str, Any]) -> Dict[str, Fraction]:
    params = ex.get("parameters") or {}
    return {k: f for k, f in ((k, to_fraction(v)) for k, v in params.items()) if f is not None}


def _close(a: Fraction, b: Fraction, tol: float) -> bool:
    return abs(float(a - b)) <= max(tol * abs(float(b)), 1e-9)


def sample_envs(ex: Dict[str, Any], samples: int = 6) -> List[Dict[str, Fraction]]:
    """L'istanza attuale dell'esercizio seguita da altre istanze casuali dello stesso livello."""
    envs = [_env_of(ex)]
    if ex.get("random_params"):
        level = normalize_level(str(ex.get("difficulty") or "easy"))
        for _ in range(samples):
            try:
                envs.append(_generate_values(ex, level))
            except ExpressionError:
                break
    return envs


def step_prerequisites(ex: Dict[str, Any]) -> Dict[int, set]:
    """Per ogni passaggio (1, 2, ...) i passaggi da cui dipende secondo le formule del JSON
    (es. il discriminante dipende da b^2 e da 4ac). Serve a considerare svolti anche quelli."""
    steps = [st for st in (_load_json(ex.get("intermediate_steps"), []) or []) if isinstance(st, dict)]
    formulas = _formulas_of(ex)
    index = {str(st.get("target")): i + 1 for i, st in enumerate(steps) if st.get("target")}

    def deps(name: str, seen: set) -> set:
        out: set = set()
        f = formulas.get(name)
        if not f:
            return out
        for n in re.findall(r"[A-Za-z_]\w*", str(f["expr"])):
            if n in formulas and n not in seen:
                seen.add(n)
                if n in index:
                    out.add(index[n])
                out |= deps(n, seen)
        return out

    return {i + 1: deps(str(st.get("target")), {str(st.get("target"))}) - {i + 1}
            for i, st in enumerate(steps) if st.get("target")}


def check_formula_answer(ex: Dict[str, Any], step: Dict[str, Any], text: Any, tol: float = 0.0,
                         samples: int = 6, shared_envs: Optional[List[Dict[str, Fraction]]] = None) -> Dict[str, Any]:
    """Valuta la risposta dello studente a un passaggio.

    status:
      correct      tutte le parti coincidono con il risultato atteso (has_formula dice se c'è una formula)
      wrong_formula  una formula scritta non è equivalente a quella corretta
      wrong_calc   la formula è giusta ma il calcolo con i numeri (o il risultato finale) è sbagliato
      wrong_value  nessuna formula e il numero è sbagliato
      trivial      ha scritto solo il nome della grandezza
      unreadable   non si riesce a interpretare il testo
    """
    res: Dict[str, Any] = {"status": "unreadable", "has_formula": False}
    env0 = _env_of(ex)
    expected = [f for f in (to_fraction(v) for v in (step.get("values") or [step.get("value")])) if f is not None]
    target = str(step.get("target") or "").strip()
    if not expected:
        return res
    parts = split_student_answer(text)
    if not parts:
        return res

    # altre istanze dello stesso esercizio, per riconoscere le formule equivalenti
    envs, targets = [env0], [expected[0]]
    if target and len(expected) == 1 and ex.get("random_params"):
        pool = shared_envs if shared_envs else sample_envs(ex, samples)
        for e in pool[1:]:
            try:
                targets.append(evaluate_expression(target, e))
                envs.append(e)
            except ExpressionError:
                continue

    names = set(env0)
    parsed = []
    for i, p in enumerate(parts):
        try:
            expr, used = _to_expression(p, names)
        except ExpressionError:
            if i == 0 and len(parts) > 1 and re.fullmatch(r"[^\W\d_][\w]*", p):
                continue                      # etichetta iniziale ("x = ...", "S = ...")
            return res
        if expr in names or (len(used) == 1 and expr == used[0]):
            if len(parts) > 1 and i == 0:
                continue                      # etichetta iniziale ("b2 = 16")
            res["status"] = "trivial"
            return res
        parsed.append((expr, used))
    if not parsed:
        return res

    bad_formula = bad_calc = False
    for expr, used in parsed:
        try:
            v0 = evaluate_expression(expr, env0)
        except ExpressionError:
            return res
        if not isinstance(v0, Fraction):
            return res
        if used:
            res["has_formula"] = True
            ok = any(_close(v0, e, tol) for e in expected)
            for env, tgt in zip(envs[1:], targets[1:]):
                if not ok:
                    break
                try:
                    ok = _close(evaluate_expression(expr, env), tgt, tol)
                except ExpressionError:
                    continue                  # in quell'istanza non è calcolabile: si ignora
            if not ok:
                bad_formula = True
        elif not any(_close(v0, e, tol) for e in expected):
            bad_calc = True
    if bad_formula:
        res["status"] = "wrong_formula"
    elif bad_calc:
        res["status"] = "wrong_calc" if res["has_formula"] else "wrong_value"
    else:
        res["status"] = "correct"
    return res


# ---------------------------------------------------------------------------
# Confronto e validazione
# ---------------------------------------------------------------------------
def numeric_match(student: str, expected_values: List[str], tol: float, unordered: bool) -> bool:
    """True se il valore dello studente (intero, decimale o frazione) coincide con uno atteso."""
    s = to_fraction(student)
    if s is None:
        return False
    for exp in expected_values:
        e = to_fraction(exp)
        if e is not None and abs(float(s - e)) <= max(tol * abs(float(e)), 1e-9):
            return True
    return False


def validate_template(item: Dict[str, Any]) -> List[str]:
    """Controlla un esercizio: campi obbligatori e, se parametrico, che si generi a tutti i livelli."""
    errors: List[str] = []
    if not item.get("exercise_id"):
        errors.append("Manca exercise_id")
    if not item.get("correct_solution"):
        errors.append("Manca correct_solution")
    if str(item.get("type", "")).lower() == "parametric":
        for level in LEVELS:
            try:
                for _ in range(30):
                    ex = materialize(item, level)
                    texts = [ex["problem_statement"], ex["correct_solution"]]
                    for st in ex["intermediate_steps"]:
                        texts += [st["description"]] + st["hints"]
                        if not st["value"]:
                            errors.append(f"[{level}] passaggio senza valore")
                    if any("{" in t for t in texts):
                        errors.append(f"[{level}] segnaposto non risolto")
                        break
                    for f in ex.get("checks_failed", []):
                        errors.append(f"[{level}] {f}")
                    if ex.get("checks_failed"):
                        break
            except ExpressionError as e:
                errors.append(f"[{level}] {e}")
    return errors
