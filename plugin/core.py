"""Funzioni basate su SymPy.

Usi:
  * eval_number / equivalent : interpretare e confrontare le risposte scritte dallo studente
    (numeri, frazioni, radici, espressioni algebriche equivalenti);
  * run_checks              : verifiche indipendenti sugli esercizi generati (identità che devono
    valere, equazioni che SymPy risolve e che devono dare le soluzioni attese).

L'input dello studente passa SEMPRE da safe_parse: lunghezza limitata, solo caratteri aritmetici,
solo simboli di una lettera (con al più una cifra) e le funzioni sqrt/abs.
Le espressioni degli esercizi (campi 'checks' e 'solve_checks' del JSON) sono scritte dal docente.
"""
import re
import string
from fractions import Fraction
from typing import Any, Dict, List, Optional

import sympy as sp

from .parametric import normalize_math_text

_SYMBOL_RE = re.compile(r"^[A-Za-z]\d?$")
_CHARS_RE = re.compile(r"^[0-9A-Za-z_+\-*/^().\s]*$")
_UNDERSCORES_RE = re.compile(r"_{2}")
_ALLOWED_FUNCS = ("sqrt", "abs")


def _namespace() -> Dict[str, Any]:
    """Le lettere maiuscole (E, I, S, N, Q, O) hanno un significato in SymPy: qui sono simboli."""
    ns: Dict[str, Any] = {c: sp.Symbol(c) for c in string.ascii_letters}
    ns["sqrt"] = sp.sqrt
    ns["abs"] = sp.Abs
    return ns


def safe_parse(text: Any):
    """Converte il testo dello studente in un'espressione SymPy; None se non è ammesso o non è valido."""
    t = normalize_math_text(text)
    if not t or len(t) > 120 or not _CHARS_RE.match(t) or _UNDERSCORES_RE.search(t):
        return None
    for name in re.findall(r"[A-Za-z_]\w*", t):
        if name not in _ALLOWED_FUNCS and not _SYMBOL_RE.match(name):
            return None
    try:
        return sp.sympify(t, locals=_namespace(), rational=True)
    except Exception:
        return None


def eval_number(text: Any) -> Optional[Fraction]:
    """Valore esatto di un'espressione numerica ('(-12)^2', 'sqrt(144)', '5/2', '2,5'); None se non è razionale."""
    expr = safe_parse(text)
    if expr is None or not getattr(expr, "is_Rational", False):
        return None
    return Fraction(int(expr.p), int(expr.q))


def equivalent(expected: Any, student: Any) -> Optional[bool]:
    """True se le due espressioni sono algebricamente equivalenti (es. x**2-6x+8 e (x-2)(x-4)).
    None se una delle due non si può interpretare."""
    a, b = safe_parse(expected), safe_parse(student)
    if a is None or b is None:
        return None
    try:
        return bool(sp.simplify(a - b) == 0)
    except Exception:
        return None


def _rational_locals(values: Dict[str, Fraction]) -> Dict[str, Any]:
    return {k: sp.Rational(v.numerator, v.denominator) for k, v in values.items()}


def run_checks(item: Dict[str, Any], values: Dict[str, Fraction]) -> List[str]:
    """Esegue i controlli dell'esercizio sui valori generati. Restituisce la lista degli errori (vuota se ok).

    'checks'       : espressioni che devono valere 0, es. "W - dEc".
    'solve_checks' : {"equation": "a*x**2 + b*x + c", "variable": "x", "expected": ["x1", "x2"]};
                     SymPy risolve equation = 0 e le soluzioni devono coincidere con i valori attesi.
    """
    errors: List[str] = []
    loc = _rational_locals(values)
    for chk in item.get("checks") or []:
        try:
            res = sp.simplify(sp.sympify(str(chk), locals=dict(loc), rational=True))
            if res != 0:
                errors.append(f"controllo fallito: {chk} = {res}")
        except Exception as e:
            errors.append(f"controllo non valutabile: {chk} ({e})")
    for sc in item.get("solve_checks") or []:
        try:
            var = sp.Symbol(str(sc.get("variable", "x")))
            ns = dict(loc)
            ns[str(var)] = var
            poly = sp.sympify(str(sc["equation"]), locals=ns, rational=True)
            found = set(sp.solve(sp.Eq(poly, 0), var))
            wanted = {sp.Rational(values[n].numerator, values[n].denominator) for n in sc["expected"]}
            if found != wanted:
                errors.append(f"soluzioni di {sc['equation']} = 0: trovate {sorted(map(str, found))}, attese {sorted(map(str, wanted))}")
        except Exception as e:
            errors.append(f"controllo di risoluzione non valutabile: {sc} ({e})")
    return errors


def check_answer(problem: str, user_answer: str) -> Dict[str, Any]:
    """Compatibilità: equivalenza tra la risposta attesa e quella dello studente."""
    res = equivalent(problem, user_answer)
    if res is None:
        return {"ok": False, "error": "espressione non interpretabile"}
    return {"ok": True, "result": {"ok_answer": res}}
