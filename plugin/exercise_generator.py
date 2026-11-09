import inspect
import json
import os
import random
import re
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from fastapi import UploadFile
from starlette.responses import StreamingResponse

from cat import (
    AuthPermission,
    AuthResource,
    AuthorizedInfo,
    StrayCat,
    check_permissions,
    endpoint,
    hook,
    tool,
)
from cat.db.cruds.settings import crud as crud_settings

try:
    from cat import log
except Exception:  # pragma: no cover
    import logging
    log = logging.getLogger("exercise_generator")

from .constants import FILENAME_EXERCISES, FILENAME_TEMPLATE, PLUGIN_DIR, PLUGIN_ID
from .models import ExerciseDataManager, ExerciseTemplate
from .parametric import (
    ExpressionError,
    answer_label,
    answer_claim,
    check_formula_answer,
    eval_student_number,
    fmt,
    match_claim,
    materialize,
    normalize_level,
    numeric_match,
    render,
    sample_envs,
    step_prerequisites,
    validate_template,
)
try:  # SymPy: se manca o fallisce, il plugin continua con il calcolo esatto interno
    from .core import equivalent as _sp_equivalent, eval_number as _sp_number
except Exception:  # pragma: no cover
    _sp_equivalent = None
    _sp_number = None


_NESTED_KEYS = (
    "difficulty", "difficulty_levels", "random_params", "derived_params", "formulas", "formula_texts", "constraints",
    "intermediate_steps_template", "intermediate_steps", "parameters", "solution_exprs",
)


def _get_file_path(cat: StrayCat) -> str:
    return os.path.join(cat.agent_key, "plugins", PLUGIN_ID)


def _get_db_key(cat: StrayCat) -> str:
    return f"plugins:{PLUGIN_ID}:{cat.agent_key}:{cat.id}"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return fmt(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _jsonable(obj.item())
        except Exception:
            pass
    return obj


async def _store_active_exercise(cat: StrayCat, exercise: Dict[str, Any]) -> None:
    await crud_settings.store(_get_db_key(cat), _jsonable(exercise))


def _unwrap_state(raw: Any) -> Dict[str, Any] | None:
    """Normalizza quanto restituito da crud_settings.read in un dict dell'esercizio.

    Accetta: dict, wrapper {"value": ...}, lista di dict, lista di coppie (chiave, valore).
    """
    for _ in range(6):
        if isinstance(raw, list):
            if raw and all(isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str) for x in raw):
                raw = {k: v for k, v in raw}
                continue
            dicts = [x for x in raw if isinstance(x, (dict, list))]
            raw = dicts[-1] if dicts else None
        elif isinstance(raw, dict) and "intermediate_steps" not in raw and isinstance(raw.get("value"), (dict, list)):
            raw = raw["value"]
        else:
            break
    return raw if isinstance(raw, dict) else None


async def _get_current_exercise(cat: StrayCat) -> Dict[str, Any]:
    raw = await crud_settings.read(_get_db_key(cat))
    ex = _unwrap_state(raw)
    if not ex:
        if raw:
            log.warning(f"exercise_generator: stato esercizio non riconosciuto ({type(raw)}): {str(raw)[:300]}")
        raise ValueError("Nessun esercizio attivo. Avviane uno con start_exercise.")
    return ex


def _load_dataset(cat: StrayCat):
    """Legge exercises.json dal file manager del Cat (data/storage/<agente>/plugins/<plugin>/),
    con ripiego sulla cartella del plugin. Non usa la funzione open vietata."""
    content = cat.file_manager.read_file(FILENAME_EXERCISES, _get_file_path(cat))
    if not content:
        local = Path(PLUGIN_DIR) / FILENAME_EXERCISES
        if local.exists():
            content = local.read_bytes()
    if content:
        if isinstance(content, str):
            content = content.encode("utf-8")
        return ExerciseDataManager.load_data(content)
    return None


async def _get_tolerance(cat: StrayCat) -> float:
    try:
        settings = cat.mad_hatter.get_plugin().load_settings()
        if inspect.isawaitable(settings):
            settings = await settings
        if isinstance(settings, dict):
            return float(settings.get("tolerance", 0.01))
    except Exception:
        pass
    return 0.01


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items() if x is not None and not (isinstance(x, float) and x != x)}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "tolist"):
        return _clean(v.tolist())
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except Exception:
            pass
    return v


def _as_step_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = ExerciseDataManager.parse_steps(raw)
    return [s for s in (raw or []) if isinstance(s, dict)]


def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec = _clean(dict(rec))
    for key in _NESTED_KEYS:
        val = rec.get(key)
        if isinstance(val, str) and val.strip()[:1] in ("{", "["):
            try:
                rec[key] = json.loads(val)
            except Exception:
                pass
    return rec


def _prepare_static(rec: Dict[str, Any]) -> Dict[str, Any]:
    ex = dict(rec)
    params = ex.get("parameters") or {}
    display = {k: fmt(v) for k, v in params.items()} if isinstance(params, dict) else {}

    tpl = str(ex.get("problem_template") or "")
    ex["problem_statement"] = ex.get("problem_statement") or (render(tpl, display) if tpl else "")
    ex["correct_solution"] = render(str(ex.get("correct_solution", "")), display)

    steps = []
    for st in _as_step_list(ex.get("intermediate_steps")):
        value = render(str(st.get("value", "")), display).strip()
        hint = render(str(st.get("hint") or ""), display).strip()
        steps.append({
            "description": render(str(st.get("description", "")), display),
            "value": value,
            "values": [value] if value else [],
            "unordered": bool(st.get("unordered", False)),
            "hint": hint or "Ricontrolla i calcoli.",
            "hints": [hint] if hint else [],
        })
    ex["intermediate_steps"] = steps
    ex["mode"] = "EXACT"
    return ex


def _new_progress() -> Dict[str, Any]:
    # mode: "free" = lo studente risolve come vuole e scrive anche solo una parte; "guided" = passo passo
    return {"current_step": 1, "hint_level": 0, "wrong_attempts": 0, "mode": "free", "done": []}


def _done_of(prog: Dict[str, Any], steps: List[Dict[str, Any]]) -> set:
    """Passaggi (1, 2, ...) già svolti. Per gli stati salvati prima di questa versione si usa current_step."""
    if isinstance(prog.get("done"), list):
        return {int(x) for x in prog["done"]}
    return set(range(1, int(prog.get("current_step", 1))))


def _next_step(steps: List[Dict[str, Any]], done: set) -> int:
    for i in range(1, len(steps) + 1):
        if i not in done:
            return i
    return len(steps) + 1


def _label(step: Dict[str, Any]) -> str:
    return str(step.get("label") or step.get("target") or "").strip()


def _labels(steps: List[Dict[str, Any]], idxs) -> str:
    return ", ".join(_label(steps[i - 1]) for i in sorted(idxs) if 0 < i <= len(steps))


def _mark_done(prog: Dict[str, Any], steps: List[Dict[str, Any]], idxs) -> set:
    done = _done_of(prog, steps)
    new = set(idxs) - done
    done |= set(idxs)
    prog["done"] = sorted(done)
    prog["current_step"] = _next_step(steps, done)
    if new:
        prog["hint_level"] = 0
        prog["wrong_attempts"] = 0
    return new


async def _activate(cat: StrayCat, record: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
    rec = _normalize_record(record)
    if str(rec.get("type", "static")).lower() == "parametric":
        ex = materialize(rec, difficulty)
    else:
        ex = _prepare_static(rec)
    ex["progress"] = _new_progress()
    if not (ex.get("random_params") and all(st.get("target") for st in _steps_of(ex))):
        ex["progress"]["mode"] = "guided"       # senza obiettivi per passaggio si può solo procedere a passi
    await _store_active_exercise(cat, ex)
    if not _unwrap_state(await crud_settings.read(_get_db_key(cat))):
        raise RuntimeError("Esercizio generato ma non salvato nello stato attivo.")
    return ex


def _step_presentation(ex: Dict[str, Any], n: int) -> str:
    steps = _steps_of(ex)
    return (f"Passaggio {n}/{len(steps)}: {steps[n - 1].get('description', '')}\n\n"
            f"Istruzioni per te: presenta questo passaggio e chiedi la FORMULA, poi i valori sostituiti e il risultato "
            f"(formula = valori = risultato). NON rivelare valori né la soluzione, NON parlare di suggerimenti.")


def _intro(ex: Dict[str, Any], origin: str) -> str:
    head = f"--- ESERCIZIO ATTIVATO ({origin}) ---\n{ex.get('problem_statement', '')}\n\n"
    if (ex.get("progress") or {}).get("mode") == "guided":
        return head + _step_presentation(ex, 1)
    return (head +
            "Modalità LIBERA: lo studente risolve come preferisce e può scrivere anche solo una parte della soluzione.\n\n"
            "Istruzioni per te: presenta allo studente SOLO il testo dell'esercizio e invitalo a scrivere ciò che sa fare: "
            "formule, valori sostituiti e risultati, anche parziali (nella forma formula = valori = risultato). "
            "NON mostrare né anticipare passaggi, formule o valori, NON fare esempi con i numeri di questo esercizio, "
            "NON parlare di suggerimenti. Puoi aggiungere che, se preferisce, può chiederti di guidarlo passo passo.")


async def _start_from_dataset(cat: StrayCat, topic: str, difficulty: str) -> str:
    df = _load_dataset(cat)
    if df is None or df.empty:
        return f"Errore: dataset {FILENAME_EXERCISES} non trovato o vuoto."

    topics = sorted({str(t) for t in df["topic"].to_list()})
    if topic:
        wanted = topic.strip().lower().replace(" ", "_")
        col = df["topic"].astype(str).str.lower().str.replace(" ", "_")
        filtered = df[col == wanted]
        if filtered.empty:
            # corrispondenza parziale (es. "equazioni" -> "equazioni_secondo_grado")
            filtered = df[col.apply(lambda t: wanted in t or t in wanted)]
        if filtered.empty and len(topics) == 1:
            # un solo argomento disponibile: usalo
            filtered = df
        if filtered.empty:
            return f"Nessun esercizio per il topic '{topic}'. Argomenti disponibili: {', '.join(topics)}."
        df = filtered

    # preferisci il livello richiesto, se presente nel dataset
    if "difficulty" in df.columns:
        by_level = df[df["difficulty"].astype(str).str.lower() == str(difficulty).lower()]
        if not by_level.empty:
            df = by_level

    ex_id = str(df.sample(1).iloc[0]["exercise_id"])
    record = ExerciseDataManager.get_exercise_by_id(df, ex_id)
    ex = await _activate(cat, record, difficulty)
    return _intro(ex, f"dataset {ex_id}")


async def _start_by_id(cat: StrayCat, exercise_id: str, difficulty: str) -> str:
    df = _load_dataset(cat)
    if df is None:
        return f"Errore: dataset {FILENAME_EXERCISES} non trovato."
    clean_id = str(exercise_id).strip().upper()
    try:
        record = ExerciseDataManager.get_exercise_by_id(df, clean_id)
        if record is None:
            record = ExerciseDataManager.get_exercise_by_id(df, str(exercise_id).strip())
    except Exception:
        record = None
    if record is None:
        return f"Impossibile trovare l'esercizio '{clean_id}' nel dataset."
    ex = await _activate(cat, record, difficulty)
    return _intro(ex, f"dataset {clean_id}")


def _validate_generated(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("JSON non è un oggetto")
    if not str(data.get("problem_statement", "")).strip():
        raise ValueError("manca problem_statement")
    if not str(data.get("correct_solution", "")).strip():
        raise ValueError("manca correct_solution")
    steps = _as_step_list(data.get("intermediate_steps"))
    if not steps:
        raise ValueError("mancano intermediate_steps")
    norm = []
    for st in steps:
        value = str(st.get("value", "")).strip()
        hint = str(st.get("hint") or "").strip()
        if not value:
            raise ValueError("uno step non ha 'value'")
        norm.append({
            "description": str(st.get("description", "")),
            "value": value, "values": [value], "unordered": False,
            "hint": hint or "Ricontrolla i calcoli.", "hints": [hint] if hint else [],
        })
    data["intermediate_steps"] = norm
    return data


async def _start_ai_generated(cat: StrayCat, topic: str) -> str:
    df = _load_dataset(cat)
    topics: List[str] = []
    if df is not None and not df.empty:
        topics = sorted({str(t) for t in df["topic"].dropna().to_list()})
    if topic and topics and topic.lower() not in [t.lower() for t in topics]:
        return f"Topic '{topic}' non presente. Topic disponibili: {', '.join(topics)}"
    if not topic and topics:
        topic = random.choice(topics)

    sample = ""
    if df is not None and not df.empty:
        sample = f"Esempio di stile: {df.sample(1).iloc[0].get('problem_template')}"

    prompt = f"""Sei un docente di matematica. Crea UN esercizio originale.
Argomento: {topic or 'matematica'}
{sample}

Rispondi SOLO con un oggetto JSON valido, senza testo prima o dopo:
{{"exercise_id": "AI_GEN", "topic": "{topic or 'matematica'}",
  "problem_statement": "testo con dati numerici",
  "correct_solution": "risultato finale",
  "intermediate_steps": [
    {{"description": "passaggio 1", "value": "risultato numerico 1", "hint": "suggerimento 1"}},
    {{"description": "passaggio 2", "value": "risultato numerico 2", "hint": "suggerimento 2"}}
  ]}}"""
    try:
        raw = await _llm_call(cat, prompt)
        match = re.search(r"\{[\s\S]*\}", str(raw).replace("```json", "").replace("```", ""))
        if not match:
            raise ValueError("nessun JSON nella risposta del modello")
        data = _validate_generated(json.loads(match.group(0)))
        data["mode"] = "GENERATED"
        data["verified"] = False
        data["progress"] = _new_progress()
        await _store_active_exercise(cat, data)
        return _intro(data, "generato dall'AI, non verificato")
    except Exception as e:
        log.warning(f"generate_ai_exercise fallita: {e}")
        return f"Errore nella generazione con l'LLM ({e})."


_SAFE_INPUT = re.compile(r"[\w\s+\-*/^().,;=]*")


def _symbolic_ok(expected: str, student: str) -> bool:
    """Equivalenza algebrica con SymPy (es. '(x-2)(x-4)' e 'x^2-6x+8')."""
    if not expected or _sp_equivalent is None:
        return False
    try:
        return bool(_sp_equivalent(expected, student))
    except Exception as e:
        log.warning(f"exercise_generator: confronto simbolico fallito: {e}")
        return False


def _answer_ok(student: str, expected_values: List[str], expected_text: str,
               tol: float, unordered: bool) -> bool:
    student = str(student).strip()
    if not student:
        return False
    res = numeric_match(student, expected_values, tol, unordered)
    if res:
        return True
    if student.lower() == expected_text.strip().lower():
        return True
    return _symbolic_ok(expected_text or ", ".join(expected_values), student)


def _steps_of(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _as_step_list(ex.get("intermediate_steps"))


# ---------------------------------------------------------------------------
# Ultimo messaggio dello studente, conservato come ripiego: se l'argomento passato dal modello a un tool
# non si riesce a interpretare, la verifica riprova con il testo originale scritto dallo studente.
# ---------------------------------------------------------------------------
_LAST_USER_TEXT: Dict[str, str] = {}
_NUM_TOKEN = re.compile(r"-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?")


@hook(priority=10)
async def before_cat_reads_message(user_message_json, cat: StrayCat):
    try:
        if isinstance(user_message_json, dict):
            text = user_message_json.get("text")
        else:
            text = getattr(user_message_json, "text", None)
        _LAST_USER_TEXT[str(cat.id)] = str(text or "")
    except Exception as e:
        log.warning(f"exercise_generator: impossibile leggere il messaggio dello studente: {e}")
    return user_message_json


def _student_numbers(text: str) -> List[Fraction]:
    """Numeri scritti dallo studente, ignorando 'passaggio N' e le etichette x1, x2."""
    t = str(text or "").replace("\u2212", "-")
    t = re.sub(r"\bpassagg\w*\s*(?:n\.?\s*|numero\s*)?\d+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bx\s*_?\s*\d\b", " ", t, flags=re.IGNORECASE)
    out: List[Fraction] = []
    for m in _NUM_TOKEN.finditer(t):
        try:
            out.append(Fraction(m.group(0).replace(",", ".").replace(" ", "")))
        except (ValueError, ZeroDivisionError):
            continue
    return out


def _raw_user_text(cat: StrayCat) -> str:
    return _LAST_USER_TEXT.get(str(cat.id), "")


@tool(examples=["Quali argomenti ci sono?", "Elenca gli argomenti degli esercizi", "Su cosa posso esercitarmi?"])
def list_topics(cat: StrayCat) -> str:
    """Elenca gli argomenti (topic) degli esercizi disponibili nel dataset."""
    df = _load_dataset(cat)
    if df is None or df.empty:
        return "Nessun argomento trovato o dataset vuoto."
    topics = sorted({str(t) for t in df["topic"].to_list()})
    return f"Argomenti disponibili: {', '.join(topics)}"


@tool(examples=[
    "Dammi un esercizio sulle equazioni",
    "Voglio un esercizio facile",
    "Dammi un altro esercizio",
    "Dammi un esercizio di fisica difficile",
    "Carica l'esercizio MATH_QE_001",
    "Vorrei esercitarmi con le equazioni di secondo grado"
])
async def start_exercise(mode: str = "dataset", topic: str = "", difficulty: str = "easy",
                         exercise_id: str = "", cat: StrayCat = None) -> str:
    """Avvia un nuovo esercizio per lo studente. Argomenti: mode ('dataset' di default, 'by_id', 'generated');
    topic = nome esatto di un argomento restituito da list_topics (es. equazioni_secondo_grado), vuoto per qualsiasi;
    difficulty = easy, medium o hard."""
    try:
        m = (mode or "dataset").strip().lower()
        level = normalize_level(difficulty)
        if m == "by_id":
            if not exercise_id:
                return "Errore: exercise_id mancante."
            return await _start_by_id(cat, exercise_id, level)
        if m == "generated":
            return await _start_ai_generated(cat, topic)
        return await _start_from_dataset(cat, topic, level)
    except Exception as e:
        return f"Errore durante l'avvio dell'esercizio: {e}"


async def _do_show(cat: StrayCat) -> str:
    try:
        ex = await _get_current_exercise(cat)
    except ValueError as e:
        return str(e)
    steps, prog = _steps_of(ex), ex.get("progress") or {}
    done = _done_of(prog, steps)
    cur = _next_step(steps, done)
    guided = prog.get("mode") == "guided"
    out = f"Testo: {ex.get('problem_statement', '')}\nModalità: {'guidata (passo passo)' if guided else 'libera'}."
    if prog.get("solved"):
        return out + "\nEsercizio completato: la soluzione finale è corretta ✓."
    if done:
        out += f"\nParti già svolte: {_labels(steps, done)}."
    if cur > len(steps):
        out += "\nHai svolto tutte le parti: scrivimi la soluzione finale."
    elif guided:
        out += f"\nPassaggio corrente: {cur}/{len(steps)} - {steps[cur - 1].get('description', '')}"
    return out


async def _do_hint(cat: StrayCat) -> str:
    try:
        ex = await _get_current_exercise(cat)
    except ValueError as e:
        return str(e)
    steps, prog = _steps_of(ex), ex.setdefault("progress", _new_progress())
    cur = _next_step(steps, _done_of(prog, steps))
    if cur > len(steps):
        return "Hai svolto tutte le parti: scrivimi la soluzione finale."
    step = steps[cur - 1]
    hints = step.get("hints") or [step.get("hint") or "Ricontrolla i calcoli."]
    level = int(prog.get("hint_level", 0))
    hint = hints[min(level, len(hints) - 1)]
    prog["hint_level"] = level + 1
    await _store_active_exercise(cat, ex)
    where = (f"per il passaggio {cur}" if prog.get("mode") == "guided"
             else f"per la prossima parte da svolgere ({_label(step)})")
    return f"Suggerimento {where} ({level + 1}/{len(hints)}): {hint}"


@tool(examples=[
    "Guidami passo passo",
    "Seguimi passo per passo",
    "Non so da dove partire",
    "Voglio risolverlo a passi",
    "Aiutami a risolverlo un passaggio alla volta"
])
async def guide_step_by_step(cat: StrayCat) -> str:
    """Passa alla modalità guidata, un passaggio alla volta. Usalo SOLO se lo studente chiede esplicitamente di essere
    seguito passo passo (o dice di non sapere da dove iniziare). Non usarlo per altre richieste."""
    try:
        ex = await _get_current_exercise(cat)
    except ValueError as e:
        return str(e)
    steps, prog = _steps_of(ex), ex.setdefault("progress", _new_progress())
    prog["mode"] = "guided"
    await _store_active_exercise(cat, ex)
    done = _done_of(prog, steps)
    cur = _next_step(steps, done)
    pre = f"MODALITÀ GUIDATA ATTIVATA. Parti già svolte: {_labels(steps, done) or 'nessuna'}.\n"
    if cur > len(steps):
        return pre + "Tutte le parti sono già svolte: chiedi la soluzione finale."
    return pre + _step_presentation(ex, cur)


_SEP_RE = re.compile(r"(?:\bfa\b|\bfanno\b|\bè\b|\be'\b|\bvale\b|\buguale a\b|\brisulta\b|\bviene\b|=|:)", re.I)


def _student_answer(text: str):
    """Numero esatto scritto dallo studente, anche come espressione ('(-12)^2', 'sqrt(144)', '5/2').
    Si guarda la parte dopo 'fa', '=', ':' ...; prima con SymPy, poi con il calcolo interno."""
    t = _STEP_RE.sub(" ", str(text or ""))
    parts = _SEP_RE.split(t)
    cand = (parts[-1] if len(parts) > 1 else t).strip(" \t.;!?")
    if not cand:
        return None
    for fn in (_sp_number, eval_student_number):
        if fn is None:
            continue
        try:
            v = fn(cand)
        except Exception:
            v = None
        if v is not None:
            return v
    return None


def _require_formula(ex: Dict[str, Any]) -> bool:
    """Per default lo studente deve scrivere la formula; si disattiva con "require_formula": false nel JSON."""
    v = ex.get("require_formula")
    return True if v is None else str(v).strip().lower() not in ("false", "0", "no")


_NEED_FORMULA = ("scrivi anche la formula che usi e come sostituisci i valori.")


def _free_match(ex: Dict[str, Any], steps: List[Dict[str, Any]], answers: List[str], tol: float) -> Dict[str, Any]:
    """Modalità libera: a quale parte dell'esercizio corrisponde quello che ha scritto lo studente?
    Si prova con ogni passaggio (formula verificata su più istanze). Se ha scritto un'etichetta ("Δ = ...",
    "W = ...") si parte da quella grandezza. Restituisce il miglior risultato con "step" = numero del passaggio."""
    envs = sample_envs(ex)
    label = ""
    for cand in answers:
        label = answer_label(cand)
        if label:
            break
    preferred = next((i for i, st in enumerate(steps, 1) if label and str(st.get("target")) == label), 0)
    claimed = 0
    for cand in answers:
        claimed = match_claim(answer_claim(cand), steps)
        if claimed:
            break
    preferred = preferred or claimed
    order = ([preferred] if preferred else []) + [i for i in range(1, len(steps) + 1) if i != preferred]
    best: Dict[str, Any] = {"status": "unreadable", "has_formula": False, "step": 0}
    best_score = -1
    for i in order:
        r: Dict[str, Any] = {"status": "unreadable", "has_formula": False}
        for cand in answers:
            r = check_formula_answer(ex, steps[i - 1], cand, tol, shared_envs=envs)
            if r["status"] != "unreadable":
                break
        r["step"], r["preferred"] = i, i == preferred
        st = r["status"]
        if st == "correct" and r["has_formula"]:
            score = 5
        elif st == "wrong_calc":
            score = 4
        elif st == "correct":
            score = 3
        elif i == preferred and st in ("wrong_formula", "wrong_value", "trivial"):
            score = 2
        elif st == "unreadable":
            score = 0
        else:
            score = 1
        if score > best_score:
            best, best_score = r, score
    best["score"], best["claimed"] = best_score, claimed
    return best


async def _check_step(cat: StrayCat, step_number: int, student_answer: str) -> Dict[str, Any]:
    """Verifica quello che lo studente ha scritto e restituisce esito e testi: head (esito) e tail.
    Nessun suggerimento viene dato qui: i suggerimenti arrivano solo se lo studente li chiede.
    Modalità guidata: si verifica il passaggio corrente. Modalità libera: si capisce da solo a quale parte
    dell'esercizio corrisponde la risposta (anche parziale) e si segnano svolte anche le parti da cui dipende."""
    ex = await _get_current_exercise(cat)
    tol = await _get_tolerance(cat)
    steps, prog = _steps_of(ex), ex.setdefault("progress", _new_progress())
    mode = str(prog.get("mode") or "free")
    done = _done_of(prog, steps)
    cur = _next_step(steps, done)
    if cur > len(steps) and mode == "guided":
        return {"done": True, "head": "Hai completato tutti i passaggi: scrivimi la soluzione finale.", "tail": ""}

    raw = _raw_user_text(cat)
    answer = str(student_answer or "").strip() or raw
    answers = [c for c in dict.fromkeys(c for c in (answer, raw) if c)]
    try:
        n = int(step_number)
    except (TypeError, ValueError):
        n = 0
    info: Dict[str, Any] = {"done": False, "n": n, "value": answer, "expected": "", "description": "",
                            "problem": ex.get("problem_statement", ""), "topic": ex.get("topic", ""), "tail": ""}
    parametric = bool(ex.get("random_params")) and all(st.get("target") is not None for st in steps)
    need = _require_formula(ex)
    ok, counts_as_error = False, False

    if parametric and mode == "free":
        res = _free_match(ex, steps, answers, tol)
        status, i = res["status"], int(res.get("step") or 0)
        label = _label(steps[i - 1]) if i else ""
        if res["score"] >= 5 or (res["score"] == 3 and not need and i):
            ok = True
            new = _mark_done(prog, steps, {i} | step_prerequisites(ex).get(i, set()))
            others = new - {i}
            claimed = int(res.get("claimed") or 0)
            if claimed and claimed != i:
                info["head"] = (f"CORRETTO ✓: formula e risultato sono giusti, ma si riferiscono a {label} e non a "
                                f"«{_label(steps[claimed - 1])}» come hai scritto: controlla il nome della grandezza.")
            elif i not in new:
                info["head"] = f"CORRETTO ✓ ({label}): questa parte era già stata svolta."
            else:
                info["head"] = (f"CORRETTO ✓: hai calcolato correttamente {label}"
                                + (": formula e risultato sono giusti." if res["has_formula"] else "."))
            if others:
                info["head"] += f" Di conseguenza risultano svolti anche: {_labels(steps, others)}."
            done = _done_of(prog, steps)
            info["tail"] = ("Hai svolto tutte le parti: scrivimi la soluzione finale." if len(done) >= len(steps)
                            else f"Parti svolte finora: {_labels(steps, done)}.")
            info["n"], info["description"], info["expected"] = i, label, str(steps[i - 1].get("value", ""))
        elif res["score"] == 4:
            info["head"] = (f"NON corretto ✗: la formula per {label} è giusta, "
                            f"ma il calcolo con i valori (o il risultato) è sbagliato.")
            counts_as_error = True
        elif res["score"] == 3:
            info["head"] = f"Quello che hai scritto non basta: {_NEED_FORMULA}"
        elif res["score"] == 2:
            msgs = {"wrong_formula": f"NON corretto ✗: la formula che hai scritto per {label} non è quella giusta.",
                    "wrong_value": f"NON corretto ✗: il risultato per {label} non è giusto.",
                    "trivial": "Scrivi la formula, non solo il nome della grandezza."}
            info["head"] = msgs.get(status, "NON corretto ✗.")
            counts_as_error = status != "trivial"
        elif res["score"] == 1:
            info["head"] = ("Non riesco a collegare quello che hai scritto a nessuna parte dell'esercizio, quindi non posso "
                            "dire se sia giusto o sbagliato (e non so dove sia un eventuale errore). Riscrivi la formula "
                            "con le lettere del testo (per esempio a, b, c, x1, x2) e i valori sostituiti.")
        else:
            info["head"] = ("Non riesco a leggere la tua risposta. Scrivila come "
                            "formula = valori sostituiti = risultato (usa ^ per le potenze).")
        info["status"] = status
    elif parametric:
        if cur > len(steps):
            return {"done": True, "head": "Hai completato tutti i passaggi: scrivimi la soluzione finale.", "tail": ""}
        n = cur                                  # in modalità guidata si verifica sempre il passaggio corrente
        step = steps[n - 1]
        info.update({"n": n, "expected": str(step.get("value", "")), "description": step.get("description", "")})
        res = {"status": "unreadable", "has_formula": False}
        for cand in answers:
            res = check_formula_answer(ex, step, cand, tol)
            if res["status"] != "unreadable":
                break
        status, has_formula = res["status"], bool(res.get("has_formula"))
        ok = status == "correct" and (has_formula or not need)
        if ok:
            info["head"] = (f"Passaggio {n} CORRETTO ✓: la formula e il risultato sono giusti." if has_formula
                            else f"Passaggio {n} CORRETTO ✓: il risultato è giusto.")
        elif status == "correct":
            info["head"] = (f"Passaggio {n}: il risultato è giusto ✓ ma il passaggio NON è completo: {_NEED_FORMULA}")
        elif status == "wrong_formula":
            info["head"] = f"Passaggio {n} NON corretto ✗: la formula che hai scritto non è quella giusta."
        elif status == "wrong_calc":
            info["head"] = (f"Passaggio {n} NON corretto ✗: la formula è giusta, "
                            f"ma il calcolo con i valori (o il risultato) è sbagliato.")
        elif status == "wrong_value":
            info["head"] = f"Passaggio {n} NON corretto ✗: il risultato non è giusto."
        elif status == "trivial":
            info["head"] = f"Passaggio {n}: scrivi la formula, non solo il nome della grandezza."
        else:
            info["head"] = (f"Passaggio {n}: non riesco a leggere la tua risposta. Scrivila come "
                            f"formula = valori sostituiti = risultato (usa ^ per le potenze).")
        info["status"] = status
        counts_as_error = status in ("wrong_formula", "wrong_calc", "wrong_value")
        if ok:
            _mark_done(prog, steps, {n})
            nxt = _next_step(steps, _done_of(prog, steps))
            info["tail"] = (f"Passa al passaggio {nxt}: {steps[nxt - 1]['description']}" if nxt <= len(steps)
                            else "Hai completato tutti i passaggi! Scrivimi la soluzione finale.")
    else:
        # esercizio statico: solo il valore finale del passaggio corrente
        if cur > len(steps):
            return {"done": True, "head": "Hai completato tutti i passaggi: scrivimi la soluzione finale.", "tail": ""}
        n = n if 1 <= n <= len(steps) and (n in done or n == cur) else cur
        step = steps[n - 1]
        parsed = next((v for v in (_student_answer(c) for c in answers) if v is not None), None)
        value = fmt(parsed) if parsed is not None else str(answer)
        expected_values = step.get("values") or ([step["value"]] if step.get("value") else [])
        ok = _answer_ok(value, expected_values, str(step.get("value", "")), tol, bool(step.get("unordered", False)))
        info.update({"n": n, "value": value, "expected": str(step.get("value", "")),
                     "description": step.get("description", "")})
        info["head"] = (f"Passaggio {n} CORRETTO ✓ (valore dello studente: {value})." if ok
                        else f"Passaggio {n} NON corretto ✗ (valore dello studente: {value}).")
        counts_as_error = not ok
        if ok:
            _mark_done(prog, steps, {n})
            nxt = _next_step(steps, _done_of(prog, steps))
            info["tail"] = (f"Passa al passaggio {nxt}: {steps[nxt - 1]['description']}" if nxt <= len(steps)
                            else "Hai completato tutti i passaggi! Scrivimi la soluzione finale.")

    info["ok"] = ok
    if counts_as_error:
        prog["wrong_attempts"] = int(prog.get("wrong_attempts", 0)) + 1
    await _store_active_exercise(cat, ex)
    return info


async def _do_verify(cat: StrayCat, step_number: int, student_answer: str) -> str:
    try:
        r = await _check_step(cat, step_number, student_answer)
        return f"{r['head']} {r['tail']}".strip()
    except Exception as e:
        return f"Errore durante la verifica: {e}"


async def _check_final(cat: StrayCat, student_solution: str) -> Dict[str, Any]:
    ex = await _get_current_exercise(cat)
    tol = await _get_tolerance(cat)
    expected_text = str(ex.get("correct_solution", "")).strip()
    expected_values = ex.get("solution_values") or ([expected_text] if expected_text else [])

    raw = _raw_user_text(cat)
    student_nums = _student_numbers(raw)
    expected_nums = _student_numbers(expected_text)
    shown = raw.strip() or student_solution

    ok = False
    if student_nums and expected_nums:
        # confronto dei valori, indipendente dall'ordine e dal formato ("x1 = 2, x2 = 4", "2 e 4", ...)
        ok = sorted(student_nums) == sorted(expected_nums)
    if not ok:
        ok = _answer_ok(student_solution, expected_values, expected_text,
                        tol, bool(ex.get("solution_unordered", True)))
    info = {"ok": ok, "shown": shown, "expected": expected_text, "problem": ex.get("problem_statement", ""),
            "topic": ex.get("topic", ""), "final": True}
    if ok:
        info["head"], info["tail"] = f"Soluzione finale CORRETTA ✓ ({shown}).", "Ottimo lavoro!"
    else:
        info["head"], info["tail"] = f"Soluzione finale NON corretta ✗ ({shown}).", "Ricontrolla i calcoli e riprova."
    return info


async def _do_final(cat: StrayCat, student_solution: str) -> str:
    try:
        r = await _check_final(cat, student_solution)
        if r.get("ok"):
            ex = await _get_current_exercise(cat)
            prog = ex.setdefault("progress", _new_progress())
            _mark_done(prog, _steps_of(ex), set(range(1, len(_steps_of(ex)) + 1)))
            prog["solved"] = True
            await _store_active_exercise(cat, ex)
        return f"{r['head']} {r['tail']}".strip()
    except Exception as e:
        return f"Errore nel confronto: {e}"


# ---------------------------------------------------------------------------
# Accesso al modello linguistico del Cat (usato solo dalla generazione libera di esercizi).
# ---------------------------------------------------------------------------


async def _llm_call(cat: StrayCat, prompt: str) -> str:
    """Interroga il modello del Cat (cat.large_language_model, un LLM LangChain) e restituisce il testo."""
    llm = getattr(cat, "large_language_model", None)
    if llm is None:
        raise RuntimeError("modello linguistico non disponibile")
    if hasattr(llm, "ainvoke"):
        res = await llm.ainvoke(prompt)
    else:
        res = llm.invoke(prompt)
    return str(getattr(res, "content", res))


async def _llm_text(cat: StrayCat, prompt: str) -> str:
    try:
        return re.sub(r"\s+", " ", await _llm_call(cat, prompt)).strip()[:400]
    except Exception as e:
        log.warning(f"exercise_generator: commento del LLM non disponibile: {e}")
        return ""


@tool(examples=["Qual è l'esercizio attivo?", "A che punto sono?", "Ripeti il testo dell'esercizio"])
async def show_active_exercise(cat: StrayCat) -> str:
    """Mostra il testo dell'esercizio attivo, la modalità (libera o guidata) e le parti già svolte."""
    return await _do_show(cat)


@tool(examples=["Dammi un suggerimento", "Non so come continuare", "Mi puoi aiutare con questo passaggio?"])
async def request_hint(cat: StrayCat) -> str:
    """Dà un suggerimento progressivo per la prossima parte da svolgere, senza rivelare la soluzione."""
    return await _do_hint(cat)


@tool(examples=[
    "Il passaggio 1 fa b^2 = (-4)^2 = 16",
    "Δ = b² - 4ac = 16 + 20 = 36",
    "x = (-b - √Δ)/(2a) = (4 - 6)/2 = -1",
    "4ac = 4*1*(-5) = -20",
    "v = v0 + a·t = 3 + 2·4 = 11",
    "Provo a calcolare il discriminante: b^2 - 4ac = 4",
    "Secondo me viene 12"
])
async def verify_intermediate_step(step_number: int = 0, student_answer: str = "", student_value: str = "",
                                   cat: StrayCat = None) -> str:
    """Verifica una risposta, anche parziale, dello studente: formula, sostituzione dei valori e risultato.
    Usalo ogni volta che lo studente scrive una formula o un calcolo dell'esercizio, qualunque parte stia svolgendo.
    student_answer deve essere ESATTAMENTE quello che lo studente ha scritto nel suo ultimo messaggio
    (la formula, i valori sostituiti e il risultato): non calcolarlo, non correggerlo, non completarlo.
    step_number: 0 se lo studente non indica un passaggio (in modalità libera viene ignorato).
    student_value è un vecchio nome di student_answer (alias): se lo usi, vale come student_answer."""
    return await _do_verify(cat, step_number, student_answer or student_value)


@tool(examples=["La soluzione è x=3?", "Verifica la mia soluzione finale", "x1 = 1 e x2 = 3", "Le radici sono 2 e 4"])
async def compare_final_solution(student_solution: str, cat: StrayCat) -> str:
    """Verifica la soluzione finale proposta dallo studente.
    student_solution deve essere il testo ESATTO scritto dallo studente nel suo ultimo messaggio."""
    return await _do_final(cat, student_solution)


_STEP_RE = re.compile(r"\bpassagg\w*\s*(?:n\.?\s*|numero\s*)?(\d+)", re.I)


TUTOR_RULES = """
=== TUTOR DI MATEMATICA: REGOLE (OBBLIGATORIE) ===
- Rispondi nella STESSA lingua dello studente, con il "tu", in modo breve e incoraggiante.
- Gli esercizi, gli esiti e i suggerimenti arrivano SOLO dagli strumenti del plugin: non inventare mai esercizi, numeri, passaggi o soluzioni. Ignora i frammenti di "Context" con testi di esercizi o segnaposto tra graffe.
- Esercizio richiesto (anche "un altro", per livello o argomento) -> chiama start_exercise (topic vuoto o un nome restituito da list_topics) e presenta il testo che restituisce.
- MODALITÀ LIBERA (default): lo studente risolve come preferisce e può scrivere anche solo una parte della soluzione. NON mostrare passaggi, formule o valori e NON dire quale sia la prossima parte: invitalo a scrivere formule, valori sostituiti e risultati (formula = valori = risultato).
- MODALITÀ GUIDATA: SOLO se lo studente chiede esplicitamente di essere seguito passo passo (o dice di non sapere da dove partire) chiama guide_step_by_step; da quel momento presenta un passaggio alla volta, come indicato dagli strumenti.
- Quando lo studente scrive una formula, un calcolo o un risultato dell'esercizio -> chiama verify_intermediate_step passando ESATTAMENTE il testo scritto dallo studente, senza calcolarlo, correggerlo o completarlo.
- Soluzione finale -> compare_final_solution. "A che punto sono?" -> show_active_exercise. Argomenti -> list_topics.
- SUGGERIMENTI: dai un suggerimento SOLO se lo studente lo chiede esplicitamente (in quel caso chiama request_hint). Se sbaglia, di' solo dove sta l'errore (formula, calcolo o risultato) come indicato dallo strumento, senza suggerimenti, senza correggere, senza proporre un'altra formula e senza dire che può chiedere aiuto.
- Riporta SOLO ciò che dice lo strumento: NON aggiungere diagnosi, cause dell'errore, parti dell'esercizio o spiegazioni che lo strumento non indica (se dice che non riesce a collegare lo scritto a una parte, NON dire che l'errore è nella formula o nel calcolo).
- L'esito restituito dallo strumento (CORRETTO/NON corretto) va riportato fedelmente: puoi riformularlo e aggiungere una breve frase di incoraggiamento, ma NON cambiare l'esito, NON dare il risultato corretto, NON scrivere la formula giusta e NON anticipare parti future.
- Domande di teoria o di spiegazione (senza risposta da verificare): rispondi tu, in modo chiaro e senza rivelare la soluzione dell'esercizio attivo.
"""


# ---------------------------------------------------------------------------
# I tool vengono proposti al modello cercando per somiglianza tra il messaggio dello studente e le
# descrizioni e gli esempi dei tool (RecallSettings: k=5 e threshold=0.5 di default). Con quei valori un
# messaggio come "il passaggio 1 fa b^2 = (7)^2 = 49" può non trovare verify_intermediate_step, e il modello
# scrive la chiamata come testo. Si allarga la ricerca, così i tool del tutor arrivano sempre al modello;
# resta il modello a decidere se usarli.
# ---------------------------------------------------------------------------
TOOL_RECALL_K = 60
TOOL_RECALL_THRESHOLD = 0.1


@hook(priority=10)
async def before_cat_recalls_memories(config, cat: StrayCat):
    try:
        k = max(int(getattr(config, "k", 0) or 0), TOOL_RECALL_K)
        thr = getattr(config, "threshold", None)
        thr = TOOL_RECALL_THRESHOLD if thr is None else min(float(thr), TOOL_RECALL_THRESHOLD)
        if hasattr(config, "model_copy"):
            return config.model_copy(update={"k": k, "threshold": thr})
        config.k, config.threshold = k, thr
    except Exception as e:
        log.warning(f"exercise_generator: impossibile allargare la ricerca dei tool: {e}")
    return config


@hook(priority=1)
async def agent_prompt_prefix(prefix: str, cat: StrayCat) -> str:
    prefix += TUTOR_RULES
    try:
        ex = await _get_current_exercise(cat)
    except Exception:
        return prefix
    steps, prog = _steps_of(ex), ex.get("progress") or {}
    done = _done_of(prog, steps)
    cur = _next_step(steps, done)
    if prog.get("mode") == "guided":
        step_line = (f"Modalità: GUIDATA. Passaggio corrente: {cur}/{len(steps)} - {steps[cur - 1].get('description', '')}"
                     if cur <= len(steps) else "Modalità: GUIDATA. Tutti i passaggi completati.")
    else:
        step_line = ("Modalità: LIBERA (non mostrare né anticipare i passaggi)."
                     + (f" Parti già svolte dallo studente: {_labels(steps, done)}." if done else ""))
    prefix += f"\n=== ESERCIZIO ATTIVO ===\nTesto: \"{ex.get('problem_statement', '')}\"\n{step_line}\n"
    return prefix