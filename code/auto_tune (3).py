# =============================================================================
# auto_tune.py — Bayesian Gain Optimiser for WAM-V FPD Stabilisation Platform
# =============================================================================
#
# WHAT IT DOES
#   Reads fpd_tune_data_*.csv logs produced by main_fpd_tune.py, scores each
#   run on stabilisation performance, and uses Bayesian optimisation (Optuna /
#   TPE sampler) to suggest the next gains.json to try.  The model learns from
#   every run — the more data it sees, the smarter its suggestions become.
#
#   Because prior test history already exists, seed the model first so it has
#   a strong prior before suggesting anything new.
#
# SCORE FUNCTION
#   score = roll_weight  * roll_reduction_%
#         + pitch_weight * pitch_reduction_%
#         - amp_penalty  * max(0, peak_amps - amp_baseline)
#         - instability_penalty   (if growing oscillation detected)
#   Higher is better.  Pitch is weighted slightly higher (1.2 vs 1.0) because
#   it has historically been the harder axis to stabilise.
#
# DEPENDENCY
#   pip install optuna
#
# -----------------------------------------------------------------------------
# QUICKSTART
# -----------------------------------------------------------------------------
#
#   1. FIRST TIME — seed with all historical CSV logs:
#        python3 tools/auto_tune.py --seed "path/to/logs/*.csv"
#
#   2. AFTER EVERY TEST RUN going forward:
#        python3 tools/auto_tune.py logs/fpd_tune_data_YYYYMMDD_HHMMSS.csv
#        # scores the run, updates the model, writes new gains.json
#        python3 main_fpd_tune.py   # run the suggested gains
#
#   3. OTHER COMMANDS:
#        python3 tools/auto_tune.py --best                    # show best gains so far
#        python3 tools/auto_tune.py run.csv --dry-run         # preview without writing
#        python3 tools/auto_tune.py run.csv --no-suggest      # ingest only, no suggestion
#        python3 tools/auto_tune.py run.csv --pitch-weight 1.5  # reprioritise pitch
#        python3 tools/auto_tune.py --reset                   # wipe study, start fresh
#
# -----------------------------------------------------------------------------
# FILES CREATED
#   tune_study.db     — Optuna SQLite database, persists between runs
#   tune_pending.json — trial number of the last suggestion (for reference)
#   gains.json        — overwritten with each new suggestion
#
# -----------------------------------------------------------------------------
# GAIN SEARCH BOUNDS  (edit BOUNDS dict below to widen/narrow the search space)
#   ROLL_GAIN    : 300 – 700    (ceiling ~750 before oscillation)
#   PITCH_GAIN   : 150 – 380    (ceiling ~400 without sufficient D)
#   ROLL/PITCH KD, KFF, filters, deadbands — see BOUNDS dict
# =============================================================================

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("[ERROR] optuna not installed.  Run:  pip install optuna")
    sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent
STUDY_DB   = str(REPO_ROOT / "tune_study.db")
GAINS_FILE = str(REPO_ROOT / "gains.json")
STUDY_NAME = "fpd_tune"


# ── Score weights ─────────────────────────────────────────────────────────────

DEFAULT_ROLL_WEIGHT   = 1.0
DEFAULT_PITCH_WEIGHT  = 1.2   # pitch historically harder — weight it higher
AMP_PENALTY           = 0.8   # penalty per amp above AMP_BASELINE
AMP_BASELINE          = 5.0   # amps below this are not penalised
INSTABILITY_PENALTY   = 35.0  # applied when oscillation detected
SHORT_RUN_PENALTY     = 15.0  # applied when run is shorter than MIN_RUN_SECONDS
MIN_RUN_SECONDS       = 20.0
MIN_DISTURBANCE_DEG   = 0.5   # skip runs with almost no wave disturbance


# ── Default gains ─────────────────────────────────────────────────────────────

DEFAULTS = {
    "ROLL_GAIN": 600.0,   "PITCH_GAIN": 300.0,
    "ROLL_KI":   3.0,     "PITCH_KI":   2.0,
    "MAX_I_ERPM": 300.0,
    "ROLL_KD":   10.0,    "PITCH_KD":  -5.0,
    "MAX_D_ERPM": 600.0,
    "D_FILTER_ALPHA": 0.15, "FF_FILTER_ALPHA": 0.4,
    "ANGLE_FILTER_ALPHA": 1.0,
    "FF_RATE_DEADBAND_DPS": 1.5,
    "ROLL_DEADBAND_DEG": 0.5, "PITCH_DEADBAND_DEG": 1.0,
    "MAX_MOTOR_AMPS": 10.0,   "MAX_ERPM": 5000.0,
    "ROLL_KFF": 70.0,    "PITCH_KFF": 40.0,
}

# Gains the optimiser will tune (others kept at their current value)
TUNABLE = [
    "ROLL_GAIN",   "PITCH_GAIN",
    "ROLL_KI",     "PITCH_KI",
    "ROLL_KD",     "PITCH_KD",
    "ROLL_KFF",    "PITCH_KFF",
    "MAX_D_ERPM",
    "D_FILTER_ALPHA",   "FF_FILTER_ALPHA",
    "FF_RATE_DEADBAND_DPS",
    "ROLL_DEADBAND_DEG", "PITCH_DEADBAND_DEG",
]

# ── Search bounds [min, max] ──────────────────────────────────────────────────
# Based on known-stable ranges from test history.
# Ceiling notes: ROLL_GAIN >= 750 oscillates
#                PITCH_GAIN >= 400 oscillates without sufficient D

BOUNDS = {
    "ROLL_GAIN":            [300.0,  700.0],
    "PITCH_GAIN":           [150.0,  380.0],
    "ROLL_KI":              [0.0,    10.0],
    "PITCH_KI":             [0.0,    8.0],
    "ROLL_KD":              [2.0,    30.0],
    "PITCH_KD":             [-25.0, -1.0],
    "MAX_D_ERPM":           [200.0,  1000.0],
    "ROLL_KFF":             [20.0,   160.0],
    "PITCH_KFF":            [10.0,   100.0],
    "D_FILTER_ALPHA":       [0.05,   0.5],
    "FF_FILTER_ALPHA":      [0.1,    0.8],
    "FF_RATE_DEADBAND_DPS": [0.5,    5.0],
    "ROLL_DEADBAND_DEG":    [0.1,    2.0],
    "PITCH_DEADBAND_DEG":   [0.1,    2.5],
}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _col(rows, name):
    out = []
    for r in rows:
        try:
            out.append(float(r[name]))
        except (KeyError, ValueError):
            pass
    return out


def _col_str(rows, name):
    return [r.get(name, "") for r in rows]


def rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def detect_oscillation(signal, threshold_ratio=2.5):
    """Growing amplitude in second half of run = instability heuristic."""
    if len(signal) < 40:
        return False
    n = len(signal)
    mid = n // 2
    rms1 = rms(signal[:mid])
    rms2 = rms(signal[mid:])
    if rms1 > 0 and rms2 / rms1 > threshold_ratio:
        return True
    tail = signal[int(0.8 * n):]
    if tail:
        pp = max(tail) - min(tail)
        overall_pp = max(signal) - min(signal)
        if overall_pp > 0 and pp / overall_pp > 0.85:
            return True
    return False


def load_gains_for_csv(csv_path):
    """Find gains.json beside the CSV, fall back to GAIN CHANGE log, then defaults."""
    gains = dict(DEFAULTS)
    candidates = [
        Path(csv_path).parent / "gains.json",
        Path(csv_path).parent.parent / "gains.json",
        Path(GAINS_FILE),
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c) as f:
                    loaded = json.load(f)
                for k in DEFAULTS:
                    if k in loaded:
                        gains[k] = float(loaded[k])
                return gains
            except Exception:
                pass

    # Fall back to GAIN CHANGE entries in CSV
    try:
        rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        for entry in _col_str(rows, "gain_event"):
            if "CHANGED:" not in entry:
                continue
            for part in entry.replace("CHANGED:", "").split(","):
                part = part.strip()
                if ":" in part and "->" in part:
                    key, val = part.split(":", 1)
                    key = key.strip()
                    new_val = val.split("->")[-1].strip()
                    if key in DEFAULTS:
                        try:
                            gains[key] = float(new_val)
                        except ValueError:
                            pass
    except Exception:
        pass
    return gains


def score_csv(csv_path, roll_weight=DEFAULT_ROLL_WEIGHT,
              pitch_weight=DEFAULT_PITCH_WEIGHT, verbose=False):
    """
    Parse a CSV and return (score, metrics_dict, gains_dict).
    Returns (None, None, None) if the CSV is unusable.
    """
    try:
        rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except Exception as e:
        print(f"  [WARN] Could not read {Path(csv_path).name}: {e}")
        return None, None, None

    if len(rows) < 100:
        if verbose:
            print(f"  [SKIP] {Path(csv_path).name} — too few rows ({len(rows)})")
        return None, None, None

    times  = _col(rows, "t_s")
    r1     = _col(rows, "imu1_roll_deg")
    p1     = _col(rows, "imu1_pitch_deg")
    r2     = _col(rows, "imu2_roll_deg")
    p2     = _col(rows, "imu2_pitch_deg")
    r_amps = _col(rows, "motor_roll_amps")
    p_amps = _col(rows, "motor_pitch_amps")

    if not r2 or not p2:
        if verbose:
            print(f"  [SKIP] {Path(csv_path).name} — missing IMU2 columns")
        return None, None, None

    rms_r2 = rms(r2); rms_r1 = rms(r1)
    rms_p2 = rms(p2); rms_p1 = rms(p1)

    if rms_r2 < MIN_DISTURBANCE_DEG and rms_p2 < MIN_DISTURBANCE_DEG:
        if verbose:
            print(f"  [SKIP] {Path(csv_path).name} — disturbance too small")
        return None, None, None

    roll_red  = (1 - rms_r1 / rms_r2) * 100 if rms_r2 > 0 else 0.0
    pitch_red = (1 - rms_p1 / rms_p2) * 100 if rms_p2 > 0 else 0.0

    max_amps   = max(max(r_amps) if r_amps else 0,
                     max(p_amps) if p_amps else 0)
    duration_s = (times[-1] - times[0]) if len(times) > 1 else 0.0

    oscillating = detect_oscillation(r1) or detect_oscillation(p1)

    score  = roll_weight * roll_red + pitch_weight * pitch_red
    score -= AMP_PENALTY * max(0.0, max_amps - AMP_BASELINE)
    if oscillating:
        score -= INSTABILITY_PENALTY
    if duration_s < MIN_RUN_SECONDS:
        score -= SHORT_RUN_PENALTY

    gains = load_gains_for_csv(csv_path)

    metrics = {
        "csv":         Path(csv_path).name,
        "duration_s":  round(duration_s, 1),
        "roll_red":    round(roll_red,  1),
        "pitch_red":   round(pitch_red, 1),
        "rms_r2":      round(rms_r2, 3),
        "rms_r1":      round(rms_r1, 3),
        "rms_p2":      round(rms_p2, 3),
        "rms_p1":      round(rms_p1, 3),
        "max_amps":    round(max_amps, 2),
        "oscillating": oscillating,
        "score":       round(score, 2),
    }

    if verbose:
        flag = "  *** OSCILLATING ***" if oscillating else ""
        print(f"  {Path(csv_path).name}")
        print(f"    Roll {roll_red:.1f}%  Pitch {pitch_red:.1f}%  "
              f"Amps {max_amps:.1f}A  {duration_s:.0f}s  "
              f"score={score:.1f}{flag}")

    return score, metrics, gains


# ── Optuna helpers ────────────────────────────────────────────────────────────

def get_study():
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",
        load_if_exists=True,
    )


def ingest_run(csv_path, roll_weight=DEFAULT_ROLL_WEIGHT,
               pitch_weight=DEFAULT_PITCH_WEIGHT, verbose=True):
    """Score a CSV and add it as a completed trial to the Optuna study."""
    score, metrics, gains = score_csv(csv_path, roll_weight, pitch_weight, verbose)
    if score is None:
        return False

    study = get_study()

    # Skip if already ingested
    for t in study.trials:
        if t.user_attrs.get("csv") == metrics["csv"]:
            if verbose:
                print(f"  [SKIP] {metrics['csv']} already in study.")
            return False

    # Only include gains that fall within search bounds
    distributions = {}
    params = {}
    for k in TUNABLE:
        if k in gains:
            lo, hi = BOUNDS[k]
            val = gains[k]
            if lo <= val <= hi:
                distributions[k] = optuna.distributions.FloatDistribution(lo, hi)
                params[k] = val

    if not params:
        if verbose:
            print(f"  [SKIP] {metrics['csv']} — gains outside search bounds.")
        return False

    trial = optuna.trial.create_trial(
        params=params,
        distributions=distributions,
        value=score,
        user_attrs=metrics,
    )
    study.add_trial(trial)
    return True


def suggest_next(roll_weight=DEFAULT_ROLL_WEIGHT,
                 pitch_weight=DEFAULT_PITCH_WEIGHT,
                 dry_run=False):
    """Ask Optuna for the next gains to try and write gains.json."""
    study = get_study()
    n = len(study.trials)

    sampler = (optuna.samplers.RandomSampler()
               if n < 5
               else optuna.samplers.TPESampler(n_startup_trials=5))

    study2 = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    trial = study2.ask()
    suggested = {}
    for k in TUNABLE:
        lo, hi = BOUNDS[k]
        suggested[k] = trial.suggest_float(k, lo, hi)

    # Save pending trial number so we can mark it complete after next run
    pending_path = str(REPO_ROOT / "tune_pending.json")
    with open(pending_path, "w") as f:
        json.dump({"trial_number": trial.number, "suggested": suggested}, f, indent=2)

    # Merge onto current gains.json
    base_gains = dict(DEFAULTS)
    if os.path.exists(GAINS_FILE):
        try:
            with open(GAINS_FILE) as f:
                base_gains.update(json.load(f))
        except Exception:
            pass
    base_gains.update(suggested)

    if not dry_run:
        with open(GAINS_FILE, "w") as f:
            json.dump(base_gains, f, indent=2)
        print(f"[INFO] gains.json written  (trial #{trial.number})")
        print(f"[INFO] Pending state saved to tune_pending.json")
    else:
        print("[DRY RUN] Suggested gains (not written):")

    return base_gains


def print_diff(old, new):
    print()
    print(f"  {'Key':<28} {'Current':>10}  {'Suggested':>10}  {'Change':>10}")
    print("  " + "-" * 64)
    changed = False
    for k in TUNABLE:
        ov = old.get(k); nv = new.get(k)
        if ov is None or nv is None:
            continue
        delta = nv - ov
        if abs(delta) > 1e-6:
            changed = True
            arrow = "+" if delta > 0 else ""
            print(f"  {k:<28} {ov:>10.3f}  {nv:>10.3f}  {arrow}{delta:>9.3f}")
    if not changed:
        print("  (no changes)")
    print()


def print_best():
    study = get_study()
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        print("[INFO] No completed trials yet. Run --seed first.")
        return

    best = study.best_trial
    a = best.user_attrs
    print()
    print("=" * 64)
    print(f"  Best run:  trial #{best.number}  score={best.value:.2f}")
    print(f"  CSV:       {a.get('csv','?')}")
    print(f"  Roll:      {a.get('roll_red','?')}%   "
          f"Pitch: {a.get('pitch_red','?')}%   "
          f"Amps: {a.get('max_amps','?')}A   "
          f"Duration: {a.get('duration_s','?')}s")
    print()
    print("  Gains:")
    for k, v in best.params.items():
        print(f"    {k:<28} {v:.4f}")
    print("=" * 64)

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print()
    print(f"  Top 5 runs:")
    print(f"  {'#':<5} {'Score':>7}  {'Roll%':>7}  {'Pitch%':>7}  {'Amps':>6}  CSV")
    print("  " + "-" * 72)
    for t in top5:
        ua = t.user_attrs
        print(f"  {t.number:<5} {t.value:>7.1f}  "
              f"{ua.get('roll_red','?'):>7}  "
              f"{ua.get('pitch_red','?'):>7}  "
              f"{ua.get('max_amps','?'):>6}  "
              f"{ua.get('csv','?')}")
    print()


def expand_paths(path_args):
    out = []
    for p in path_args:
        expanded = glob.glob(p, recursive=True)
        if expanded:
            out += [e for e in expanded if e.endswith(".csv")]
            continue
        if os.path.isdir(p):
            out += [str(f) for f in sorted(Path(p).glob("fpd_tune_data_*.csv"))]
            continue
        if os.path.isfile(p) and p.endswith(".csv"):
            out.append(p)
        else:
            print(f"[WARN] Could not find: {p}")
    return sorted(set(out))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian FPD gain optimiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csvs", nargs="*",
                        help="CSV log file(s) or folder to ingest")
    parser.add_argument("--seed", nargs="+", metavar="PATH",
                        help="Ingest all historical CSVs — use on first run")
    parser.add_argument("--best",       action="store_true",
                        help="Show best gains found so far")
    parser.add_argument("--reset",      action="store_true",
                        help="Delete study database and start fresh")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print suggested gains without writing gains.json")
    parser.add_argument("--no-suggest", action="store_true",
                        help="Ingest only, do not suggest next gains")
    parser.add_argument("--roll-weight",  type=float, default=DEFAULT_ROLL_WEIGHT)
    parser.add_argument("--pitch-weight", type=float, default=DEFAULT_PITCH_WEIGHT)
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(STUDY_DB):
            os.remove(STUDY_DB)
            print(f"[INFO] Study reset — deleted {STUDY_DB}")
        else:
            print("[INFO] No study database found.")
        return

    if args.best:
        print_best()
        return

    rw = args.roll_weight
    pw = args.pitch_weight

    # ── Seed from historical data ─────────────────────────────────────────
    if args.seed:
        paths = expand_paths(args.seed)
        if not paths:
            print("[ERROR] No CSV files found for --seed.")
            sys.exit(1)
        print(f"[INFO] Seeding with {len(paths)} CSV file(s)...\n")
        ingested = sum(ingest_run(p, rw, pw, verbose=True) for p in paths)
        study = get_study()
        print(f"\n[INFO] Done. {ingested} new runs added.  "
              f"Study total: {len(study.trials)} trials.")
        print_best()
        if not args.no_suggest:
            old = dict(DEFAULTS)
            if os.path.exists(GAINS_FILE):
                with open(GAINS_FILE) as f:
                    old.update(json.load(f))
            new = suggest_next(rw, pw, args.dry_run)
            print_diff(old, new)
        return

    # ── Ingest new run(s) and suggest ─────────────────────────────────────
    if not args.csvs:
        parser.print_help()
        return

    paths = expand_paths(args.csvs)
    if not paths:
        print("[ERROR] No CSV files found.")
        sys.exit(1)

    print(f"[INFO] Ingesting {len(paths)} CSV file(s)...\n")
    ingested = sum(ingest_run(p, rw, pw, verbose=True) for p in paths)
    study = get_study()
    print(f"\n[INFO] {ingested} new run(s) added.  "
          f"Study total: {len(study.trials)} trials.")

    if args.no_suggest:
        return

    old = dict(DEFAULTS)
    if os.path.exists(GAINS_FILE):
        try:
            with open(GAINS_FILE) as f:
                old.update(json.load(f))
        except Exception:
            pass

    print("\n[INFO] Generating next suggestion...\n")
    new = suggest_next(rw, pw, args.dry_run)
    print_diff(old, new)

    if not args.dry_run:
        print("[INFO] Run main_fpd_tune.py to test the suggested gains.")


if __name__ == "__main__":
    main()
