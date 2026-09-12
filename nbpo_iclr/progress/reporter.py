#!/usr/bin/env python3
"""Thirty-minute reporter: snapshot, manuscript status block, PDF, Korean summary.

Runs independently of the job controller. Every cycle it reads the real GPU and
job state off the cluster, counts main-body completion against a frozen
execution matrix, rewrites only the marked status block in main_v6.tex, builds
the PDF in a scratch directory, and atomically replaces the published PDF only
when that build actually succeeded.

Design rules this file follows, because each has already gone wrong once here:
  * a single lock, so two reporters cannot both write the manuscript;
  * re-read the manuscript hash immediately before editing, so a concurrent edit
    by the user is never clobbered -- only the marker block is touched;
  * schedule on a fixed wall-clock grid, so a slow cycle does not make the
    period drift;
  * never report a PDF as fresh when the build failed: keep the last good one
    and say how old it is.
"""
from __future__ import annotations

import json, os, re, shutil, subprocess, sys, tempfile, time, hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))
ROOT = Path("/home/sjkim/MNPO")
PAPER = ROOT / "nbpo_iclr"
TEX = PAPER / "main_v6.tex"
PDF = PAPER / "main_v6.pdf"
PROG = PAPER / "progress"
LOCK = PROG / "reporter.lock"
TEXBIN = "/home/sjkim/MNPO/.TinyTeX/bin/x86_64-linux"
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"
POD = ["kubectl", "exec", "-n", "p-aipr", "nbpo-judge", "-c", "main", "--", "bash", "-lc"]
BEGIN = "% BEGIN AUTO EXPERIMENT STATUS"
END = "% END AUTO EXPERIMENT STATUS"


def now():
    return datetime.now(KST)


def pod(cmd, timeout=120):
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    try:
        r = subprocess.run(POD + [cmd], capture_output=True, text=True, timeout=timeout, env=env)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def snapshot():
    gpus = []
    out = pod("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
              "--format=csv,noheader,nounits")
    for line in out.strip().splitlines():
        try:
            i, u, m, t = (x.strip() for x in line.split(","))
            gpus.append({"gpu": int(i), "util": int(u), "used_mib": int(m), "total_mib": int(t)})
        except ValueError:
            continue
    raw = pod("cat /work/uf4_20260910/jobs/state.json")
    jobs, live_specs = {}, set()
    try:
        payload = json.loads(raw) if raw.strip() else {}
        jobs = payload.get("jobs", {})
        live_specs = set(payload.get("queued_specs", []))
    except Exception:
        jobs, live_specs = {}, set()
    for g in gpus:
        g["job"] = "-"
    for name, e in jobs.items():
        if e.get("state") == "RUNNING":
            for d in e.get("devices", []) or []:
                for g in gpus:
                    if g["gpu"] == d:
                        g["job"] = name
    counts = {}
    for e in jobs.values():
        counts[e.get("state", "?")] = counts.get(e.get("state", "?"), 0) + 1
    # A job requeued under a new id leaves its old FAILED/BLOCKED record behind
    # with no spec to schedule. Those entries are history, not open problems,
    # and counting them with live failures buries a genuinely new one in a
    # constant of six. Report the two separately.
    stale = {s: sorted(n for n, e in jobs.items()
                       if e.get("state") == s and n not in live_specs)
             for s in ("FAILED", "BLOCKED")}
    return {"gpus": gpus, "jobs": jobs, "state_counts": counts,
            "live_specs": sorted(live_specs), "superseded": stale,
            "unreachable": not bool(out.strip())}


def capability_cells_in_manuscript():
    """Count measured cells in tab:general-capability, the exhibit the matrix scores.

    The denominator is methods x benchmarks in execution_matrix.json, so the
    numerator must come from the same table: a data cell is filled when it is
    not \\pending. Sub-rows carrying a seed dispersion are not separate cells.
    """
    text = TEX.read_text(encoding="utf-8")
    start = text.find("\\label{tab:general-capability}")
    if start < 0:
        return 0
    body = text[text.find("\\midrule", start):text.find("\\bottomrule", start)]
    filled = 0
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith("\\\\") or line.startswith("\\midrule"):
            continue
        cells = line[:-2].split("&")
        if len(cells) < 6 or "sample SD over seeds" in cells[0]:
            continue
        filled += sum(1 for c in cells[2:6] if "\\pending" not in c and c.strip())
    return filled


def completion():
    """Count filled main-body cells from the artifacts the cluster actually has."""
    m = json.loads((PROG / "execution_matrix.json").read_text())
    done = pod("ls -d /work/uf4_20260910/evaluation/final_eval/*/complete.json 2>/dev/null")
    evaluated = {Path(p).parent.name for p in done.split() if p.strip()}
    # Every declared main-body method, so a finished arm is actually counted.
    # A prefix missing here silently freezes the objective-row numerator.
    arms_to_method = {"nbpo_mse": "nbpo", "util_mse": "game_utilitarian",
                      "fixedref_mse": "fixed_reference_nash", "btrm_mse": "bt_rm_nash",
                      "maxmin_mse": "game_maxmin", "dpo_uniform_mse": "dpo_uniform",
                      "prosper_mse": "prosper_adapt", "mopo_mse": "mopo_adapt"}
    obj_methods = set()
    if evaluated:
        obj_methods.add("base")
    for arm in evaluated:
        for pref, meth in arms_to_method.items():
            if arm.startswith(pref):
                obj_methods.add(meth)
    # Cross-play and DPO rows are counted from the artifacts that would prove
    # them, so the counters cannot stay at zero after the results land.
    # Judged pairs live one level deeper, under crossplay/pairs/<pair>/, so the
    # old glob matched nothing and the counter sat at 0 while six pairs were in
    # flight. Retired runs carry a dotted suffix and must not be counted.
    cp = pod("ls -d /work/uf4_20260910/evaluation/crossplay/pairs/*/complete.json "
             "2>/dev/null; true")
    crossplay_done = len([q for q in cp.split() if q.strip()
                          and "." not in q.split("/")[-2]])
    dpo_done = len([a for a in evaluated if a.startswith("dpo_")])
    lm_cells = pod("ls -d /work/uf4_20260910/evaluation/capability/*/*/results.json 2>/dev/null")
    lm_cells = len([p for p in lm_cells.split() if p.strip()])
    cap_cells = capability_cells_in_manuscript()
    bench = pod("ls -d /work/nbpo_repair_20260909/responses/uf4_*/ifeval.jsonl 2>/dev/null")
    bench_arms = len([p for p in bench.split() if p.strip()])
    return {
        "objective_rows_done": len(obj_methods),
        "objective_rows_total": len(m["objective_rows"]["methods"]),
        "objective_arms_evaluated": sorted(evaluated),
        "crossplay_done": crossplay_done,
        # The numerator counts judged PAIRS, so the denominator must be pairs
        # too: C(4,2) = 6 unordered pairs over the declared four-policy bank,
        # not the four policies. Reporting 6/4 was the alternative.
        "crossplay_total": (len(m["crossplay"]["entries"])
                            * (len(m["crossplay"]["entries"]) - 1)) // 2,
        "dpo_done": dpo_done, "dpo_total": len(m["dpo_weights"]["weights"]),
        "capability_cells_done": cap_cells,
        "capability_cells_total": len(m["capability"]["methods"]) * len(m["capability"]["benchmarks"]),
        "lm_eval_cells_done": lm_cells,
        "bench_arms_generated": bench_arms,
    }


def esc(text):
    """LaTeX-escape a status string so an underscore in a job name cannot break the build."""
    out = []
    for ch in str(text):
        if ch in "#$%&_{}":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append("\\textbackslash{}")
        elif ch in "~^":
            out.append("\\char`\\" + ch + "{}")
        else:
            out.append(ch)
    return "".join(out)


def status_block(snap, comp, pdf_note):
    t = now().strftime("%Y-%m-%d %H:%M KST")
    rows = []
    for g in snap["gpus"]:
        # Escape once, here. The joined line is inserted verbatim below: running
        # esc() over it again turns every \_ into \textbackslash{}\_ and prints
        # the escapes instead of the job name.
        rows.append("GPU%d: %s, %d\\%%, %d/%d MiB" %
                    (g["gpu"], esc(g["job"]), g["util"], g["used_mib"], g["total_mib"]))
    counts = ", ".join(
        "%s %d%s" % (k, v, (" (%d superseded)" % len(snap.get("superseded", {}).get(k, []))
                            if snap.get("superseded", {}).get(k) else ""))
        for k, v in sorted(snap["state_counts"].items()))
    lines = [
        BEGIN,
        "\\ifshowreviews",
        "\\begin{center}\\small\\fbox{\\parbox{0.95\\linewidth}{%",
        "\\textbf{Draft status, auto-updated %s.} Not part of the reported results." % esc(t),
        "\\\\[2pt]",
        "Main-body objective rows %d/%d; cross-play %d/%d; DPO weights %d/%d; capability cells %d/%d; lm-eval cells %d." %
        (comp["objective_rows_done"], comp["objective_rows_total"],
         comp["crossplay_done"], comp["crossplay_total"],
         comp["dpo_done"], comp["dpo_total"],
         comp["capability_cells_done"], comp["capability_cells_total"],
         comp["lm_eval_cells_done"]),
        "\\\\[2pt]",
        " | ".join(rows),
        "\\\\[2pt]",
        "Queue: %s. %s" % (esc(counts), esc(pdf_note)),
        "}}\\end{center}",
        "\\fi",
        END,
    ]
    text = "\n".join(lines)
    # The LaTeX draft block is English by instruction, and pdflatex here has no
    # CJK font: one Hangul character makes the whole build fail and the cycle
    # then publishes nothing. Strip rather than raise, so a wording slip costs a
    # legible warning instead of the status updates.
    if not text.isascii():
        print("status block had non-ASCII %r; stripped (this block must be English)"
              % sorted({c for c in text if not c.isascii()}), flush=True)
        text = text.encode("ascii", "replace").decode("ascii")
    return text


def update_manuscript(block):
    """Rewrite only the marker block, on whatever the manuscript is right now."""
    text = TEX.read_text()
    before = hashlib.sha256(text.encode()).hexdigest()
    if BEGIN in text and END in text:
        start, end = text.index(BEGIN), text.index(END) + len(END)
        text = text[:start] + block + text[end:]
    else:
        anchor = "\\maketitle"
        if anchor in text:
            i = text.index(anchor) + len(anchor)
            text = text[:i] + "\n\n" + block + "\n" + text[i:]
        else:
            return False, before, "no anchor"
    tmp = TEX.with_suffix(".tex.tmp")
    tmp.write_text(text)
    os.replace(tmp, TEX)
    return True, before, "ok"


def build_pdf():
    """Compile in a scratch copy; only replace the published PDF on success."""
    stamp = now().strftime("%Y%m%d_%H%M_KST")
    with tempfile.TemporaryDirectory(prefix="nbpo_build_") as tmp:
        work = Path(tmp) / "paper"
        work.mkdir()
        for pattern in ("*.tex", "*.sty", "*.bst", "*.bib", "*.bbl"):
            for f in PAPER.glob(pattern):
                if f.is_file():
                    shutil.copy2(f, work / f.name)
        for sub in ("figures", "data"):
            if (PAPER / sub).is_dir():
                shutil.copytree(PAPER / sub, work / sub)
        env = dict(os.environ, PATH=TEXBIN + os.pathsep + os.environ.get("PATH", ""))
        log = work / "main_v6.log"
        ok = False
        try:
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-file-line-error", "main_v6.tex"],
                           cwd=work, env=env, capture_output=True, text=True, timeout=900)
            ok = (work / "main_v6.pdf").exists()
        except Exception:
            ok = False
        if not ok:
            # latexmk can stop on a stale aux; a plain three-pass often succeeds
            for _ in range(3):
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "main_v6.tex"],
                               cwd=work, env=env, capture_output=True, text=True, timeout=600)
            ok = (work / "main_v6.pdf").exists()
        detail = {"stamp": stamp, "ok": ok}
        if log.exists():
            txt = log.read_text(errors="ignore")
            detail["pages"] = (re.findall(r"Output written.*?\((\d+) pages", txt) or [None])[0]
            detail["undefined"] = txt.count("undefined")
            detail["overfull"] = txt.count("Overfull")
            # TeX recovers from several real errors and still writes a PDF: an
            # over-long table row loses its last cell, a bad reference prints
            # empty. The fallback pdflatex pass below runs in nonstopmode, so
            # nothing else would notice. Count the errors and refuse to publish.
            errors = re.findall(r"(?m)^(?:! |[^\n:]+\.tex:\d+: )(.+)$", txt)
            detail["errors"] = len(errors)
            detail["first_errors"] = errors[:3]
            if errors:
                ok = False
                detail["ok"] = False
                detail["failure"] = "LaTeX reported %d error(s); the PDF was not replaced" % len(errors)
            shutil.copy2(log, PROG / "logs" / ("build_%s.log" % stamp))
        if ok:
            archive = PROG / "pdfs" / ("main_v6_%s.pdf" % stamp)
            shutil.copy2(work / "main_v6.pdf", archive)
            tmp_pdf = PDF.with_suffix(".pdf.tmp")
            shutil.copy2(work / "main_v6.pdf", tmp_pdf)
            os.replace(tmp_pdf, PDF)
            detail["archive"] = str(archive)
        return detail


def korean_summary(snap, comp, build, extra):
    t = now().strftime("%H:%M")
    g = {x["gpu"]: x for x in snap["gpus"]}
    def gl(i):
        x = g.get(i)
        if not x:
            return "GPU%d: 조회 실패" % i
        return "GPU%d: %s, %d%%, %d/%d MiB" % (i, x["job"], x["util"], x["used_mib"], x["total_mib"])
    counts = ", ".join(
        "%s %d%s" % (k, v, ("(대체완료 %d)" % len(snap.get("superseded", {}).get(k, []))
                            if snap.get("superseded", {}).get(k) else ""))
        for k, v in sorted(snap["state_counts"].items()))
    if build["ok"]:
        pdfline = "PDF: 성공 %s, %s쪽, nbpo_iclr/main_v6.pdf" % (t, build.get("pages"))
    else:
        last = sorted((PROG / "pdfs").glob("*.pdf"))
        pdfline = "PDF: 실패, 최신 성공본 %s 유지" % (last[-1].name if last else "없음")
    return "\n".join([
        "[KST %s] 본문 완료: objective %d/%d, cross-play %d/%d, DPO weights %d/7, capability %d/%d"
        % (t, comp["objective_rows_done"], comp["objective_rows_total"],
           comp["crossplay_done"], comp["crossplay_total"], comp["dpo_done"],
           comp["capability_cells_done"], comp["capability_cells_total"])
        + " (lm-eval %d)" % comp["lm_eval_cells_done"],
        "진행/다음: 큐 %s → %s" % (counts, extra.get("next", "READY 최상위")),
        gl(0) + " | " + gl(1),
        gl(2) + " | " + gl(3),
        "ETA: %s" % extra.get("eta", "산정 대기"),
        "이번 완료: %s" % extra.get("done", "새 완료 결과 없음"),
        pdfline,
        "Blocker/복구: %s" % extra.get("blocker", "없음"),
    ])


MANIFEST = PROG / "main_results_manifest.json"
# Measured medians (minutes) from logs/queue_report.jsonl. Training is exclusive
# on all four cards; generation fans out over them; the judge takes one card but
# the controller holds the rest for the next 4-GPU training, so it serialises too.
COST = {"train": 184, "gen": 9, "judge": 24}


def refresh_manifest():
    """Rebuild the main-body manifest; returns it, or None if the build failed."""
    try:
        subprocess.run([sys.executable, str(PROG / "build_main_manifest.py")],
                       capture_output=True, text=True, timeout=600, check=True)
        return json.loads(MANIFEST.read_text())
    except Exception:
        return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None


def max_steps_of(job_id):
    """The arm's declared max_steps, read from the config the job actually runs."""
    arm = job_id[len("uf4_train_"):] if job_id.startswith("uf4_train_") else job_id
    out = pod("grep -m1 '^max_steps:' /work/uf4_20260910/configs/%s.yaml 2>/dev/null; true" % arm)
    m = re.search(r"max_steps:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def eval_steps_of(job_id):
    """The arm's dev-pass interval, so remaining passes can be counted."""
    arm = job_id[len("uf4_train_"):] if job_id.startswith("uf4_train_") else job_id
    out = pod("grep -m1 '^eval_steps:' /work/uf4_20260910/configs/%s.yaml 2>/dev/null; true" % arm)
    m = re.search(r"eval_steps:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def train_bar(log, total):
    """The training progress bar, never the periodic evaluation's.

    The trainer runs a dev pass every 250 updates and tqdm prints a second bar
    for it, with a larger denominator. Taking the last bar therefore reports the
    eval's minutes remaining as the training's, which understates the ETA by
    about an hour. Match the denominator instead.
    """
    if not total:
        return None
    # capture the rate field too: the remaining estimate is built from s/it, and
    # a pattern that stops at the elapsed<remaining pair makes the caller fall
    # back to a constant without any sign that it did.
    out = pod(r"tr '\r' '\n' < %s | grep -oE '[0-9]+/%d \[[0-9:]+<[0-9:]+, *[0-9.]+s/it' | tail -1"
              % (log, total))
    return out.strip() or None


DEV_PASS_MIN = 15.7          # measured wall time of one dev pass on the 28k pair set


def running_train_remaining(snap):
    """Minutes left on the training that is on the cards.

    Neither of the obvious readings works. tqdm's own remaining field collapses
    right after each dev pass, because the pause lands in its rate -- it once
    read 15 hours on a run with 40 minutes left. Amortising wall time over
    completed steps fixes that but breaks at the other end: early in a run the
    elapsed time is nearly all fixed startup (model load, dataset map), so
    dividing it across four completed steps projected 41 hours on a run that had
    just begun.

    So: take the per-step rate from the bar, which is a training rate and does
    not include startup, and add the dev passes still to come explicitly. The
    result is then clamped to the measured duration of a finished arm, because
    every arm here runs the same 1,250 updates on the same hardware and no
    honest estimate for one of them is several times another.
    """
    for job_id, entry in snap["jobs"].items():
        if entry.get("state") != "RUNNING" or "train" not in job_id:
            continue
        total = max_steps_of(job_id)
        bar = train_bar(entry.get("log", ""), total)
        m = re.match(r"(\d+)/(\d+).*?([0-9.]+)s/it", bar or "")
        if not (m and total):
            return COST["train"] / 2
        done, per_step = int(m.group(1)), float(m.group(3))
        loop = (total - done) * per_step / 60.0
        every = eval_steps_of(job_id)
        passes = len([s for s in range(done + 1, total + 1) if every and s % every == 0])
        estimate = loop + passes * DEV_PASS_MIN
        return max(0.0, min(estimate, 1.5 * COST["train"]))
    return 0


def etas(man, snap):
    """Three milestone ranges, built from real remaining work, not from a past run."""
    if not man:
        return "산정 대기: manifest 생성 실패"
    rows = man["rows"]
    def undone(pred):
        return [r for r in rows if pred(r) and r["state"] != "DONE"]
    live = running_train_remaining(snap)
    running_arm = next((j[len("uf4_train_"):] for j, e in snap["jobs"].items()
                        if e.get("state") == "RUNNING" and j.startswith("uf4_train_")), None)

    def span(trains, judges, label, note=None, counts_live=False):
        """Minutes of remaining work, as a range.

        `counts_live` says whether the training currently on the cards is one of
        this milestone's own trainings. When it is, it must not be charged twice
        -- once in `live` and again in `trains`. When it is not (the cards are
        busy with some other arm) the live time still has to elapse first, so it
        is added and no training is discounted.
        """
        if note:
            return "%s [산정 대기: %s]" % (label, note)
        if trains == 0 and judges == 0:
            # Nothing outstanding. Without this the milestone echoes the running
            # training's remaining time and a finished milestone reads as hours
            # away -- which it did, showing 2.2-3.1h after the fixed-reference
            # control completed.
            return "%s [완료]" % label
        chargeable = max(0, trains - 1) if counts_live else trains
        lo = live + chargeable * COST["train"] + judges * (COST["gen"] + COST["judge"])
        hi = lo * 1.35 + 10                       # manuscript edit and PDF build
        return "%s [%.1f-%.1fh]" % (label, lo / 60, hi / 60)

    # milestone 1: the fixed-reference row, the matched mechanism control
    # The core-mechanism milestone is both matched controls, not fixed-reference
    # alone. Fixed-reference answered the opponent-adaptation question at three
    # seeds; global game-maxmin now carries the aggregation question and is the
    # arm whose remaining seeds the claim turns on, so the milestone follows the
    # family that is still incomplete. Capability and cross-play rows for the
    # same checkpoints are separate milestones and must not inflate this one.
    MECHANISM = ("fixedref_mse_s", "maxmin_mse_s")

    def mechanism(r):
        return (r["exhibit_label"] == "tab:uf-objectives"
                and r["checkpoint_id"].startswith(MECHANISM))
    m1_tr = undone(lambda r: mechanism(r) and "(train)" in r["seed_or_weight"])
    m1_ju = undone(lambda r: mechanism(r) and "(train)" not in r["seed_or_weight"])
    one = span(len(m1_tr), len(m1_ju), "핵심 기전",
               counts_live=bool(running_arm and running_arm.startswith(MECHANISM)))

    # milestone 2: one evaluated seed for every declared main-body method
    first = {}
    for r in rows:
        if r["exhibit_label"] != "tab:uf-objectives" or "(train)" in r["seed_or_weight"]:
            continue
        first.setdefault(r["method"], []).append(r)
    m2_missing = [m for m, rs in first.items() if not any(x["state"] == "DONE" for x in rs)]

    # "Unimplemented" means no training for it exists anywhere in the queue, not
    # merely that it has no result yet. PROSPER has a solved target, a
    # materialized dataset and a queued training arm, so blaming it for blocking
    # this milestone -- which the previous text did -- is simply wrong.
    trains = {}
    for r in rows:
        if r["exhibit_label"] == "tab:uf-objectives" and "(train)" in r["seed_or_weight"]:
            trains.setdefault(r["method"], []).append(r["state"])
    unimplemented = sorted(m for m in m2_missing
                           if all(s == "NOT_STARTED" for s in trains.get(m, ["NOT_STARTED"])))
    reachable = [m for m in m2_missing if m not in unimplemented]
    if unimplemented:
        # Report the reachable part as a range and name only what actually blocks.
        partial = span(len(reachable), len(reachable), "본문 첫 전체 평가", counts_live=True)
        two = "%s (%d/9, %s 선언 대기)" % (partial, 9 - len(unimplemented),
                                          "/".join(unimplemented))
    else:
        two = span(len(m2_missing), len(m2_missing), "본문 첫 전체 평가", counts_live=True)

    # milestone 3: every declared method, seed and weight, with the paired CI
    tr = undone(lambda r: "(train)" in r["seed_or_weight"])
    ju = undone(lambda r: r["exhibit_label"] in ("tab:uf-objectives", "fig:uf-tradeoffs")
                and "(train)" not in r["seed_or_weight"])
    if unimplemented:
        three = span(0, 0, "본문 최종",
                     "%s 선언 + cross-play bank 미확정" % "/".join(unimplemented))
    else:
        three = span(len(tr), len(ju), "본문 최종", counts_live=True)
    return " / ".join([one, two, three])


def next_ready(ready):
    """The job the controller will actually dispatch next: lowest (priority, id).

    Sorting READY names alphabetically names a different job than the one that
    runs, which is exactly the kind of status line that stops being believed.
    """
    if not ready:
        return "없음"
    out = pod("python3 -c \"import json,glob;"
              "print(json.dumps({json.load(open(p))['job_id']: json.load(open(p)).get('priority',100)"
              " for p in glob.glob('/work/uf4_20260910/jobs/queue/*.json')}))\"")
    try:
        prio = json.loads(out)
    except Exception:
        return sorted(ready)[0] + " (우선순위 조회 실패)"
    best = sorted(ready, key=lambda j: (prio.get(j, 100), j))[0]
    return "%s (prio %s)" % (best, prio.get(best, "?"))


def newly_done(man):
    """Rows that became DONE since the previous cycle, read off status.jsonl."""
    if not man:
        return "새 완료 결과 없음"
    now_done = {"%s|%s|%s" % (r["exhibit_label"], r["method"], r["seed_or_weight"])
                for r in man["rows"] if r["state"] == "DONE"}
    path = PROG / "done_set.json"
    was = set(json.loads(path.read_text())) if path.exists() else None
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(now_done)))
    os.replace(tmp, path)
    if was is None:
        return "기준선 기록 (%d개 완료 상태)" % len(now_done)
    new = sorted(now_done - was)
    return ", ".join(new) if new else "새 완료 결과 없음"


def fill_exhibits():
    """Write any newly measured UF-4 numbers into Table 1's marked region.

    Runs before the build so a judge that finished since the last cycle appears
    in this cycle's PDF. It only rewrites the marker regions, and it reports
    which attributes clear 0.5 so a stale interpretive caption is visible rather
    than silently rewritten.
    """
    try:
        r = subprocess.run([sys.executable, str(PROG / "fill_exhibits.py")],
                           capture_output=True, text=True, timeout=2400)
        if r.returncode != 0:
            return {"ok": False, "detail": (r.stderr or r.stdout)[-300:]}
        return {"ok": True, "detail": (r.stdout or "").strip()[-300:]}
    except Exception as error:                               # noqa: BLE001
        return {"ok": False, "detail": repr(error)[-300:]}


def cycle():
    snap = snapshot()
    comp = completion()
    man = refresh_manifest()
    filled = fill_exhibits()
    pdf_note = "PDF rebuilt each cycle."
    update_manuscript(status_block(snap, comp, pdf_note))
    build = build_pdf()
    ready = [j for j, e in snap["jobs"].items() if e.get("state") == "READY"]
    extra = {"next": next_ready(ready),
             "eta": etas(man, snap),
             "done": newly_done(man),
             "blocker": ("클러스터 조회 실패" if snap["unreachable"]
                         else ("표 자동기입 실패: " + filled["detail"]) if not filled["ok"]
                         else "없음")}
    text = korean_summary(snap, comp, build, extra)
    stamp = now().strftime("%Y%m%d_%H%M_KST")
    (PROG / "latest.md").write_text(text + "\n")
    (PROG / "logs" / ("status_%s.md" % stamp)).write_text(text + "\n")
    with (PROG / "status.jsonl").open("a") as f:
        f.write(json.dumps({"kst": now().strftime("%Y-%m-%d %H:%M:%S"),
                            "gpus": snap["gpus"], "state_counts": snap["state_counts"],
                            "completion": comp, "build": build,
                            "exhibit_fill": filled}) + "\n")
    print(text, flush=True)
    print("-" * 60, flush=True)


def main():
    PROG.mkdir(parents=True, exist_ok=True)
    (PROG / "logs").mkdir(exist_ok=True)
    (PROG / "pdfs").mkdir(exist_ok=True)
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            os.kill(old, 0)
            print("another reporter is alive at pid %d; exiting" % old); return
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()))
    period = 1800
    try:
        while True:
            started = time.time()
            try:
                cycle()
            except Exception as error:                       # noqa: BLE001
                print("cycle error: %r" % (error,), flush=True)
            # fixed grid so a slow cycle does not make the period drift
            sleep = period - ((time.time() - started) % period)
            time.sleep(max(60, sleep))
    finally:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()


if __name__ == "__main__":
    main()
