"""V1-V6: ECC verification-loop (tests, build, lint, types, security, inventory). Never imports `floodnet`.

ruff and pyright run from a throwaway venv in .tmp/auditenv so the project environment stays unchanged.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent import OUT, ROOT, guard, write_json

PY = ROOT / ".venv/Scripts/python.exe"
TOOLENV = ROOT / ".tmp/auditenv"


def run(command, **kwargs):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs)
    return result.returncode, result.stdout + result.stderr


def main():
    report = {}
    OUT.mkdir(parents=True, exist_ok=True)
    # V1 fresh pytest
    code, output = run([str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(ROOT / ".tmp/pytest-audit"),
                        "--junitxml", str(OUT / "pytest.xml")])
    suites = list(ET.parse(OUT / "pytest.xml").getroot().iter("testsuite"))
    total = lambda key: sum(int(s.attrib.get(key, 0)) for s in suites)
    report["V1"] = {"exit_code": code, "tests": total("tests"), "failures": total("failures"), "errors": total("errors"),
                    "skipped": total("skipped"), "tail": output.strip().splitlines()[-1:],
                    "pass": code == 0 and total("failures") == 0 and total("errors") == 0 and total("skipped") == 0}
    # V2 build: bytecode compile + JS syntax
    compile_code, compile_out = run([str(PY), "-m", "compileall", "-q", "src", "audit", "scripts", "tests"])
    node = {p.name: run(["node", "--check", str(p)])[0] for p in sorted((ROOT / "web/assets/js").glob("*.js"))}
    report["V2"] = {"compileall_exit": compile_code, "compileall_output": compile_out.strip()[:500], "node_check": node,
                    "pass": compile_code == 0 and not any(node.values())}
    # V3/V4 lint and types from a throwaway tool venv
    tool_py = TOOLENV / "Scripts/python.exe"
    if not tool_py.exists():
        run([sys.executable, "-m", "venv", str(TOOLENV)])
        run([str(tool_py), "-m", "pip", "install", "-q", "ruff", "pyright"])
    _, versions = run([str(tool_py), "-m", "pip", "freeze"])
    report["tool_versions"] = [line for line in versions.splitlines() if line.lower().startswith(("ruff", "pyright"))]
    ruff_code, ruff_out = run([str(TOOLENV / "Scripts/ruff.exe"), "check", "--select", "F,E9", "--output-format", "json", "src", "audit", "scripts", "tests"])
    findings = json.loads(ruff_out[ruff_out.index("["):]) if "[" in ruff_out else []
    report["V3"] = {"rules": "F (pyflakes), E9 (syntax)", "findings": len(findings),
                    "by_rule": {r: sum(f["code"] == r for f in findings) for r in sorted({f["code"] for f in findings})},
                    "items": [f"{Path(f['filename']).relative_to(ROOT).as_posix()}:{f['location']['row']} {f['code']} {f['message']}" for f in findings][:40],
                    "pass": not findings}
    pyright_code, pyright_out = run([str(TOOLENV / "Scripts/pyright.exe"), "--pythonpath", str(PY), "--outputjson", "src"], timeout=900)
    try:
        parsed = json.loads(pyright_out[pyright_out.index("{"):])
        diagnostics = parsed.get("generalDiagnostics", [])
        errors = [d for d in diagnostics if d["severity"] == "error"]
        report["V4"] = {"errors": len(errors), "warnings": sum(d["severity"] == "warning" for d in diagnostics),
                        "items": [f"{Path(d['file']).relative_to(ROOT).as_posix()}:{d['range']['start']['line'] + 1} {d['message'][:140]}" for d in errors][:40],
                        "pass": None}
    except (ValueError, KeyError) as error:
        report["V4"] = {"error": f"pyright output not parsed: {error}", "raw": pyright_out[:500], "pass": None}
    # V5 secrets and absolute user paths
    patterns = {"api_key_like": r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|api[_-]?key\s*[:=]\s*['\"][^'\"]+)",
                "absolute_user_path": r"[A-Za-z]:[\\/]+Users[\\/]"}
    targets = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "web/assets").rglob("*.*")) + [ROOT / "web/data/results.json", ROOT / "web/index.html"]
    hits = []
    for path in targets:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                hits.append(f"{path.relative_to(ROOT).as_posix()}: {name}: {match.group(0)[:40]}")
    report["V5"] = {"files_scanned": len(targets), "hits": hits, "pass": not hits}
    # V6 inventory (gp/ is untracked, so there is no diff) and push safety
    _, status = run(["git", "-C", str(ROOT.parent), "status", "--short", "--untracked-files=all", "gp"])
    lines = [l for l in status.splitlines() if l.strip()]
    groups = {}
    for line in lines:
        parts = line[3:].strip().strip('"').split("/")
        key = "/".join(parts[:2]) if len(parts) > 2 else parts[-1]
        groups[key] = groups.get(key, 0) + 1
    _, ahead = run(["git", "-C", str(ROOT.parent), "rev-list", "--count", "@{u}..HEAD"])
    _, last = run(["git", "-C", str(ROOT.parent), "log", "-1", "--format=%h %s"])
    skills = {}
    for repo in ("superpowers", "ECC"):
        _, remotes = run(["git", "-C", str(ROOT.parents[1] / "audit-skills" / repo), "remote", "-v"])
        skills[repo] = remotes.strip().splitlines()
    report["V6"] = {"untracked_files_in_gp": len(lines), "by_folder": dict(sorted(groups.items())),
                    "local_commits_not_pushed": ahead.strip(), "last_commit": last.strip(), "skill_repo_remotes": skills,
                    "pass": True}
    guard()
    write_json("code.json", report)
    print(json.dumps({k: v.get("pass") for k, v in report.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
