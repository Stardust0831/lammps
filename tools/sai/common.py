#!/usr/bin/env python3
"""Files and reports shared by the SAI external GPU tester."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

CONTROL_FILES = ("common.py", "endpoint.py", "suite.py", "site.json")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def control_digest(root):
    text = "".join(name + " " + digest(Path(root) / name) + "\n"
                   for name in CONTROL_FILES)
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def reports(run, cases, machine=None, version="", banner=""):
    """Use the JUnit + summary fields consumed by LAMMPS fetch_unittest.py."""
    run = Path(run)
    plan = read(run / "plan.json")
    output = run / "results"
    output.mkdir(exist_ok=True)
    failed = sum(case["status"] == "failure" for case in cases)
    errors = sum(case["status"] == "error" for case in cases)
    now = stamp()
    summary = dict(date=now, commit=plan["source_sha"], branch="develop", version=version,
                   compiler=(machine or {}).get("compiler", "see help.txt in raw results"),
                   operating_system=(machine or {}).get("operating_system", "Linux x86_64"),
                   url=plan["report_url"], status="FAIL" if failed or errors else "PASS",
                   tests=len(cases), failures=failed, errors=errors, skipped=0,
                   passed=len(cases) - failed - errors, machine=machine or {},
                   action_url=plan["action_url"], run_id=plan["run_id"],
                   source_repository="https://github.com/lammps/lammps",
                   controller_sha=plan["controller_sha"], controller_digest=plan["control_digest"],
                   image_sha256=plan["image_sha256"], recipe_sha256=plan["recipe_sha256"],
                   slurm_job=plan.get("job"), scope="LJ and PPPM CPU/GPU numerical comparisons; "
                   "1 GPU, 2 GPUs on one node, and 4 GPUs on two nodes. "
                   "Timings are diagnostic, not a scaling benchmark.")
    suite = ET.Element("testsuite", name="SAI MPI CUDA regression", timestamp=now,
                       tests=str(len(cases)), failures=str(failed), errors=str(errors),
                       skipped="0", time=str(sum(c.get("seconds", 0) for c in cases)))
    properties = ET.SubElement(suite, "properties")
    for key, value in (("commit", plan["source_sha"]), ("site", "SAI"),
                       ("source_repository", summary["source_repository"]),
                       ("test_kind", "MPI/GPU numerical regression"),
                       ("controller_sha", plan["controller_sha"]),
                       ("image_sha256", plan["image_sha256"]),
                       ("report_url", plan["report_url"]),
                       ("action_url", plan["action_url"]),
                       ("hardware", json.dumps(machine or {}, sort_keys=True))):
        ET.SubElement(properties, "property", name=key, value=str(value))
    for case in cases:
        node = ET.SubElement(suite, "testcase", name=case["name"],
                             classname="SAI.LAMMPS.MPI.CUDA", time=str(case.get("seconds", 0)))
        if case["status"] != "pass":
            ET.SubElement(node, case["status"], message=case["message"]).text = case["message"]
        ET.SubElement(node, "system-out").text = banner + "\n" + json.dumps(case, indent=2)
    ET.indent(suite)
    ET.ElementTree(suite).write(output / "junit.xml", encoding="utf-8", xml_declaration=True)
    write(output / "summary.json", summary)
    write(output / "cases.json", cases)
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(case.get(key, ""))) + "</td>"
                  for key in ("name", "status", "seconds", "message")) + "</tr>" for case in cases)
    report = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
              "<meta name='viewport' content='width=device-width,initial-scale=1'>"
              "<title>SAI LAMMPS multi-GPU tests</title><style>body{max-width:1100px;margin:3em auto;"
              "padding:0 1em;font:16px system-ui}table{border-collapse:collapse}td,th{padding:.5em;"
              "border:1px solid #bbb;text-align:left}pre{white-space:pre-wrap}</style>"
              "<h1>SAI LAMMPS multi-GPU tests: " + summary["status"] + "</h1>"
              "<p>Develop commit <a href='https://github.com/lammps/lammps/commit/" + plan["source_sha"]
              + "'>" + plan["source_sha"] + "</a><br>Completed: " + now + "</p>"
              "<p>" + html.escape(summary["scope"]) + "</p><p>"
              "<a href='junit.xml'>JUnit XML</a> | <a href='summary.json'>Summary JSON</a> | "
              "<a href='raw.tar.gz'>Raw inputs and outputs</a> | "
              "<a href='manifest.json'>File hashes</a> | <a href='" + html.escape(plan["action_url"])
              + "'>GitHub run</a></p><table><tr><th>Case</th><th>Status</th><th>Seconds</th>"
              "<th>Details</th></tr>" + rows + "</table><h2>Machine and build</h2><pre>"
              + html.escape(json.dumps(machine or {}, indent=2)) + "</pre>"
              "<p>Compute provided by <a href='https://www.open-sai.com/'>SAI</a>.</p></html>\n")
    (output / "index.html").write_text(report)
    return summary
