# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure functions behind the Clinical Acceleration tiers (no network)."""

import base64
import io
import re
import zipfile

import pytest

from sas_mcp_server.helpers import clinical_helpers as ch

# --- paths, names, versions -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "/"), ("", "/"), ("/", "/"), ("a/b/", "/a/b"), ("//a//b", "/a/b"), ("\\a\\b.sas", "/a/b.sas")],
)
def test_normalize_path(raw, expected):
    assert ch.normalize_path(raw) == expected


@pytest.mark.parametrize("raw", ["/a/../b", "./x", "/a/./b"])
def test_normalize_path_refuses_navigation(raw):
    with pytest.raises(ch.ClinicalError):
        ch.normalize_path(raw)


def test_split_and_url_path():
    assert ch.split_path("/a/b.sas") == ("/a", "b.sas")
    assert ch.split_path("/b.sas") == ("/", "b.sas")
    assert ch.url_path("/My Study/a#b.sas") == "/My%20Study/a%23b.sas"


def test_check_location():
    assert ch.check_location(" Workspace ") == "workspace"
    with pytest.raises(ch.ClinicalError):
        ch.check_location("files")


def test_file_and_folder_names():
    ch.validate_file_name("adsl.sas")
    for bad in ("adsl", "a;b.sas", "a?.sas"):
        with pytest.raises(ch.ClinicalError):
            ch.validate_file_name(bad)
    ch.validate_folder_name("programs v2")
    for bad in ("", ".hidden", " lead", "trail ", 'a"b', "a:b", "a　b", "x" * 256):
        with pytest.raises(ch.ClinicalError):
            ch.validate_folder_name(bad)


def test_comment_and_version_validation():
    assert ch.validate_comment("  why  ") == "why"
    with pytest.raises(ch.ClinicalError):
        ch.validate_comment("x" * 1025)
    assert ch.validate_version("minor") == "MINOR"
    assert ch.validate_version("2.10") == "2.10"
    assert ch.validate_version(" ") is None
    with pytest.raises(ch.ClinicalError):
        ch.validate_version("v1")
    with pytest.raises(ch.ClinicalError):
        ch.validate_version("MAJOR", allow_bump=False)


def test_next_version_and_sorting():
    assert ch.next_version("major", "2.3") == "3.0"
    assert ch.next_version("minor", "2.3") == "2.4"
    assert ch.next_version("minor", "-") == "0.1"
    with pytest.raises(ch.ClinicalError):
        ch.next_version("minor", "x.y")
    assert sorted(["9.1", "10.0", "1.2"], key=ch.version_key, reverse=True) == ["10.0", "9.1", "1.2"]
    assert ch.version_key("abc") == (-1,)


# --- content -----------------------------------------------------------------------------


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_unwrap_single_file_zip():
    assert ch.unwrap_single_file_zip(_zip({"a.sas": b"data x;"}), "a.sas") == b"data x;"
    two = _zip({"a": b"1", "b": b"2"})
    assert ch.unwrap_single_file_zip(two, "a.sas") == two
    one = _zip({"a": b"1"})
    assert ch.unwrap_single_file_zip(one, "bundle.zip") == one  # a real zip stays a zip
    assert ch.unwrap_single_file_zip(b"PK\x03\x04garbage", "a.sas") == b"PK\x03\x04garbage"
    assert ch.unwrap_single_file_zip(b"plain", "a.sas") == b"plain"


def test_decode_text():
    assert ch.decode_text("ä".encode()) == "ä"
    assert ch.decode_text("caf\xe9".encode("latin-1")) == "caf\xe9"
    assert ch.decode_text(b"\x00\x01\x02") is None
    assert ch.decode_text(b"\xff\x01\x02") is None


@pytest.mark.parametrize(
    "payload",
    [["WRITE", "READ"], "READ, WRITE", {"permissions": ["READ", "WRITE"]}, {"items": ["WRITE", "READ", "READ"]}],
)
def test_parse_access_list(payload):
    assert ch.parse_access_list(payload) == ["READ", "WRITE"]


def test_parse_access_list_unknown_shape():
    assert ch.parse_access_list(42) is None


# --- shaping --------------------------------------------------------------------------


def test_repo_item_summary_marks_your_checkout():
    item = {
        "name": "run.sas",
        "path": "/s/run.sas",
        "checkedOut": True,
        "checkedOutBy": "michael",
        "tags": ["t"],
        "links": [{"rel": "self"}],
        "description": "",
        "synchronizationInfo": {"workspaceStatus": "IN_SYNC"},
    }
    out = ch.repo_item_summary(item, "michael")
    assert out["checkedOutByYou"] is True
    assert out["workspaceStatus"] == "IN_SYNC"
    assert "links" not in out and "description" not in out
    assert ch.repo_item_summary(item, "other")["checkedOutByYou"] is False


def test_workspace_item_summary():
    item = {
        "name": "a.sas",
        "path": "/a.sas",
        "type": "FILE",
        "synchronizationInfo": {"checkedOut": True, "checkedOutBy": "u", "syncable": "ALLOW", "fileVersion": ""},
    }
    out = ch.workspace_item_summary(item, "u")
    assert out["sync"] == {"checkedOut": True, "checkedOutBy": "u", "checkedOutByYou": True}
    assert "sync" not in ch.workspace_item_summary({"name": "x", "synchronizationInfo": {}})


@pytest.mark.parametrize(
    ("completion", "ended", "status"),
    [
        ("INFO", True, "completed"),
        ("WARN", True, "completed_with_warnings"),
        ("ERROR", True, "failed"),
        ("", False, "still_running"),
    ],
)
def test_action_result(completion, ended, status):
    summary = {"id": "act-1", "completionStatus": completion, "message": "m", "action": "Copy"}
    if ended:
        summary["endTimeStamp"] = "2026-11-13T12:00:00Z"
    detail = {"itemIdentifier": "/a.sas", "completionStatus": completion, "message": "done"}
    out = ch.action_result({"summary": summary, "details": [detail]})
    assert out["status"] == status
    assert out["items"] == [{"path": "/a.sas", "status": completion, "message": "done"}]
    assert ("action_status_id" in out) == (not ended)


def test_batch_bodies_match_what_each_action_takes():
    paths = ["/a.sas"]
    assert ch.batch_body("UNDO_CHECKOUT", paths) == paths
    assert ch.batch_body("REPOSITORY_CHECK_OUT", paths, copy_content=False) == {"paths": paths, "copyContent": False}
    assert ch.batch_body("COPY_FOLDER_STRUCTURE", paths) == {"paths": paths, "includeSubfolders": True}
    assert ch.batch_body("COPY_SPECIFIC_VERSION", paths, file_version="1.2") == [
        {"path": "/a.sas", "fileVersion": "1.2"}
    ]
    assert ch.batch_body("ENABLE_VERSIONING", paths, comment="c", file_version="1.0") == {
        "comment": "c",
        "fileSpecifications": [{"path": "/a.sas", "fileVersion": "1.0"}],
    }
    assert ch.batch_body("DISABLE_VERSIONING", paths, comment="c") == {"comment": "c", "paths": paths}
    with pytest.raises(ValueError):
        ch.batch_body("NOPE", paths)


# --- check-in planning ----------------------------------------------------------------


def _cand(path, **info):
    return {"path": path, "name": path.rsplit("/", 1)[1], "synchronizationInfo": info}


def test_plan_check_in_sorts_and_explains():
    candidates = [
        _cand("/s/new.sas", workspaceStatus="NOT_SYNCED"),
        _cand("/s/mine.sas", checkedOut=True, checkedOutBy="me", fileVersion="2.3", workspaceStatus="OUT_OF_SYNC"),
        _cand("/s/unver.sas", checkedOut=True, checkedOutBy="me", fileVersion="-"),
        _cand("/s/theirs.sas", checkedOut=True, checkedOutBy="bob", checkedOutByDisplayName="Bob"),
        _cand("/s/copy.sas", workspaceStatus="OUT_OF_SYNC"),
        _cand("/s/same.sas", workspaceStatus="IN_SYNC"),
    ]
    body, included, skipped = ch.plan_check_in(
        candidates, ["/s", "/other.sas"], "me", bump="major", new_file_version="1.0", comment="why"
    )
    assert body["comment"] == "why" and body["includeNewFiles"] is True
    assert body["fileSpecifications"] == [
        {"path": "/s/new.sas", "type": "FILE", "fileVersion": "1.0"},
        {"path": "/s/mine.sas", "type": "FILE", "fileVersion": "3.0"},
        {"path": "/s/unver.sas", "type": "FILE", "fileVersion": "1.0"},
    ]
    assert [i["kind"] for i in included] == ["new", "new_version", "unversioned"]
    assert skipped == [
        {"path": "/s/theirs.sas", "reason": "checked_out_by_other", "by": "Bob"},
        {"path": "/s/copy.sas", "reason": "not_checked_out"},
        {"path": "/other.sas", "reason": "not_offered"},
    ]


def test_plan_check_in_without_versions_leaves_files_unversioned():
    body, _, _ = ch.plan_check_in([_cand("/n.sas")], ["/n.sas"], None, bump="minor", new_file_version=None, comment="")
    assert body["fileSpecifications"] == [{"path": "/n.sas", "type": "FILE"}]


# --- datasets into WORK -----------------------------------------------------------------


def _decode_datalines(programs):
    data = b""
    for program in programs:
        lines = program.split("datalines;\n", 1)[1].split("\n;\n", 1)[0].splitlines()
        data += b"".join(base64.b64decode(line) for line in lines if line)
    return data


def test_upload_to_work_code_round_trips_the_bytes(monkeypatch):
    monkeypatch.setattr(ch, "BYTES_PER_CHUNK", 57 * 2)
    data = bytes(range(256))
    programs = ch.upload_to_work_code("lib", "x.sas7bdat", data)
    assert len(programs) == 3
    assert programs[0].startswith('data _null_; length d $1024; d=dcreate("lib", pathname("work")); run;')
    assert "recfm=n;" in programs[0] and "recfm=n mod;" in programs[1]
    assert _decode_datalines(programs) == data
    assert len(ch.upload_to_work_code("lib", "e.xpt", b"")) == 1


def test_work_path_literal_refuses_odd_names():
    assert ch.work_path_literal("lib", "a.xpt") == '"%sysfunc(pathname(work))/lib/a.xpt"'
    with pytest.raises(ch.ClinicalError):
        ch.work_path_literal("lib", "a b.xpt")


def test_transport_formats():
    assert ch.detect_transport_format(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000") == "xport5"
    assert ch.detect_transport_format(b"HEADER RECORD*******LIBV8   HEADER RECORD!!!!!!!000") == "xport8"
    assert ch.detect_transport_format(b"**COMPRESSED** **COMPRESSED** ...") == "cport"
    assert ch.detect_transport_format(b"nope") is None
    v5 = ch.transport_load_code("xport5", "_src", "ae.xpt", "_caae")
    assert "libname _XPTIN xport" in v5 and "proc copy in=_XPTIN out=_caae" in v5
    assert "proc datasets lib=_caae kill" in v5
    assert "%xpt2loc(libref=_caae" in ch.transport_load_code("xport8", "_src", "ae.xpt", "_caae")
    assert "proc cimport" in ch.transport_load_code("cport", "_src", "ae.xpt", "_caae")


def test_find_sas_errors_joins_continuations():
    log = "NOTE: ok\nERROR 22-322: Syntax error, expecting\n       one of the following.\nNOTE: x\nERROR: second"
    assert ch.find_sas_errors(log) == ["ERROR 22-322: Syntax error, expecting one of the following.", "ERROR: second"]


# --- jobs ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "category"),
    [
        ("RUNNING_TASKS", "running"),
        ("COMPLETED_SUCCESSFUL", "successful"),
        ("COMPLETED_PROGRAM_WARNINGS", "warnings"),
        ("COMPLETED_CANCELED", "canceled"),
        ("COMPLETED_SUCCESSFUL_CHECKIN_ERROR", "error"),
        ("COMPLETED_FAILED", "error"),
        (None, "running"),
    ],
)
def test_run_category(state, category):
    assert ch.run_category(state) == category


def test_run_kinds_and_reruns():
    assert ch.run_kind("INTERACTIVE_TEST") == "test"
    assert ch.run_kind("INTERACTIVE_RUN_AND_POPULATE_RELATIVE") == "run_and_populate"
    assert ch.run_kind("SCHEDULER") == "scheduled"
    assert ch.run_kind("INTERACTIVE") == "run"
    assert ch.rerun_options("INTERACTIVE_TEST") == ("test", False)
    assert ch.rerun_options("INTERACTIVE_RUN_AND_POPULATE_RELATIVE") == ("run_and_populate", True)
    assert ch.rerun_options("INTERACTIVE_RUN_AND_POPULATE") == ("run_and_populate", False)
    assert ch.rerun_options("SCHEDULER") == ("run", False)


def test_normalize_submission_flat_and_nested():
    flat = {
        "id": "r1",
        "name": "j.cjob",
        "jobPath": "/w/j.cjob",
        "jobResourceLocation": "WORKSPACE",
        "jobVersion": "-",
        "state": "COMPLETED_SUCCESSFUL_CHECKIN_ERROR",
        "jobRunContext": "INTERACTIVE",
        "manifest": {"path": "/w/j.cmnf"},
        "checkinFailures": [{"itemIdentifier": "/w/out.lst", "message": "checked out"}],
    }
    out = ch.normalize_submission(flat)
    assert out["location"] == "workspace" and "job_version" not in out
    assert out["category"] == "error" and out["manifest_path"] == "/w/j.cmnf"
    assert out["checkin_failures"] == [{"path": "/w/out.lst", "message": "checked out"}]
    nested = ch.normalize_submission(
        {"id": "r2", "job": {"name": "k", "path": "/r/k.cjob", "resourceLocation": "REPOSITORY"}}
    )
    assert nested["job_path"] == "/r/k.cjob" and nested["location"] == "repository"


def test_job_paths():
    assert ch.job_reference("/w/a/j.cjob", "workspace") == {
        "name": "j.cjob",
        "path": "/w/a/j.cjob",
        "location": "/w/a",
        "resourceLocation": "WORKSPACE",
    }
    assert ch.resolve_job_path("/w/a/j.cjob", "./p.sas") == "/w/a/p.sas"
    assert ch.resolve_job_path("/w/a/j.cjob", "../data") == "/w/data"
    assert ch.resolve_job_path("/w/a/j.cjob", "/abs/x") == "/abs/x"


PARAMS = [
    {"name": "study", "type": "CHARACTER", "defaultValue": "S1"},
    {"name": "pw", "type": "CHARACTER_MASKED"},
    {"name": "n", "type": "NUMERIC", "defaultValue": "3"},
    {"name": "cutoff", "type": "DATE", "defaultValue": "2026-01-01T00:00:00Z"},
    {"name": "infile", "type": "FILE", "defaultValue": {"path": "./data/a.sas7bdat"}},
    {"name": "outdir", "type": "FOLDER", "defaultValue": {"path": "../out", "includeSubfolders": True}},
]


def test_encode_parameter_values_per_type():
    values = ch.encode_parameter_values(
        PARAMS,
        {"PW": "secret", "n": 5, "cutoff": "2026-11-13", "infile": {"path": "/r/x.xpt", "version": "1.0"}},
        "/w/a/j.cjob",
    )
    by = {v["name"]: v["value"] for v in values}
    assert by == {
        "study": "S1",
        "pw": "secret",
        "n": "5",
        "cutoff": "2026-11-13T00:00:00Z",
        "infile": {"path": "/r/x.xpt", "version": "1.0"},
        "outdir": {"path": "/w/out", "includeSubfolders": True},
    }
    assert all(set(v) == {"name", "type", "value"} for v in values)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "'pw'"),  # no default
        ({"pw": "x", "nope": 1}, "no parameter named nope"),
        ({"pw": "x", "n": "five"}, "not a number"),
        ({"pw": "x", "cutoff": "13/11/2026"}, "YYYY-MM-DD"),
        ({"pw": "x", "infile": {"version": "1"}}, "needs a path"),
    ],
)
def test_encode_parameter_values_refusals(values, message):
    with pytest.raises(ch.ClinicalError, match=re.escape(message)):
        ch.encode_parameter_values(PARAMS, values, "/w/j.cjob")


def test_validate_job():
    assert ch.validate_job({"programs": [{"path": "./a.sas"}], "parameters": [{"name": "x", "type": "DATE"}]}) == []
    problems = ch.validate_job(
        {
            "programs": [{}],
            "parameters": [
                {"name": "1x", "type": "TEXT"},
                {"name": "a", "type": "NUMERIC"},
                {"name": "A", "type": "NUMERIC"},
            ],
            "publishCheckin": {"fileVersion": "v2"},
        }
    )
    assert len(problems) == 5
    assert ch.validate_job({})[0].startswith("The job runs no programs")


def test_program_jobs():
    job = ch.program_job_definition("/w/adsl.sas", "/w/adsl.cjob")
    assert job["resource"]["resourceLocation"] == "WORKSPACE"
    assert job["programs"] == [{"path": "/w/adsl.sas", "name": "adsl.sas", "location": "/w"}]
    assert ch.job_runs_only({"programs": [{"path": "./adsl.sas"}]}, "/w/adsl.sas", "/w/adsl.cjob")
    assert not ch.job_runs_only({"programs": [{"path": "./a.sas"}, {"path": "./b.sas"}]}, "/w/a.sas", "/w/a.cjob")


def test_fill_checkin_locations_from_cmnf():
    manifest = {"checkinSpecification": {"fileVersion": "MAJOR", "locations": []}}
    cmnf = {
        "checkinSpecification": {
            "enableVersioningOnNewFiles": True,
            "locations": [
                "/r/out",
                {"path": "/r/tlf", "includeSubfolders": True},
                {"resource": {"path": "/r/x"}},
                {},
                3,
            ],
        }
    }
    filled = ch.fill_checkin_locations(manifest, cmnf)
    assert filled["checkinSpecification"] == {
        "enableVersioningOnNewFiles": True,
        "fileVersion": "MAJOR",
        "locations": [{"path": "/r/out"}, {"path": "/r/tlf", "includeSubfolders": True}, {"path": "/r/x"}],
    }
    has = {"checkinSpecification": {"locations": [{"path": "/a"}]}}
    assert ch.fill_checkin_locations(has, cmnf) is has
    assert ch.fill_checkin_locations(manifest, None) is manifest
    assert ch.fill_checkin_locations(manifest, {"checkinSpecification": {"locations": []}}) is manifest


def test_summarize_manifest_reads_like_an_audit_record():
    res = lambda path, **kw: {"resource": {"path": path, "resourceLocation": "REPOSITORY", **kw}}  # noqa: E731
    manifest = {
        "type": "REPOSITORY",
        "jobSubmission": {"id": "r1", "state": "COMPLETED_SUCCESSFUL", "jobRunContext": "INTERACTIVE"},
        "job": res("/r/j.cjob", fileVersion="1.0"),
        "programs": [{**res("/r/a.sas", fileVersion="-"), "executionState": "SUCCESSFUL", "executionTime": 41}],
        "parameters": [{"name": "pw", "type": "CHARACTER_MASKED", "value": "secret"}],
        "inputs": [{**res("/r/adsl.sas7bdat", fileVersion="2.0"), "digest": {"value": "abc"}}],
        "outputs": [res("/r/j.log"), res("/r/t.rtf")],
        "logs": [res("/r/j.log")],
        "externalInputs": [{"name": "fmt", "location": "/opt", "referencingProgram": "a.sas"}],
        "manifest": res("/r/j.cmnf"),
    }
    out = ch.summarize_manifest(manifest)
    assert out["parameters"] == [{"name": "pw", "type": "CHARACTER_MASKED", "value": "*****"}]
    assert out["inputs"] == [{"path": "/r/adsl.sas7bdat", "location": "repository", "version": "2.0", "md5": "abc"}]
    assert [o["path"] for o in out["outputs"]] == ["/r/t.rtf"]
    assert out["programs"][0]["result"] == "SUCCESSFUL" and "version" not in out["programs"][0]
    assert out["manifest_path"] == "/r/j.cmnf" and out["location"] == "repository"


# --- schedules ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trigger", "expected"),
    [
        (
            {"type": "datelist", "dates": "2026-12-01", "hours": 6, "minutes": "05", "timeZone": "UTC"},
            {"type": "DATELIST", "timeZone": "UTC", "dates": ["2026-12-01"], "hours": "6", "minutes": "5"},
        ),
        (
            {"type": "MINUTELY", "interval": 15, "timeZone": "UTC"},
            {"type": "MINUTELY", "timeZone": "UTC", "interval": 15},
        ),
        (
            {"type": "HOURLY", "minutes": 5, "timeZone": "UTC", "start": "2026-12-01", "numOccurrences": 3},
            {
                "type": "HOURLY",
                "timeZone": "UTC",
                "interval": 1,
                "minutes": "5",
                "start": "2026-12-01",
                "numOccurrences": 3,
            },
        ),
        (
            {"type": "WEEKLY", "daysOfWeek": ["monday", "THU"], "hours": 7, "minutes": 0, "timeZone": "UTC"},
            {
                "type": "WEEKLY",
                "timeZone": "UTC",
                "interval": 1,
                "hours": "7",
                "minutes": "0",
                "daysOfWeek": ["MON", "THU"],
            },
        ),
        (
            {
                "type": "MONTHLY",
                "occurrence": "first",
                "daysOfWeek": ["MON"],
                "hours": 1,
                "minutes": 0,
                "timeZone": "UTC",
            },
            {
                "type": "MONTHLY",
                "timeZone": "UTC",
                "interval": 1,
                "hours": "1",
                "minutes": "0",
                "occurrence": "FIRST",
                "daysOfWeek": ["MON"],
            },
        ),
        (
            {"type": "YEARLY", "dayOfMonth": 15, "monthOfYear": "june", "hours": 0, "minutes": 0, "timeZone": "UTC"},
            {
                "type": "YEARLY",
                "timeZone": "UTC",
                "interval": 1,
                "hours": "0",
                "minutes": "0",
                "dayOfMonth": 15,
                "monthOfYear": "JUNE",
            },
        ),
    ],
)
def test_build_schedule_trigger(trigger, expected):
    assert ch.build_schedule_trigger(trigger) == expected


@pytest.mark.parametrize(
    "trigger",
    [
        {"type": "SOMETIMES", "timeZone": "UTC"},
        {"type": "DAILY", "hours": 1, "minutes": 0},
        {"type": "DAILY", "minutes": 0, "timeZone": "UTC"},
        {"type": "DAILY", "hours": 24, "minutes": 0, "timeZone": "UTC"},
        {"type": "DAILY", "hours": "x", "minutes": 0, "timeZone": "UTC"},
        {"type": "DATELIST", "dates": ["soon"], "hours": 1, "minutes": 0, "timeZone": "UTC"},
        {"type": "WEEKLY", "hours": 1, "minutes": 0, "timeZone": "UTC"},
        {"type": "MONTHLY", "hours": 1, "minutes": 0, "timeZone": "UTC"},
        {"type": "YEARLY", "dayOfMonth": 1, "hours": 1, "minutes": 0, "timeZone": "UTC"},
        {"type": "DAILY", "hours": 1, "minutes": 0, "timeZone": "UTC", "end": "tomorrow"},
    ],
)
def test_build_schedule_trigger_refusals(trigger):
    with pytest.raises(ch.ClinicalError):
        ch.build_schedule_trigger(trigger)


def test_summarize_schedule_and_values():
    schedule = {
        "id": "s1",
        "name": "nightly",
        "status": "ENABLED",
        "createdBy": "michael",
        "jobAvailable": True,
        "jobInfo": {
            "job": {"path": "/r/j.cjob", "resourceLocation": "REPOSITORY"},
            "parameterValues": [
                {"name": "pw", "type": "CHARACTER_MASKED", "value": "s", "label": "null", "version": 1},
                {"name": "n", "type": "NUMERIC", "value": "1", "label": "null"},
            ],
        },
        "trigger": {"version": 1, "type": "DAILY", "startDate": "2026-12-01"},
    }
    out = ch.summarize_schedule(schedule, "michael")
    assert out["yours"] is True and out["location"] == "repository"
    assert out["trigger"] == {"type": "DAILY", "start": "2026-12-01"}
    assert out["parameter_values"][0]["value"] == "*****"
    assert ch.schedule_parameter_values(schedule) == [
        {"name": "pw", "type": "CHARACTER_MASKED", "value": "s"},
        {"name": "n", "type": "NUMERIC", "value": "1"},
    ]


def test_parse_json_bytes():
    assert ch.parse_json_bytes(b'\xef\xbb\xbf{"a": 1}', "x") == {"a": 1}
    with pytest.raises(ch.ClinicalError):
        ch.parse_json_bytes(b"[1]", "x")
    with pytest.raises(ch.ClinicalError):
        ch.parse_json_bytes(b"{", "x")
