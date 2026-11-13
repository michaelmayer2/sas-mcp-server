# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 12 (Clinical Repository) and Tier 13 (Clinical Jobs) tools against a routed fake CA.

The fake answers with real ``httpx.Response`` objects, so headers, binary
content and error bodies behave as they do against SAS Viya. The requests the
tools send are checked against what the live services accept (see the
module docstring of ``helpers/clinical_helpers.py``).
"""

import io
import json
import re
import zipfile
from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp import Client, FastMCP

from sas_mcp_server import config
from sas_mcp_server.helpers import clinical_helpers as ch
from sas_mcp_server.tools import clinical_jobs, clinical_repository, register_tools

RUN = {"id": "item-run", "name": "run.sas", "path": "/S/programs/run.sas", "primaryType": "FILE", "versioned": True}
FOLDER = {"id": "item-prog", "name": "programs", "path": "/S/programs", "primaryType": "FOLDER"}
CONTEXT = {"id": "item-s", "name": "S", "path": "/S", "primaryType": "CONTEXT"}
ENDED = {"summary": {"id": "act-1", "endTimeStamp": "t", "completionStatus": "INFO", "message": "1 of 1 processed"}}


class FakeCA:
    """An async client double: routes by method and path regex, records what was sent."""

    def __init__(self):
        self.routes: list[tuple[str, re.Pattern, object]] = []
        self.sent: list[tuple[str, str, dict]] = []
        self.headers = {"Authorization": "Bearer test-token"}
        self.add("GET", r"^/clinicalRepository/$", {"links": []})
        self.add("GET", r"^/clinicalJobs/submissions$", {"items": [], "links": []})
        self.add("GET", r"^/clinicalActionStatus/actionstatus/", ENDED)

    def add(self, method, pattern, body=None, status=200, headers=None):
        """``body`` may be data, bytes, or ``fn(path, kwargs) -> (status, data)``."""
        self.routes.insert(0, (method, re.compile(pattern), (body, status, headers or {})))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def _request(self, method, url, **kwargs):
        path = url.removeprefix(config.VIYA_ENDPOINT)
        self.sent.append((method, path, kwargs))
        for m, pattern, (body, status, headers) in self.routes:
            if m == method and pattern.search(path.split("?")[0]):
                if callable(body):
                    status, body = body(path, kwargs)
                request = httpx.Request(method, url, params=kwargs.get("params"))
                if isinstance(body, bytes):
                    return httpx.Response(status, content=body, headers=headers, request=request)
                if isinstance(body, str):
                    return httpx.Response(status, text=body, headers=headers, request=request)
                return httpx.Response(status, json=body, headers=headers, request=request)
        raise AssertionError(f"unrouted {method} {path}")

    async def get(self, url, **kw):
        return await self._request("GET", url, **kw)

    async def post(self, url, **kw):
        return await self._request("POST", url, **kw)

    async def put(self, url, **kw):
        return await self._request("PUT", url, **kw)

    async def patch(self, url, **kw):
        return await self._request("PATCH", url, **kw)

    async def delete(self, url, **kw):
        return await self._request("DELETE", url, **kw)

    def calls(self, method, pattern):
        return [(path, kw) for m, path, kw in self.sent if m == method and re.search(pattern, path)]

    def batch(self, action):
        """The body of the batch action ``action`` that was started."""
        for _, kw in self.calls("POST", r"/items/batch$"):
            if kw["params"]["action"] == action:
                return kw["json"]
        raise AssertionError(f"no {action} batch")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 1.0
        return self.now


@pytest.fixture
def ca(monkeypatch):
    fake = FakeCA()
    monkeypatch.setattr("sas_mcp_server.tools._common.make_client", lambda token: fake)
    monkeypatch.setattr("sas_mcp_server.tools._common.get_cached_session", AsyncMock(return_value="sess-1"))
    monkeypatch.setattr(ch, "_sleep", AsyncMock())
    monkeypatch.setattr(ch, "_clock", Clock())
    for module in (clinical_repository, clinical_jobs):
        monkeypatch.setattr(module, "resolve_user_id", AsyncMock(return_value="michael"))
    ch.clear_service_cache()
    mcp = FastMCP("clinical")

    async def get_token(ctx):
        return "test-token"

    register_tools(mcp, get_token, tiers="12,13", read_only=False, apps=False)
    yield mcp, fake
    ch.clear_service_cache()


async def call(mcp, tool, args=None):
    async with Client(mcp) as client:
        result = await client.call_tool(tool, args or {}, raise_on_error=False)
    if result.is_error:
        return {"error": result.content[0].text}
    if result.data is not None:
        return result.data
    return result.content[0]


def repo_path(fake, item):
    """Route ``/repository/paths?path=`` lookups for ``item``."""

    def lookup(path, kw):
        wanted = kw["params"]["path"]
        for it in item if isinstance(item, list) else [item]:
            if it["path"] == wanted:
                return 200, it
        return 404, {"errorCode": 145016, "message": f'No resource with the path "{wanted}" was found.'}

    fake.add("GET", r"/clinicalRepository/repository/paths$", lookup)


# --- detection --------------------------------------------------------------------------


async def test_missing_service_is_reported_not_raised(ca):
    mcp, fake = ca
    fake.add("GET", r"^/clinicalRepository/$", {"message": "no"}, status=404)
    out = await call(mcp, "clinical_list_items", {})
    assert out["status"] == "service_unavailable"
    assert "Clinical Acceleration Repository" in out["service"]


async def test_jobs_tools_need_job_management(ca):
    mcp, fake = ca
    fake.add("GET", r"^/clinicalJobs/submissions$", {"message": "no"}, status=404)
    out = await call(mcp, "clinical_list_job_runs", {})
    assert out["status"] == "service_unavailable" and "Job Management" in out["message"]


# --- browse and read --------------------------------------------------------------------


async def test_list_repository_folder(ca):
    mcp, fake = ca
    repo_path(fake, FOLDER)
    child = {**RUN, "checkedOut": True, "checkedOutBy": "michael", "links": [{"rel": "next"}]}
    fake.add("GET", r"/repository/items/item-prog/children$", {"items": [child], "links": []})
    out = await call(mcp, "clinical_list_items", {"path": "/S/programs/"})
    assert out["path"] == "/S/programs" and out["truncated"] is False
    assert out["items"][0]["checkedOutByYou"] is True and "links" not in out["items"][0]
    ((_, kw),) = fake.calls("GET", r"/children$")
    assert kw["params"]["includeSynchronizationInfo"] == "true"


async def test_list_follows_next_links(ca):
    mcp, fake = ca
    repo_path(fake, FOLDER)

    def pages(path, kw):
        if "start=1" in path:
            return 200, {"items": [{"name": "b.sas", "path": "/S/programs/b.sas"}], "links": []}
        return 200, {
            "items": [{"name": "a.sas", "path": "/S/programs/a.sas"}],
            "links": [{"rel": "next", "href": "/clinicalRepository/repository/items/item-prog/children?start=1"}],
        }

    fake.add("GET", r"/repository/items/item-prog/children$", pages)
    out = await call(mcp, "clinical_list_items", {"path": "/S/programs"})
    assert [i["name"] for i in out["items"]] == ["a.sas", "b.sas"]


async def test_list_items_refuses_a_file_and_bad_paths(ca):
    mcp, fake = ca
    repo_path(fake, RUN)
    assert (await call(mcp, "clinical_list_items", {"path": RUN["path"]}))["status"] == "refused"
    assert (await call(mcp, "clinical_list_items", {"path": "/S/../x"}))["status"] == "refused"
    out = await call(mcp, "clinical_list_items", {"path": "/nope"})
    assert out["status"] == "not_found" and "clinical_search_repository" in out["message"]


async def test_list_workspace_folder(ca):
    mcp, fake = ca
    item = {
        "name": "run.sas",
        "path": "/S/programs/run.sas",
        "type": "FILE",
        "synchronizationInfo": {"checkedOut": False},
    }
    fake.add("GET", r"/workspaces/@currentUser/items/S/programs$", {"items": [item], "links": []})
    out = await call(mcp, "clinical_list_items", {"path": "/S/programs", "location": "workspace"})
    assert out["items"][0]["path"] == "/S/programs/run.sas"
    fake.add("GET", r"/workspaces/@currentUser/items/gone$", {"message": "x"}, status=404)
    assert (await call(mcp, "clinical_list_items", {"path": "/gone", "location": "workspace"}))["status"] == "not_found"


async def test_get_repository_item_with_access(ca):
    mcp, fake = ca
    repo_path(fake, RUN)
    sync = {"workspaceStatus": "OUT_OF_SYNC", "checkedOut": True, "checkedOutBy": "bob", "syncable": "ALLOW"}
    fake.add("GET", r"/repository/items/item-run$", {**RUN, "synchronizationInfo": sync}, headers={"ETag": '"e7"'})
    fake.add("GET", r"/authorizations/item-run/permissions$", ["READ", "WRITE"])
    fake.add("GET", r"/authorizations/item-run/privileges$", {"errorCode": 145000}, status=500)
    out = await call(mcp, "clinical_get_item", {"path": RUN["path"]})
    assert out["your_permissions"] == ["READ", "WRITE"]
    assert out["your_versioning_privileges"] is None
    assert out["workspace"] == {"workspaceStatus": "OUT_OF_SYNC", "checkedOut": True, "checkedOutBy": "bob"}


async def test_access_answers(ca):
    mcp, fake = ca
    repo_path(fake, RUN)
    fake.add("GET", r"/repository/items/item-run$", RUN)
    fake.add("GET", r"/authorizations/item-run/permissions$", {"message": "no"}, status=403)
    fake.add("GET", r"/authorizations/item-run/privileges$", {"errorCode": 145191}, status=404)
    out = await call(mcp, "clinical_get_item", {"path": RUN["path"]})
    assert out["your_permissions"] == ["READ"] and out["your_versioning_privileges"] == []


async def test_get_workspace_item(ca):
    mcp, fake = ca
    fake.add("GET", r"/workspaces/@currentUser/items/a.sas$", {"name": "a.sas", "path": "/a.sas", "type": "FILE"})
    out = await call(mcp, "clinical_get_item", {"path": "/a.sas", "location": "workspace"})
    assert out == {"location": "workspace", "name": "a.sas", "path": "/a.sas", "type": "FILE"}
    fake.add("GET", r"/workspaces/@currentUser/items/b.sas$", {}, status=404)
    assert (await call(mcp, "clinical_get_item", {"path": "/b.sas", "location": "workspace"}))["status"] == "not_found"


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


async def test_read_repository_file_latest_and_version(ca):
    mcp, fake = ca
    repo_path(fake, RUN)
    fake.add("POST", r"/repository/items/item-run/content$", _zip({"run.sas": b"data x; run;"}))
    fake.add("GET", r"/repository/items/item-run/versions/1.0/content$", b"data old; run;")
    out = await call(mcp, "clinical_read_file", {"path": RUN["path"]})
    assert out["content"] == "data x; run;" and out["version"] == "latest"
    ((_, kw),) = fake.calls("POST", r"/content$")
    assert kw["json"] == [""]  # the content read is a POST with a one-empty-string body
    out = await call(mcp, "clinical_read_file", {"path": RUN["path"], "version": "1.0"})
    assert out["content"] == "data old; run;"
    fake.add("GET", r"/versions/9.9/content$", {"message": "no"}, status=404)
    assert (await call(mcp, "clinical_read_file", {"path": RUN["path"], "version": "9.9"}))["status"] == "not_found"


async def test_read_refusals_and_binary(ca):
    mcp, fake = ca
    repo_path(fake, [FOLDER, {**RUN, "path": "/S/ae.sas7bdat", "name": "ae.sas7bdat", "id": "item-ae"}])
    assert (await call(mcp, "clinical_read_file", {"path": "/S/programs"}))["status"] == "refused"
    out = await call(mcp, "clinical_read_file", {"path": "/a.sas", "location": "workspace", "version": "1.0"})
    assert out["status"] == "refused"
    fake.add("POST", r"/repository/items/item-ae/content$", b"\x00\x01binary")
    resource = await call(mcp, "clinical_read_file", {"path": "/S/ae.sas7bdat"})
    assert resource.type == "resource" and resource.resource.blob


async def test_read_too_large(ca, monkeypatch):
    mcp, fake = ca
    monkeypatch.setattr(config, "MAX_EXPORT_INLINE_BYTES", 4)
    fake.add("POST", r"/workspaces/@currentUser/items/content$", b"data x; run;")
    out = await call(mcp, "clinical_read_file", {"path": "/a.sas", "location": "workspace"})
    assert out["status"] == "export_too_large"


async def test_read_workspace_missing(ca):
    mcp, fake = ca
    fake.add("POST", r"/workspaces/@currentUser/items/content$", {"message": "no"}, status=404)
    out = await call(mcp, "clinical_read_file", {"path": "/a.sas", "location": "workspace"})
    assert out["status"] == "not_found"


async def test_download_repository_items(ca):
    mcp, fake = ca
    repo_path(fake, [RUN, FOLDER])
    fake.add("POST", r"/repository/files/content$", _zip({"a": b"1", "b": b"2"}))
    resource = await call(mcp, "clinical_download_items", {"paths": [RUN["path"], FOLDER["path"]]})
    assert resource.resource.mimeType == "application/zip"
    ((_, kw),) = fake.calls("POST", r"/files/content$")
    assert kw["json"] == {"filter": "in(id,'item-run','item-prog')"}


async def test_download_workspace_items_announces_the_download(ca):
    mcp, fake = ca
    fake.add(
        "POST",
        r"/workspaces/@currentUser/items/batch$",
        {},
        headers={"Location": "/clinicalActionStatus/actionstatus/act-9"},
    )
    fake.add("POST", r"/workspaces/@currentUser/items/content$", b"%let x=1;")
    resource = await call(mcp, "clinical_download_items", {"paths": "/a.sas", "location": "workspace"})
    assert resource.type == "resource"
    assert fake.batch("DOWNLOAD") == ["/a.sas"]
    ((_, kw),) = fake.calls("POST", r"/items/content$")
    assert kw["json"] == {"paths": ["/a.sas"], "actionStatusId": "act-9"}
    fake.add("GET", r"/actionstatus/", {"summary": {"endTimeStamp": "t", "completionStatus": "ERROR", "message": "no"}})
    out = await call(mcp, "clinical_download_items", {"paths": ["/a.sas"], "location": "workspace"})
    assert out["status"] == "failed"


async def test_list_versions_sorted_numerically(ca):
    mcp, fake = ca
    repo_path(fake, [RUN, {**RUN, "id": "item-u", "path": "/S/u.sas", "versioned": False}])
    versions = [{"fileVersion": v, "comment": "c", "fileSize": 10} for v in ("9.1", "10.0", "1.0")]
    fake.add("GET", r"/repository/items/item-run/versions$", {"items": versions, "links": []})
    out = await call(mcp, "clinical_list_versions", {"path": RUN["path"]})
    assert [v["fileVersion"] for v in out["versions"]] == ["10.0", "9.1", "1.0"]
    assert out["versions"][0]["size"] == 10
    out = await call(mcp, "clinical_list_versions", {"path": "/S/u.sas"})
    assert out["versioned"] is False and out["versions"] == []


async def test_compare_repository_with_workspace(ca):
    mcp, fake = ca
    repo_path(fake, RUN)
    fake.add("POST", r"/repository/items/item-run/content$", b"a\nb\n")
    fake.add("POST", r"/workspaces/@currentUser/items/content$", b"a\nB\n")
    out = await call(mcp, "clinical_compare_files", {"left_path": RUN["path"]})
    assert out["identical"] is False
    assert "-b" in out["diff"] and "+B" in out["diff"]
    assert out["left"] == f"repository:{RUN['path']}" and out["right"] == f"workspace:{RUN['path']}"
    fake.add("POST", r"/workspaces/@currentUser/items/content$", b"a\nb\n")
    assert (await call(mcp, "clinical_compare_files", {"left_path": RUN["path"]}))["identical"] is True
    fake.add("POST", r"/workspaces/@currentUser/items/content$", b"\x00bin")
    assert (await call(mcp, "clinical_compare_files", {"left_path": RUN["path"]}))["status"] == "refused"


# --- search, audit, signatures ---------------------------------------------------------


async def test_search_repository_builds_one_filter(ca):
    mcp, fake = ca
    fake.add("GET", r"^/clinicalRepository/repository/items$", {"items": [RUN], "links": []})
    out = await call(
        mcp,
        "clinical_search_repository",
        {
            "name_contains": "O'Brien",
            "under_path": "/S/",
            "type_id": "sasprogram",
            "primary_type": "file",
            "checked_out": True,
        },
    )
    assert out["filter"] == (
        "and(contains(name,'O''Brien'),startsWith(path,'/S'),eq(typeId,'sasprogram'),"
        "eq(primaryType,'FILE'),eq(checkedOut,true))"
    )
    assert out["items"][0]["path"] == RUN["path"]
    assert (await call(mcp, "clinical_search_repository", {}))["status"] == "refused"
    out = await call(mcp, "clinical_search_repository", {"name_contains": "adsl"})
    assert out["filter"] == "contains(name,'adsl')"


async def test_search_audit_history(ca):
    mcp, fake = ca
    repo_path(fake, RUN)
    entry = {"id": "e1", "timestamp": "t", "userId": "michael", "action": "checkedIn", "sourceName": "run.sas"}
    fake.add("GET", r"/audit/entries$", {"items": [entry], "links": []})
    fake.add(
        "GET",
        r"/audit/entries/e1$",
        {"auditEntryDetails": [{"attributeName": "Version", "newValue": "3.1", "oldValue": None}]},
    )
    out = await call(
        mcp,
        "clinical_search_audit_history",
        {
            "path": RUN["path"],
            "action": "checkedIn",
            "user": "michael",
            "since": "2026-11-01",
            "until": "2026-11-13T10:00:00Z",
            "include_details": True,
        },
    )
    assert out["filter"] == (
        "and(eq(sourceId,'item-run'),eq(action,'checkedIn'),eq(userId,'michael'),"
        "ge(timestamp,'2026-11-01T00:00:00Z'),lt(timestamp,'2026-11-13T10:00:00Z'))"
    )
    assert out["entries"][0]["details"] == [{"attributeName": "Version", "newValue": "3.1"}]
    out = await call(mcp, "clinical_search_audit_history", {"name": "a.sas", "under_path": "/S"})
    assert out["filter"] == "and(eq(sourceName,'a.sas'),startsWith(sourceLocation,'/S'))"
    out = await call(mcp, "clinical_search_audit_history", {})
    assert out["filter"] is None


async def test_signature_history(ca):
    mcp, fake = ca
    repo_path(fake, {**RUN, "signingStatus": "SIGNED", "fileVersion": "3.1"})
    sig = {"version": 1, "signedBy": "qa", "reason": "Approved", "links": []}
    fake.add("GET", r"/repository/items/item-run/signatures$", {"items": [sig], "links": []})
    out = await call(mcp, "clinical_get_signature_history", {"path": RUN["path"]})
    assert out["signatures"] == [{"signedBy": "qa", "reason": "Approved"}]
    assert out["signing_status"] == "SIGNED"


# --- workspace actions ---------------------------------------------------------------


@pytest.fixture
def batch(ca):
    mcp, fake = ca
    fake.add(
        "POST", r"/items/batch$", {}, status=202, headers={"Location": "/clinicalActionStatus/actionstatus/act-1/"}
    )
    return mcp, fake


@pytest.mark.parametrize(
    ("args", "action", "body"),
    [
        ({"mode": "latest"}, "COPY_LATEST_VERSION", ["/S/a.sas"]),
        ({"mode": "folder_structure"}, "COPY_FOLDER_STRUCTURE", {"paths": ["/S/a.sas"], "includeSubfolders": True}),
        ({"mode": "version", "version": "1.2"}, "COPY_SPECIFIC_VERSION", [{"path": "/S/a.sas", "fileVersion": "1.2"}]),
    ],
)
async def test_copy_to_workspace(batch, args, action, body):
    mcp, fake = batch
    out = await call(mcp, "clinical_copy_to_workspace", {"paths": ["/S/a.sas"], **args})
    assert out["status"] == "completed"
    assert fake.batch(action) == body
    ((path, kw),) = fake.calls("POST", r"/batch$")
    assert path.startswith("/clinicalRepository/workspaces/@currentUser/") and kw["params"]["clientId"]


async def test_copy_refusals(batch):
    mcp, _ = batch
    assert (await call(mcp, "clinical_copy_to_workspace", {"paths": ["/a"], "mode": "version"}))["status"] == "refused"
    assert (await call(mcp, "clinical_copy_to_workspace", {"paths": ["/a"], "mode": "all"}))["status"] == "refused"
    assert (await call(mcp, "clinical_copy_to_workspace", {"paths": []}))["status"] == "refused"


async def test_batch_without_location_fails_clearly(ca):
    mcp, fake = ca
    fake.add("POST", r"/items/batch$", {}, status=202)
    out = await call(mcp, "clinical_check_out", {"paths": ["/S/a.sas"]})
    assert out["status"] == "failed" and "no action status" in out["message"]


async def test_check_out_and_undo(batch):
    mcp, fake = batch
    await call(mcp, "clinical_check_out", {"paths": '["/S/a.sas"]', "with_content": False})
    assert fake.batch("REPOSITORY_CHECK_OUT") == {"paths": ["/S/a.sas"], "copyContent": False}
    await call(mcp, "clinical_undo_check_out", {"paths": ["/S/a.sas"]})
    assert fake.batch("UNDO_CHECKOUT") == ["/S/a.sas"]


async def test_check_in_plans_and_reports(batch):
    mcp, fake = batch
    candidates = [
        {
            "path": "/S/a.sas",
            "synchronizationInfo": {"checkedOut": True, "checkedOutBy": "michael", "fileVersion": "1.4"},
        },
        {"path": "/S/b.sas", "synchronizationInfo": {"workspaceStatus": "OUT_OF_SYNC"}},
    ]
    fake.add("POST", r"/synchronization/items$", {"items": candidates})
    out = await call(mcp, "clinical_check_in", {"paths": ["/S"], "comment": "fix ADSL derivation"})
    assert out["status"] == "completed"
    assert fake.batch("CHECK_IN") == {
        "fileSpecifications": [{"path": "/S/a.sas", "type": "FILE", "fileVersion": "1.5"}],
        "comment": "fix ADSL derivation",
        "includeNewFiles": True,
    }
    assert out["skipped"][0]["reason"] == "not_checked_out" and "clinical_check_out" in out["skipped"][0]["explanation"]
    ((_, kw),) = fake.calls("POST", r"/synchronization/items$")
    assert kw["params"] == {"type": "CHECKIN"} and kw["json"] == ["/S"]


async def test_check_in_refusals(batch):
    mcp, fake = batch
    assert (await call(mcp, "clinical_check_in", {"paths": ["/S"], "comment": " "}))["status"] == "refused"
    assert (await call(mcp, "clinical_check_in", {"paths": ["/S"], "comment": "x", "bump": "patch"}))[
        "status"
    ] == "refused"
    fake.add("POST", r"/synchronization/items$", [])
    out = await call(mcp, "clinical_check_in", {"paths": ["/S/a.sas"], "comment": "x"})
    assert out["status"] == "nothing_to_check_in" and out["skipped"][0]["reason"] == "not_offered"


async def test_write_file_creates_then_updates(ca):
    mcp, fake = ca
    state = {"exists": False}
    fake.add(
        "GET",
        r"/workspaces/@currentUser/items/S/a.sas$",
        lambda p, kw: (200, {"name": "a.sas"}) if state["exists"] else (404, {}),
    )
    fake.add("PUT", r"/workspaces/@currentUser/items/S/a.sas$", {"name": "a.sas"})
    fake.add("GET", r"/workspaces/@currentUser/items/S$", {"name": "S", "type": "FOLDER"})
    out = await call(mcp, "clinical_write_file", {"path": "/S/a.sas", "content": "data x;"})
    assert out == {"status": "created", "path": "/S/a.sas", "size": 7}
    ((_, kw),) = fake.calls("PUT", r"/a.sas$")
    assert kw["params"] == {"expand": "false"}
    assert kw["files"]["file"][:2] == ("a.sas", b"data x;")
    state["exists"] = True
    assert (await call(mcp, "clinical_write_file", {"path": "/S/a.sas", "content": "y"}))["status"] == "updated"


async def test_write_file_refusals(ca):
    mcp, _ = ca
    assert (await call(mcp, "clinical_write_file", {"path": "/S/noext", "content": "x"}))["status"] == "refused"
    out = await call(mcp, "clinical_write_file", {"path": "/S/a.sas"})
    assert out["status"] == "invalid_source"


async def test_create_folder_in_both_locations(ca):
    mcp, fake = ca
    repo_path(fake, CONTEXT)
    fake.add("POST", r"/repository/items/item-s/children$", {"id": "new"})
    fake.add("POST", r"/workspaces/@currentUser/folders", {"name": "x"})
    out = await call(mcp, "clinical_create_folder", {"path": "/S/tlf", "location": "repository"})
    assert out["status"] == "created"
    ((_, kw),) = fake.calls("POST", r"/children$")
    assert kw["params"] == {"name": "tlf", "type": "FOLDER"}
    await call(mcp, "clinical_create_folder", {"path": "/top"})
    await call(mcp, "clinical_create_folder", {"path": "/S/sub"})
    paths = [p for p, _ in fake.calls("POST", r"/folders")]
    assert paths == [
        "/clinicalRepository/workspaces/@currentUser/folders",
        "/clinicalRepository/workspaces/@currentUser/folders/S",
    ]
    assert (await call(mcp, "clinical_create_folder", {"path": "/"}))["status"] == "refused"
    assert (await call(mcp, "clinical_create_folder", {"path": "/a:b"}))["status"] == "refused"
    fake.add("POST", r"/workspaces/@currentUser/folders", {}, status=404)
    assert (await call(mcp, "clinical_create_folder", {"path": "/no/x"}))["status"] == "not_found"


async def test_rename_repository_item_uses_a_fresh_etag(ca):
    mcp, fake = ca
    repo_path(fake, [RUN, CONTEXT])
    fake.add("GET", r"/repository/items/item-run$", RUN, headers={"ETag": '"e42"'})
    fake.add("PATCH", r"/repository/items/item-run$", RUN)
    out = await call(
        mcp, "clinical_rename_item", {"path": RUN["path"], "new_name": "run2.sas", "location": "repository"}
    )
    assert out["path"] == "/S/programs/run2.sas"
    ((_, kw),) = fake.calls("PATCH", r"item-run$")
    assert kw["json"] == {"name": "run2.sas"} and kw["headers"]["If-Match"] == '"e42"'
    out = await call(mcp, "clinical_rename_item", {"path": "/S", "new_name": "T", "location": "repository"})
    assert out["status"] == "refused"
    assert (await call(mcp, "clinical_rename_item", {"path": RUN["path"], "new_name": "a/b"}))["status"] == "refused"


async def test_rename_workspace_item_sends_a_json_string(ca):
    mcp, fake = ca
    fake.add("GET", r"/workspaces/@currentUser/items/S/a.sas$", {"name": "a.sas", "type": "FILE"})
    fake.add("POST", r"/workspaces/@currentUser/items/S/a.sas$", {"name": "b.sas"})
    await call(mcp, "clinical_rename_item", {"path": "/S/a.sas", "new_name": "b.sas"})
    ((_, kw),) = fake.calls("POST", r"/items/S/a.sas$")
    assert kw["params"] == {"action": "RENAME"} and kw["content"] == b'"b.sas"'
    fake.add("GET", r"/workspaces/@currentUser/items/S/x.sas$", {}, status=404)
    assert (await call(mcp, "clinical_rename_item", {"path": "/S/x.sas", "new_name": "y.sas"}))["status"] == "not_found"


async def test_delete_repository_items_to_recycle_bin(ca):
    mcp, fake = ca
    repo_path(fake, [RUN, CONTEXT, {**RUN, "id": "item-locked", "path": "/S/l.sas"}])
    fake.add("DELETE", r"/repository/items/item-run$", None, status=204)
    fake.add("DELETE", r"/repository/items/item-locked$", {"message": "The item is locked."}, status=409)
    out = await call(mcp, "clinical_delete_items", {"paths": [RUN["path"], "/S", "/S/l.sas"], "location": "repository"})
    assert out["status"] == "completed_with_errors"
    assert [i["status"] for i in out["items"]] == ["moved_to_recycle_bin", "refused", "failed"]
    assert "locked" in out["items"][2]["message"]
    ((_, kw),) = fake.calls("DELETE", r"item-run$")
    assert kw["params"] == {"permanent": "false"}


async def test_delete_workspace_items(batch):
    mcp, fake = batch
    out = await call(mcp, "clinical_delete_items", {"paths": ["/S/a.sas"]})
    assert out["status"] == "completed" and fake.batch("DELETE") == ["/S/a.sas"]


async def test_upload_to_repository(ca):
    mcp, fake = ca
    repo_path(fake, FOLDER)
    fake.add("PUT", r"/repository/items/item-prog/content$", {"id": "new"})
    out = await call(
        mcp,
        "clinical_upload",
        {
            "folder": "/S/programs",
            "name": "a.sas",
            "location": "repository",
            "content": "x",
            "version": "1.0",
            "comment": "first",
        },
    )
    assert out["status"] == "completed" and out["path"] == "/S/programs/a.sas"
    ((_, kw),) = fake.calls("PUT", r"/content$")
    assert kw["params"] == {"name": "a.sas", "expand": "false", "comment": "first", "fileVersion": "1.0"}


async def test_upload_to_workspace_announces_then_puts(batch):
    mcp, fake = batch
    fake.add("PUT", r"/workspaces/@currentUser/items", {})
    out = await call(mcp, "clinical_upload", {"folder": "/S", "name": "a.sas", "content": "x"})
    assert out["status"] == "completed" and out["path"] == "/S/a.sas"
    assert fake.batch("UPLOAD") == ["/S/a.sas"]
    ((path, kw),) = fake.calls("PUT", r"/items")
    assert path.endswith("/items/S/a.sas") and kw["params"] == {"expand": "false", "actionStatusId": "act-1"}
    await call(mcp, "clinical_upload", {"folder": "/", "name": "b.zip", "content": "PK", "expand": True})
    assert fake.batch("UPLOAD_EXPAND") == ["/b.zip"]
    path, kw = fake.calls("PUT", r"/items")[-1]
    assert path.endswith("/items") and kw["params"]["path"] == "/" and kw["params"]["expand"] == "true"


async def test_upload_refusals(ca):
    mcp, _ = ca
    out = await call(mcp, "clinical_upload", {"folder": "/S", "content": "x"})
    assert out["status"] == "refused"
    out = await call(mcp, "clinical_upload", {"folder": "/S", "name": "a.sas", "content": "x", "version": "1.0"})
    assert out["status"] == "refused" and "repository uploads" in out["message"]


@pytest.mark.parametrize(("enabled", "action"), [(True, "ENABLE_VERSIONING"), (False, "DISABLE_VERSIONING")])
async def test_set_versioning(batch, enabled, action):
    mcp, fake = batch
    repo_path(fake, {**RUN, "versioned": not enabled})
    fake.add("GET", r"/authorizations/item-run/permissions$", ["READ", "WRITE"])
    fake.add("GET", r"/authorizations/item-run/privileges$", ["PRIVILEGE_MANAGE_VERSIONING"])
    out = await call(mcp, "clinical_set_versioning", {"path": RUN["path"], "enabled": enabled, "comment": "why"})
    assert out["status"] == "completed" and out["versioned"] is enabled
    body = fake.batch(action)
    assert body["comment"] == "why"
    ((path, _),) = fake.calls("POST", r"/batch$")
    assert path == "/clinicalRepository/repository/items/batch"


async def test_set_versioning_refusals(batch):
    mcp, fake = batch
    repo_path(fake, [RUN, FOLDER])
    fake.add("GET", r"/authorizations/item-run/permissions$", ["READ"])
    fake.add("GET", r"/authorizations/item-run/privileges$", [])
    args = {"path": RUN["path"], "comment": "why"}
    assert (await call(mcp, "clinical_set_versioning", {**args, "enabled": True}))["status"] == "unchanged"
    out = await call(mcp, "clinical_set_versioning", {**args, "enabled": False})
    assert out["status"] == "refused" and "privilege" in out["message"]
    assert (await call(mcp, "clinical_set_versioning", {**args, "enabled": False, "comment": ""}))[
        "status"
    ] == "refused"
    out = await call(mcp, "clinical_set_versioning", {"path": "/S/programs", "enabled": True, "comment": "x"})
    assert out["status"] == "refused"


# --- datasets into SAS ---------------------------------------------------------------


@pytest.fixture
def sas(monkeypatch):
    submitted: list[str] = []

    async def submit(client, session_id, code):
        submitted.append(code)
        return f"job-{len(submitted)}"

    log = {"text": "NOTE: ok\n_CA_MEMBERS=AE SUPPAE"}

    async def wait(client, session_id, job_id, poll=2):
        return "completed", log["text"], ""

    monkeypatch.setattr(clinical_repository, "submit_job", submit)
    monkeypatch.setattr(clinical_repository, "wait_job", wait)
    return submitted, log


async def test_load_transport_file(ca, sas):
    mcp, fake = ca
    submitted, _ = sas
    repo_path(fake, {**RUN, "id": "item-ae", "path": "/S/ae.xpt", "name": "ae.xpt"})
    fake.add(
        "POST", r"/repository/items/item-ae/content$", b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!" + b"0" * 40
    )
    out = await call(mcp, "clinical_load_dataset", {"path": "/S/ae.xpt"})
    assert out["status"] == "loaded" and out["libref"] == "_CAAE"
    assert out["datasets"] == ["_CAAE.AE", "_CAAE.SUPPAE"]
    assert "libname _XPTIN xport" in submitted[1]
    assert "dictionary.tables where libname='_CAAE'" in submitted[2]


async def test_load_sas7bdat_and_errors(ca, sas):
    mcp, fake = ca
    submitted, log = sas
    fake.add("POST", r"/workspaces/@currentUser/items/content$", b"\x00sas7bdat bytes")
    out = await call(
        mcp, "clinical_load_dataset", {"path": "/S/ADSL.sas7bdat", "location": "workspace", "libref": "adam"}
    )
    assert out["libref"] == "ADAM"
    assert '"%sysfunc(pathname(work))/adam/adsl.sas7bdat"' in submitted[0]
    assert submitted[1] == 'libname adam "%sysfunc(pathname(work))/adam";'
    log["text"] = "ERROR: Library ADAM does not exist."
    out = await call(mcp, "clinical_load_dataset", {"path": "/S/adsl.sas7bdat", "location": "workspace"})
    assert out["status"] == "failed" and out["errors"] == ["ERROR: Library ADAM does not exist."]


async def test_load_dataset_refusals(ca, sas):
    mcp, fake = ca
    assert (await call(mcp, "clinical_load_dataset", {"path": "/S/a.sas"}))["status"] == "refused"
    assert (await call(mcp, "clinical_load_dataset", {"path": "/S/a.xpt", "libref": "toolongname"}))[
        "status"
    ] == "refused"
    fake.add("POST", r"/workspaces/@currentUser/items/content$", b"not a transport file")
    out = await call(mcp, "clinical_load_dataset", {"path": "/S/a.xpt", "location": "workspace"})
    assert out["status"] == "refused" and "transport" in out["message"]


# --- Tier 13: jobs --------------------------------------------------------------------

JOB_FILE = {
    "description": "Derive ADSL",
    "programs": [{"path": "./adsl.sas", "version": None}],
    "parameters": [
        {"name": "study", "type": "CHARACTER", "defaultValue": "S1"},
        {"name": "cutoff", "type": "DATE", "defaultValue": "2026-01-01T00:00:00Z"},
    ],
    "logsLocation": "./",
}
SUBMITTED = {"id": "run-1", "state": "STARTED", "jobRunContext": "INTERACTIVE"}
DONE = {
    "id": "run-1",
    "name": "adsl.cjob",
    "jobPath": "/S/adsl.cjob",
    "jobResourceLocation": "WORKSPACE",
    "state": "COMPLETED_SUCCESSFUL",
    "jobRunContext": "INTERACTIVE",
    "submittedBy": "michael",
}


@pytest.fixture
def jobs(ca):
    mcp, fake = ca
    fake.add("POST", r"/workspaces/@currentUser/items/content$", json.dumps(JOB_FILE).encode())
    fake.add("POST", r"^/clinicalJobs/submissions$", SUBMITTED)
    fake.add("GET", r"/clinicalJobs/submissions/run-1/state$", "COMPLETED_SUCCESSFUL")
    fake.add("GET", r"/clinicalJobs/submissions/run-1$", DONE)
    return mcp, fake


async def test_get_workspace_and_repository_job(jobs):
    mcp, fake = jobs
    out = await call(mcp, "clinical_get_job", {"path": "/S/adsl.cjob"})
    assert out["job"]["description"] == "Derive ADSL" and out["problems"] == []
    repo_path(fake, {**RUN, "id": "item-job", "path": "/S/adsl.cjob"})
    fake.add("GET", r"/clinicalJobs/repository/definitions/item-job$", {**JOB_FILE, "links": [1]})
    out = await call(mcp, "clinical_get_job", {"path": "/S/adsl.cjob", "location": "repository", "version": "1.0"})
    assert "links" not in out["job"]
    ((_, kw),) = fake.calls("GET", r"/repository/definitions/")
    assert kw["params"] == {"fileVersion": "1.0"}
    assert (await call(mcp, "clinical_get_job", {"path": "/S/adsl.sas"}))["status"] == "refused"
    out = await call(mcp, "clinical_get_job", {"path": "/S/adsl.cjob", "version": "1.0"})
    assert out["status"] == "refused"


async def test_run_job_sends_only_what_the_service_accepts(jobs):
    mcp, fake = jobs
    out = await call(
        mcp, "clinical_run_job", {"job_path": "/S/adsl.cjob", "parameter_values": {"cutoff": "2026-11-13"}}
    )
    assert out["category"] == "successful" and out["run_id"] == "run-1"
    ((_, kw),) = fake.calls("POST", r"^/clinicalJobs/submissions$")
    assert kw["params"] == {"action": "RUN"}
    assert kw["json"] == {
        "job": {"name": "adsl.cjob", "path": "/S/adsl.cjob", "location": "/S", "resourceLocation": "WORKSPACE"},
        "parameterValues": [
            {"name": "study", "type": "CHARACTER", "value": "S1"},
            {"name": "cutoff", "type": "DATE", "value": "2026-11-13T00:00:00Z"},
        ],
    }


async def test_run_and_populate_puts_relative_paths_in_the_body(jobs):
    mcp, fake = jobs
    await call(mcp, "clinical_run_job", {"job_path": "/S/adsl.cjob", "mode": "run_and_populate"})
    ((_, kw),) = fake.calls("POST", r"^/clinicalJobs/submissions$")
    assert kw["params"] == {"action": "RUN_AND_POPULATE"} and kw["json"]["useRelativePaths"] is True


async def test_run_job_refusals(jobs):
    mcp, _ = jobs
    assert (await call(mcp, "clinical_run_job", {}))["status"] == "refused"
    out = await call(mcp, "clinical_run_job", {"job_path": "/S/a.cjob", "location": "repository", "mode": "test"})
    assert out["status"] == "refused" and "workspace jobs" in out["message"]
    assert (await call(mcp, "clinical_run_job", {"job_path": "/S/a.cjob", "mode": "fast"}))["status"] == "refused"
    out = await call(mcp, "clinical_run_job", {"program_path": "/S/a.sas", "mode": "test"})
    assert out["status"] == "refused"


async def test_run_job_returns_while_still_running(jobs):
    mcp, fake = jobs
    fake.add("GET", r"/clinicalJobs/submissions/run-1/state$", "RUNNING_TASKS")
    fake.add("GET", r"/clinicalJobs/submissions/run-1$", {**DONE, "state": "RUNNING_TASKS"})
    out = await call(mcp, "clinical_run_job", {"job_path": "/S/adsl.cjob", "wait_seconds": 5})
    assert out["category"] == "running" and "clinical_get_job_run" in out["message"]


async def test_run_program_creates_its_job(jobs):
    mcp, fake = jobs
    fake.add("GET", r"/workspaces/@currentUser/items/S/adsl.sas$", {"name": "adsl.sas"})
    created = {"done": False}

    def content(path, kw):
        if kw["json"]["paths"] == ["/S/adsl.cjob"] and not created["done"]:
            return 404, {"message": "no"}
        return 200, {**JOB_FILE, "parameters": []}

    def create(path, kw):
        created["done"] = True
        return 201, {}

    fake.add("POST", r"/workspaces/@currentUser/items/content$", content)
    fake.add("POST", r"/clinicalJobs/definitions$", create)
    out = await call(mcp, "clinical_run_job", {"program_path": "/S/adsl.sas"})
    assert out["created_job"] == "/S/adsl.cjob"
    ((_, kw),) = fake.calls("POST", r"/definitions$")
    assert kw["params"] == {"useRelativePaths": "true"}
    assert kw["json"]["programs"] == [{"path": "/S/adsl.sas", "name": "adsl.sas", "location": "/S"}]


async def test_run_program_reuses_or_skips_jobs(jobs):
    mcp, fake = jobs
    fake.add("GET", r"/workspaces/@currentUser/items/S/adsl.sas$", {"name": "adsl.sas"})

    def content(path, kw):
        if kw["json"]["paths"] == ["/S/adsl.cjob"]:
            return 200, {"programs": [{"path": "./other.sas"}, {"path": "./adsl.sas"}]}
        return 200, {"programs": [{"path": "./adsl.sas"}], "parameters": []}

    fake.add("POST", r"/workspaces/@currentUser/items/content$", content)
    out = await call(mcp, "clinical_run_job", {"program_path": "/S/adsl.sas"})
    assert "created_job" not in out
    ((_, kw),) = fake.calls("POST", r"^/clinicalJobs/submissions$")
    assert kw["json"]["job"]["path"] == "/S/adsl-2.cjob"
    fake.add("GET", r"/workspaces/@currentUser/items/S/gone.sas$", {}, status=404)
    assert (await call(mcp, "clinical_run_job", {"program_path": "/S/gone.sas"}))["status"] == "not_found"
    assert (await call(mcp, "clinical_run_job", {"program_path": "/S/a.txt"}))["status"] == "refused"


async def test_rerun_repeats_mode_and_values(jobs):
    mcp, fake = jobs
    fake.add(
        "GET",
        r"/clinicalJobs/submissions/run-0$",
        {**DONE, "id": "run-0", "jobRunContext": "INTERACTIVE_RUN_AND_POPULATE_RELATIVE"},
    )
    fake.add(
        "GET",
        r"/clinicalJobs/submissions/run-0/manifest$",
        {"parameters": [{"name": "study", "type": "CHARACTER", "value": "S9", "label": "x"}]},
    )
    await call(mcp, "clinical_run_job", {"rerun_of": "run-0"})
    ((_, kw),) = fake.calls("POST", r"^/clinicalJobs/submissions$")
    assert kw["params"] == {"action": "RUN_AND_POPULATE"}
    assert kw["json"]["parameterValues"] == [{"name": "study", "type": "CHARACTER", "value": "S9"}]
    assert kw["json"]["useRelativePaths"] is True


async def test_save_job_creates_then_writes_the_file(jobs):
    mcp, fake = jobs
    fake.add("GET", r"/workspaces/@currentUser/items/S/new.cjob$", {}, status=404)
    fake.add("GET", r"/workspaces/@currentUser/items/S$", {"name": "S", "type": "FOLDER"})
    fake.add("POST", r"/clinicalJobs/definitions$", {}, status=201)
    created = {"programs": [], "parameters": [], "logsLocation": "./", "manifestLocation": "./"}
    fake.add("POST", r"/workspaces/@currentUser/items/content$", json.dumps(created).encode())
    fake.add("PUT", r"/workspaces/@currentUser/items/S/new.cjob$", {})
    job = {
        "programs": [{"path": "/S/progs/adsl.sas"}],
        "parameters": [{"name": "study", "type": "CHARACTER"}],
        "resultsLocation": "/S/out",
    }
    out = await call(mcp, "clinical_save_job", {"path": "/S/new.cjob", "job": job})
    assert out == {"status": "created", "path": "/S/new.cjob"}
    ((_, kw),) = fake.calls("POST", r"/definitions$")
    assert kw["json"]["resource"]["path"] == "/S/new.cjob" and kw["json"]["programs"] == []
    ((_, kw),) = fake.calls("PUT", r"/new.cjob$")
    saved = json.loads(kw["files"]["file"][1])
    assert saved["programs"] == [{"path": "./progs/adsl.sas", "version": None}]
    assert saved["resultsLocation"] == "./out" and saved["manifestLocation"] == "./"


async def test_save_job_refusals(jobs):
    mcp, fake = jobs
    out = await call(mcp, "clinical_save_job", {"path": "/S/a.cjob", "job": {"programs": []}})
    assert out["status"] == "invalid_job"
    assert (await call(mcp, "clinical_save_job", {"path": "/S/a.json", "job": JOB_FILE}))["status"] == "refused"
    fake.add("GET", r"/workspaces/@currentUser/items/S/a.cjob$", {"name": "a.cjob"})
    fake.add("PUT", r"/workspaces/@currentUser/items/S/a.cjob$", {})
    out = await call(
        mcp, "clinical_save_job", {"path": "/S/a.cjob", "job": {**JOB_FILE, "programs": [{"path": "/T/x.sas"}]}}
    )
    assert out["status"] == "updated"
    ((_, kw),) = fake.calls("PUT", r"/a.cjob$")
    assert json.loads(kw["files"]["file"][1])["programs"][0]["path"] == "../T/x.sas"


async def test_get_and_list_and_cancel_runs(jobs):
    mcp, fake = jobs
    out = await call(mcp, "clinical_get_job_run", {"run_id": "run-1"})
    assert out["state"] == "COMPLETED_SUCCESSFUL"
    fake.add("GET", r"^/clinicalJobs/submissions$", {"items": [DONE], "links": [{"rel": "next"}]})
    out = await call(mcp, "clinical_list_job_runs", {"job_path": "/S/adsl.cjob"})
    assert out["has_more"] is True and out["runs"][0]["run_id"] == "run-1"
    _, kw = fake.calls("GET", r"^/clinicalJobs/submissions$")[-1]
    assert kw["params"]["filter"] == "and(eq(submittedBy,'michael'),eq(jobPath,'/S/adsl.cjob'))"
    await call(mcp, "clinical_list_job_runs", {"all_users": True})
    _, kw = fake.calls("GET", r"^/clinicalJobs/submissions$")[-1]
    assert "filter" not in kw["params"]
    fake.add("PUT", r"/submissions/run-1/state$", "CANCELING")
    assert (await call(mcp, "clinical_cancel_job_run", {"run_id": "run-1"}))["status"] == "canceling"
    ((_, kw),) = fake.calls("PUT", r"/state$")
    assert kw["params"] == {"value": "canceled"}
    fake.add(
        "PUT", r"/submissions/run-2/state$", {"message": "Completed job submissions cannot be canceled."}, status=409
    )
    out = await call(mcp, "clinical_cancel_job_run", {"run_id": "run-2"})
    assert out == {
        "status": "not_cancelable",
        "run_id": "run-2",
        "message": "Completed job submissions cannot be canceled.",
    }
    fake.add("PUT", r"/submissions/nope/state$", {}, status=404)
    assert (await call(mcp, "clinical_cancel_job_run", {"run_id": "nope"}))["status"] == "not_found"
    fake.add("GET", r"/clinicalJobs/submissions/nope$", {}, status=404)
    fake.add("GET", r"/clinicalJobs/submissions/nope/state$", "COMPLETED_FAILED")
    assert (await call(mcp, "clinical_get_job_run", {"run_id": "nope"}))["status"] == "not_found"


MANIFEST = {
    "type": "WORKSPACE",
    "jobSubmission": {"id": "run-1", "state": "COMPLETED_SUCCESSFUL", "jobRunContext": "INTERACTIVE"},
    "programs": [{"resource": {"path": "/S/adsl.sas"}, "executionState": "SUCCESSFUL", "executionTime": 41}],
    "inputs": [{"resource": {"path": "/S/dm.sas7bdat", "resourceLocation": "REPOSITORY", "fileVersion": "1.0"}}],
    "logs": [{"resource": {"path": "/S/adsl.log", "resourceLocation": "WORKSPACE"}}],
    "results": [{"resource": {"path": "/S/adsl.lst", "resourceLocation": "WORKSPACE"}}],
    "manifest": {"resource": {"path": "/S/adsl.cmnf", "resourceLocation": "WORKSPACE"}},
    "checkinSpecification": {"locations": []},
}


async def test_manifest_fills_checkin_locations_from_the_cmnf(jobs):
    mcp, fake = jobs
    fake.add("GET", r"/submissions/run-1/manifest$", MANIFEST)
    fake.add(
        "GET",
        r"/clinicalJobs/submissions/run-1$",
        {**DONE, "checkinFailures": [{"itemIdentifier": "/S/x", "message": "m"}]},
    )
    cmnf = {"jobSubmission": {"id": "run-1"}, "checkinSpecification": {"locations": ["/S/tlf"]}}
    fake.add("POST", r"/workspaces/@currentUser/items/content$", json.dumps(cmnf).encode())
    out = await call(mcp, "clinical_get_job_manifest", {"run_id": "run-1"})
    assert out["checkin"]["locations"] == [{"path": "/S/tlf"}]
    assert out["inputs"][0]["version"] == "1.0"
    assert out["checkin_failures"] == [{"path": "/S/x", "message": "m"}]
    out = await call(mcp, "clinical_get_job_manifest", {"cmnf_path": "/S/adsl.cmnf"})
    assert out["run_id"] == "run-1"


async def test_manifest_from_repository_cmnf_and_overwritten(jobs):
    mcp, fake = jobs
    repo_path(fake, {**RUN, "id": "item-cmnf", "path": "/S/adsl.cmnf", "name": "adsl.cmnf"})
    fake.add("POST", r"/repository/items/item-cmnf/content$", json.dumps({"jobSubmission": {"id": "run-1"}}).encode())
    fake.add("GET", r"/submissions/run-1/manifest$", {"message": "gone"}, status=404)
    out = await call(mcp, "clinical_get_job_manifest", {"cmnf_path": "/S/adsl.cmnf", "location": "repository"})
    assert out["status"] == "manifest_overwritten"
    assert (await call(mcp, "clinical_get_job_manifest", {}))["status"] == "refused"
    fake.add("POST", r"/repository/items/item-cmnf/content$", b"{}")
    out = await call(mcp, "clinical_get_job_manifest", {"cmnf_path": "/S/adsl.cmnf", "location": "repository"})
    assert out["status"] == "refused" and "names no run" in out["message"]


async def test_job_output_reads_logs_and_flags_errors(jobs):
    mcp, fake = jobs
    fake.add("GET", r"/submissions/run-1/manifest$", MANIFEST)
    fake.add("POST", r"/workspaces/@currentUser/items/content$", b"NOTE: x\nERROR: Variable USUBJID not found.\n")
    out = await call(mcp, "clinical_get_job_output", {"run_id": "run-1"})
    assert out["files"][0]["path"] == "/S/adsl.log" and out["errors"] == ["ERROR: Variable USUBJID not found."]
    out = await call(mcp, "clinical_get_job_output", {"run_id": "run-1", "kind": "results"})
    assert out["files"][0]["path"] == "/S/adsl.lst" and "errors" not in out
    assert (await call(mcp, "clinical_get_job_output", {"run_id": "run-1", "kind": "pdf"}))["status"] == "refused"
    fake.add("GET", r"/submissions/run-1/manifest$", {**MANIFEST, "logs": []})
    assert (await call(mcp, "clinical_get_job_output", {"run_id": "run-1"}))["status"] == "not_found"


async def test_job_output_reads_repository_files_at_their_version(jobs, monkeypatch):
    mcp, fake = jobs
    monkeypatch.setattr(clinical_jobs, "_MAX_OUTPUT_CHARS", 5)
    log = {"resource": {"path": "/S/a.log", "resourceLocation": "REPOSITORY", "id": "item-log", "fileVersion": "2.0"}}
    fake.add("GET", r"/submissions/run-1/manifest$", {**MANIFEST, "logs": [log, log]})
    fake.add("GET", r"/repository/items/item-log/versions/2.0/content$", b"0123456789")
    out = await call(mcp, "clinical_get_job_output", {"run_id": "run-1"})
    assert out["files"] == [{"path": "/S/a.log", "content": "01234", "truncated": True}]


# --- schedules ---------------------------------------------------------------------

SCHEDULE = {
    "id": "s1",
    "name": "nightly",
    "status": "ENABLED",
    "createdBy": "michael",
    "jobInfo": {
        "job": {"path": "/S/adsl.cjob", "resourceLocation": "WORKSPACE"},
        "parameterValues": [{"name": "study", "type": "CHARACTER", "value": "S1", "label": "null"}],
    },
    "trigger": {"type": "DAILY", "hours": "2", "minutes": "30", "timeZone": "UTC"},
}


async def test_list_and_get_schedules(jobs):
    mcp, fake = jobs
    fake.add("GET", r"^/clinicalJobs/schedules$", {"items": [SCHEDULE, {**SCHEDULE, "id": "s2", "createdBy": "bob"}]})
    out = await call(mcp, "clinical_list_schedules", {})
    assert [s["schedule_id"] for s in out["schedules"]] == ["s1"]
    assert len((await call(mcp, "clinical_list_schedules", {"all_users": True}))["schedules"]) == 2
    fake.add("GET", r"/clinicalJobs/schedules/s1$", SCHEDULE)
    fake.add("GET", r"/clinicalJobs/schedules/s1/future$", ["2026-11-14T02:30:00.000Z"])
    out = await call(mcp, "clinical_get_schedule", {"schedule_id": "s1"})
    assert out["next_runs"] == ["2026-11-14T02:30:00.000Z"]
    fake.add("GET", r"/clinicalJobs/schedules/s1/future$", {"items": ["2026-11-15T02:30:00.000Z"]})
    assert (await call(mcp, "clinical_get_schedule", {"schedule_id": "s1"}))["next_runs"] == [
        "2026-11-15T02:30:00.000Z"
    ]
    fake.add("GET", r"/clinicalJobs/schedules/s9$", {}, status=404)
    assert (await call(mcp, "clinical_get_schedule", {"schedule_id": "s9"}))["status"] == "not_found"


async def test_save_schedule_create_and_replace(jobs):
    mcp, fake = jobs
    fake.add("POST", r"^/clinicalJobs/schedules$", SCHEDULE)
    fake.add("PATCH", r"/clinicalJobs/schedules/s1$", SCHEDULE)
    trigger = {"type": "DAILY", "hours": 2, "minutes": 30, "timeZone": "UTC"}
    args = {"name": "nightly", "job_path": "/S/adsl.cjob", "location": "workspace", "trigger": trigger}
    out = await call(mcp, "clinical_save_schedule", {**args, "description": "d"})
    assert out["status"] == "created" and out["schedule"]["yours"] is True
    ((_, kw),) = fake.calls("POST", r"/schedules$")
    assert kw["json"]["trigger"] == {"type": "DAILY", "timeZone": "UTC", "interval": 1, "hours": "2", "minutes": "30"}
    assert kw["json"]["jobInfo"]["job"]["resourceLocation"] == "WORKSPACE"
    assert kw["json"]["description"] == "d"
    out = await call(mcp, "clinical_save_schedule", {**args, "schedule_id": "s1"})
    assert out["status"] == "updated"
    fake.add("PATCH", r"/clinicalJobs/schedules/s9$", {}, status=404)
    assert (await call(mcp, "clinical_save_schedule", {**args, "schedule_id": "s9"}))["status"] == "not_found"
    out = await call(mcp, "clinical_save_schedule", {**args, "trigger": {"type": "DAILY"}})
    assert out["status"] == "refused"


async def test_schedule_state_delete_and_run_now(jobs):
    mcp, fake = jobs
    fake.add("PUT", r"/schedules/s1/state$", {})
    assert (await call(mcp, "clinical_set_schedule_state", {"schedule_id": "s1", "enabled": False}))[
        "status"
    ] == "disabled"
    ((_, kw),) = fake.calls("PUT", r"/schedules/s1/state$")
    assert kw["params"] == {"value": "DISABLED"}
    fake.add("PUT", r"/schedules/s9/state$", {}, status=404)
    assert (await call(mcp, "clinical_set_schedule_state", {"schedule_id": "s9", "enabled": True}))[
        "status"
    ] == "not_found"
    fake.add("DELETE", r"/schedules/s1$", None, status=204)
    assert (await call(mcp, "clinical_delete_schedule", {"schedule_id": "s1"}))["status"] == "deleted"
    fake.add("DELETE", r"/schedules/s9$", {}, status=404)
    assert (await call(mcp, "clinical_delete_schedule", {"schedule_id": "s9"}))["status"] == "not_found"
    fake.add("GET", r"/clinicalJobs/schedules/s1$", SCHEDULE)
    out = await call(mcp, "clinical_run_schedule_now", {"schedule_id": "s1"})
    assert out["run_id"] == "run-1"
    ((_, kw),) = fake.calls("POST", r"^/clinicalJobs/submissions$")
    assert kw["json"]["parameterValues"] == [{"name": "study", "type": "CHARACTER", "value": "S1"}]
    fake.add("GET", r"/clinicalJobs/schedules/s2$", {**SCHEDULE, "jobInfo": {}})
    assert (await call(mcp, "clinical_run_schedule_now", {"schedule_id": "s2"}))["status"] == "refused"


# --- behaviour found against a live server ---------------------------------------------


async def test_write_file_creates_missing_workspace_folders(ca):
    mcp, fake = ca
    existing = {"/S"}
    fake.add(
        "GET",
        r"/workspaces/@currentUser/items/",
        lambda p, kw: (
            (200, {"name": "x"})
            if p.removeprefix("/clinicalRepository/workspaces/@currentUser/items") in existing
            else (404, {})
        ),
    )
    fake.add("POST", r"/workspaces/@currentUser/folders", {})
    fake.add("PUT", r"/workspaces/@currentUser/items/S/a/b/c.sas$", {})
    out = await call(mcp, "clinical_write_file", {"path": "/S/a/b/c.sas", "content": "x"})
    assert out["created_folders"] == ["/S/a", "/S/a/b"]
    assert [(p, kw["params"]) for p, kw in fake.calls("POST", r"/folders")] == [
        ("/clinicalRepository/workspaces/@currentUser/folders/S", {"name": "a"}),
        ("/clinicalRepository/workspaces/@currentUser/folders/S/a", {"name": "b"}),
    ]


async def test_versioned_upload_without_privilege_explains(ca):
    mcp, fake = ca
    repo_path(fake, FOLDER)
    fake.add("PUT", r"/repository/items/item-prog/content$", {"errorCode": 145000}, status=500)
    out = await call(
        mcp,
        "clinical_upload",
        {"folder": "/S/programs", "name": "a.sas", "location": "repository", "content": "x", "version": "1.0"},
    )
    assert out["status"] == "failed" and "versioning privilege" in out["message"]


async def test_check_in_without_privilege_gets_a_hint(batch):
    mcp, fake = batch
    fake.add("POST", r"/synchronization/items$", {"items": [{"path": "/S/n.sas", "synchronizationInfo": {}}]})
    failed = {
        "summary": {"endTimeStamp": "t", "completionStatus": "ERROR", "message": "0 of 1"},
        "details": [
            {"itemIdentifier": "/S/n.sas", "message": 'Cannot invoke "Privilege.getId()" because "priv" is null'}
        ],
    }
    fake.add("GET", r"/actionstatus/", failed)
    out = await call(mcp, "clinical_check_in", {"paths": ["/S/n.sas"], "comment": "x", "new_file_version": "1.0"})
    assert out["status"] == "failed" and "versioning privilege" in out["hint"]
