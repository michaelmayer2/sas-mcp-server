# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAS Clinical Acceleration plumbing shared by Tier 12 (repository) and Tier 13 (jobs).

Behaviour this relies on, probed against SAS Viya with Clinical Acceleration
(clinicalRepository 1.14.8, clinicalJobs and clinicalActionStatus 1.10.7) on
2026-11-13, and matching the Positron SAS extension (``client/src/clinical``),
which verified the same calls against a SAS Viya 2025 server. The public API
reference differs from the live services in several places; where it does, the
live behaviour wins and is noted next to the code.

Services and addressing
  * ``/clinicalRepository/`` answers the API root (the trailing slash matters:
    without it Viya answers 302). ``/clinicalJobs/`` answers 404, so Clinical
    Job Management is detected with ``GET /clinicalJobs/submissions?limit=1``.
  * Repository URLs take item ids; ``GET /repository/paths?path=<p>`` turns any
    path (root, folder, file) into the item, and answers 404 with errorCode
    145016 for a path that does not exist. Workspace URLs take paths. The same
    path names a file in the repository and its copy in the workspace.
  * Content is read with POST, not GET: ``POST items/{id}/content`` with the
    JSON body ``[""]`` (repository) and ``POST workspaces/@currentUser/items/
    content`` with ``{"paths": [...]}`` (workspace). A version's content is
    ``GET items/{id}/versions/{v}/content``. The extension also met single
    files wrapped in a one-entry zip, so such a payload is unwrapped.

Batch actions
  Check-out, copy, check-in, workspace delete and versioning changes are batch
  actions: ``POST …/items/batch?action=X&clientId=<uuid>`` answers 202 with a
  ``Location`` of ``/clinicalActionStatus/actionstatus/{id}``. The outcome is
  only in that status (``summary.endTimeStamp`` set when done,
  ``summary.completionStatus`` INFO/WARN/ERROR, per-item ``details``); a 2xx
  on the POST does not mean the action worked. The request body differs per
  action — a bare array, ``{paths}`` or ``{fileSpecifications}`` — see
  :func:`batch_body`.

Permissions
  ``/authorizations/{id}/permissions`` answers e.g. ``["READ","WRITE"]``.
  ``/authorizations/{id}/privileges`` answers 404 (errorCode 145191) when the
  user holds none, and on this release HTTP 500 for files and folders; both
  are reported as "unknown", and the server stays the judge of every action.

Search, audit history and signatures (not used by the extension)
  * ``GET /repository/items?filter=<RQL>`` searches the whole repository the
    user can see (``contains``/``eq``/``startsWith`` on name, path, typeId,
    primaryType, checkedOut, …).
  * ``GET /audit/entries?filter=`` filters on action, userId, mode, timestamp,
    sourceName, sourceLocation, sourceId and sourceTypeId (``name`` and
    ``location`` are refused with errorCode 1020). Newest first. A normal user
    sees only their own entries. ``/audit/entries/{id}`` adds the details,
    ``/audit/actions`` lists the action names.
  * ``GET /repository/items/{id}/signatures`` lists a file's signatures; a
    version's details carry ``signed``.

Clinical jobs
  * Run: ``POST /clinicalJobs/submissions?action=RUN|TEST|RUN_AND_POPULATE``
    with only ``{job: {name, path, location, resourceLocation},
    parameterValues}`` — the service rejects any other field (the spec's
    ``version``, a job ``id``). ``useRelativePaths`` goes in the body; as a
    query parameter it is ignored.
  * Submissions come back with flat fields (``jobPath``, ``jobResourceLocation``
    …), not the spec's nested ``job``. The list holds every user's runs, so
    "mine" is ``filter=eq(submittedBy,"<user>")``.
  * State: ``GET …/state`` (Accept ``application/json``) answers a bare string.
    A run is over when its state starts with ``COMPLETED``. Cancel is
    ``PUT …/state?value=canceled`` (lowercase).
  * Manifest: ``GET …/{id}/manifest`` answers 404 once the run's ``.cmnf`` was
    overwritten by a later run. It can come back without the check-in
    locations the ``.cmnf`` file still lists.
  * Workspace jobs have no GET; their ``.cjob`` is read as file content.
    Create is ``POST /definitions?useRelativePaths=true``; update is ``PATCH
    /definitions?useRelativePaths=true`` with the whole job, ``resource``
    included, and no id in the URL.
  * Schedules: ``GET /schedules?limit=1000`` returns every user's schedules
    (no paging). State is ``PUT …/state?value=ENABLED|DISABLED`` (uppercase).
    ``…/future`` answers a bare array of times or a collection.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import posixpath
import re
import time
import uuid
import zipfile
from collections.abc import Iterable
from typing import Any, NoReturn
from urllib.parse import quote

import httpx

from .. import config
from ..viya_client import raise_for_viya_status

REPO = "/clinicalRepository"
REPO_ITEMS = f"{REPO}/repository/items"
WORKSPACE = f"{REPO}/workspaces/@currentUser"
JOBS = "/clinicalJobs"
ACTION_STATUS = "/clinicalActionStatus/actionstatus"

ACCEPT_API = "application/vnd.sas.api+json"
ACCEPT_COLLECTION = "application/vnd.sas.collection+json"
# Creating, renaming and uploading repository items answer 406 to their own
# item media type on this release; they accept application/json.
ACCEPT_ITEM = "application/json"
ACCEPT_ACTION_STATUS = "application/vnd.sas.clinical.action.status+json"
JSON = "application/json"

REPOSITORY = "repository"
WORKSPACE_LOCATION = "workspace"
LOCATIONS = (REPOSITORY, WORKSPACE_LOCATION)

# One id per server process: the action-status service groups actions by client.
_CLIENT_ID = str(uuid.uuid4())
# The clock and sleep the pollers use; module attributes so tests can replace them.
_clock = time.monotonic
_sleep = asyncio.sleep
_POLL_INTERVAL = 2.0
MAX_PAGES = 100


class ClinicalError(ValueError):
    """A request the clinical tools cannot carry out. ``status`` is the tool result's status."""

    def __init__(self, message: str, status: str = "refused") -> None:
        super().__init__(message)
        self.status = status

    def result(self, **extra: Any) -> dict[str, Any]:
        return {"status": self.status, "message": str(self), **extra}


def fail(message: str, status: str = "refused") -> NoReturn:
    """Raise :class:`ClinicalError`; tools call this inside the ``try`` that turns it into a result."""
    raise ClinicalError(message, status)


# --- paths ------------------------------------------------------------------------------


def normalize_path(path: str | None) -> str:
    """``path`` as an absolute repository or workspace path: ``/a/b``, ``/`` for the root.

    Refuses ``.`` and ``..`` segments: a path names an item, it does not navigate.
    """
    raw = (path or "/").strip().replace("\\", "/")
    parts = [p for p in raw.split("/") if p]
    if any(p in (".", "..") for p in parts):
        raise ClinicalError(f"Path {path!r} contains '.' or '..'; pass the absolute path of the item.")
    return "/" + "/".join(parts)


def split_path(path: str) -> tuple[str, str]:
    """``(parent, name)`` of a normalized path."""
    parent, name = posixpath.split(path)
    return parent or "/", name


def url_path(path: str) -> str:
    """A normalized path for a URL, slashes kept and everything else encoded."""
    return quote(path, safe="/")


def check_location(location: str) -> str:
    value = (location or "").strip().lower()
    if value not in LOCATIONS:
        raise ClinicalError(f"location must be 'repository' or 'workspace', not {location!r}.")
    return value


# --- names, versions, comments (ported from the extension's utils.ts) ------------------

_FILE_NAME = re.compile(r"^([^/<>;\\{}?#]+)\.\w+$")
# Characters a folder name may not contain: " $ * / : < > ? \ | DEL..NBSP and Unicode spaces.
_BAD_FOLDER_CHARS = re.compile(
    r"[\x22\x24\x2a\x2f\x3a\x3c\x3e\x3f\x5c\x7c\x7f-\xa0\u1680\u180e\u2000-\u2006\u2008-\u200b\u2028\u2029\u205f\u3000]"
)
_VERSION = re.compile(r"^(\d+\.\d+|MAJOR|MINOR)$", re.IGNORECASE)
MAX_COMMENT = 1024


def validate_file_name(name: str) -> None:
    if not _FILE_NAME.match(name or ""):
        raise ClinicalError(
            f"{name!r} is not a valid file name: it needs an extension (such as .sas) and none of / < > ; \\ {{ }} ? #."
        )


def validate_folder_name(name: str) -> None:
    if not name or len(name) > 255:
        raise ClinicalError("A folder name must be 1 to 255 characters long.")
    if name.startswith("."):
        raise ClinicalError(f"Folder name {name!r} must not start with a dot.")
    if name != name.strip():
        raise ClinicalError(f"Folder name {name!r} must not start or end with a space.")
    if _BAD_FOLDER_CHARS.search(name):
        raise ClinicalError(f'Folder name {name!r} contains a character that is not allowed (" $ * / : < > ? \\ |).')


def validate_comment(comment: str | None) -> str:
    text = (comment or "").strip()
    if len(text) > MAX_COMMENT:
        raise ClinicalError(f"The comment is {len(text)} characters long; the limit is {MAX_COMMENT}.")
    return text


def validate_version(version: str | None, allow_bump: bool = True) -> str | None:
    """``version`` upper-cased (``1.0``, ``MAJOR``, ``MINOR``), or None when empty."""
    if version is None or str(version).strip() == "":
        return None
    value = str(version).strip().upper()
    if not _VERSION.match(value) or (not allow_bump and value in ("MAJOR", "MINOR")):
        allowed = "a number such as 1.0, or MAJOR / MINOR" if allow_bump else "a number such as 1.0"
        raise ClinicalError(f"Version {version!r} is not valid: use {allowed}.")
    return value


def is_versioned(file_version: str | None) -> bool:
    return bool(file_version) and file_version != "-"


def next_version(bump: str, current: str | None) -> str:
    """The next major (2.3 → 3.0) or minor (2.3 → 2.4) version of ``current``."""
    version = current if is_versioned(current) else "0.0"
    try:
        major, minor = (int(p) for p in str(version).split(".")[:2])
    except ValueError as exc:
        raise ClinicalError(f"{current!r} is not a version this tool can raise.") from exc
    return f"{major + 1}.0" if bump == "major" else f"{major}.{minor + 1}"


def version_key(version: str | None) -> tuple[int, ...]:
    """Sort key for versions: 10.0 sorts after 9.1, which a string sort gets wrong."""
    try:
        return tuple(int(p) for p in str(version).split("."))
    except ValueError:
        return (-1,)


# --- content ---------------------------------------------------------------------------

_ZIP_MAGIC = b"PK\x03\x04"


def unwrap_single_file_zip(data: bytes, name: str) -> bytes:
    """The one file inside ``data`` when the service wrapped a single file in a zip.

    A file that *is* a zip (``.zip``) is returned as is.
    """
    if not data.startswith(_ZIP_MAGIC) or name.lower().endswith(".zip"):
        return data
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            files = [i for i in archive.infolist() if not i.is_dir()]
            if len(files) == 1:
                return archive.read(files[0])
    except zipfile.BadZipFile:
        pass
    return data


def decode_text(data: bytes) -> str | None:
    """``data`` as text when it is text (UTF-8, or Latin-1 without control bytes), else None."""
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    text = data.decode("latin-1")
    if any(ord(c) < 32 and c not in "\t\r\n\f" for c in text[:8192]):
        return None
    return text


def parse_access_list(payload: Any) -> list[str] | None:
    """Permissions or privileges in any of the shapes the service answers with."""
    if isinstance(payload, dict):
        payload = payload.get("permissions") or payload.get("privileges") or payload.get("items")
    if isinstance(payload, str):
        payload = re.split(r"[,\s]+", payload)
    if isinstance(payload, list):
        return sorted({str(p).strip() for p in payload if str(p).strip()})
    return None


# --- shaping results ------------------------------------------------------------------

_ITEM_FIELDS = (
    "name",
    "path",
    "primaryType",
    "typeId",
    "size",
    "fileSize",
    "versioned",
    "fileVersion",
    "fileComment",
    "locked",
    "checkedOut",
    "checkedOutBy",
    "checkedOutByDisplayName",
    "signingStatus",
    "state",
    "owner",
    "description",
    "contentType",
    "digest",
    "modifiedTimeStamp",
    "modifiedBy",
    "createdBy",
    "creationTimeStamp",
)
SYNC_FIELDS = (
    "workspaceStatus",
    "checkedOut",
    "checkedOutBy",
    "checkedOutByDisplayName",
    "fileVersion",
    "synchronizationFileVersion",
    "synchronizationTimeStamp",
)


def repo_item_summary(item: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    """The fields of a repository item a model needs, without links and empty values."""
    out = {k: item[k] for k in _ITEM_FIELDS if item.get(k) not in (None, "", [])}
    if item.get("tags"):
        out["tags"] = item["tags"]
    if item.get("checkedOut") and user_id:
        out["checkedOutByYou"] = item.get("checkedOutBy") == user_id
    sync = item.get("synchronizationInfo") or {}
    if sync.get("workspaceStatus"):
        out["workspaceStatus"] = sync["workspaceStatus"]
    return out


def workspace_item_summary(item: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    out = {
        k: item[k]
        for k in ("name", "path", "type", "size", "contentType", "modifiedTimeStamp")
        if item.get(k) not in (None, "")
    }
    sync = {
        k: v for k, v in (item.get("synchronizationInfo") or {}).items() if k in SYNC_FIELDS and v not in (None, "")
    }
    if sync:
        if sync.get("checkedOut") and user_id:
            sync["checkedOutByYou"] = sync.get("checkedOutBy") == user_id
        out["sync"] = sync
    return out


def action_result(status: dict[str, Any]) -> dict[str, Any]:
    """An action status as a tool result: ``completed``, ``completed_with_warnings`` or ``failed``."""
    summary = status.get("summary") or {}
    completion = (summary.get("completionStatus") or "").upper()
    outcome = {"INFO": "completed", "WARN": "completed_with_warnings"}.get(completion, "failed")
    if not summary.get("endTimeStamp"):
        outcome = "still_running"
    result: dict[str, Any] = {
        "status": outcome,
        "action": summary.get("action"),
        "message": summary.get("message") or summary.get("detailMessage"),
        "items": [
            {
                "path": d.get("itemIdentifier") or d.get("itemName"),
                "status": d.get("completionStatus"),
                "message": d.get("message"),
            }
            for d in status.get("details") or []
        ],
    }
    if outcome == "still_running":
        result["action_status_id"] = summary.get("id")
        result["message"] = (
            "The server is still working on this action; it continues on its own. "
            "Check the items again in a little while."
        )
    return result


# --- batch actions --------------------------------------------------------------------

# How each batch action wants its body. Copy these exactly: the service refuses the others.
_BARE_ARRAY = {"UNDO_CHECKOUT", "COPY_LATEST_VERSION", "DELETE", "DOWNLOAD", "UPLOAD", "UPLOAD_EXPAND"}


def batch_body(action: str, paths: list[str], **options: Any) -> Any:
    """The request body of workspace/repository batch ``action``."""
    if action in _BARE_ARRAY:
        return paths
    if action == "REPOSITORY_CHECK_OUT":
        return {"paths": paths, "copyContent": bool(options.get("copy_content", True))}
    if action == "COPY_FOLDER_STRUCTURE":
        return {"paths": paths, "includeSubfolders": bool(options.get("include_subfolders", True))}
    if action == "COPY_SPECIFIC_VERSION":
        return [{"path": p, "fileVersion": options["file_version"]} for p in paths]
    if action == "ENABLE_VERSIONING":
        spec: dict[str, Any] = {}
        if options.get("file_version"):
            spec["fileVersion"] = options["file_version"]
        return {"comment": options.get("comment") or "", "fileSpecifications": [{"path": p, **spec} for p in paths]}
    if action == "DISABLE_VERSIONING":
        return {"comment": options.get("comment") or "", "paths": paths}
    raise ValueError(f"Unknown batch action {action!r}")


async def start_batch(client: httpx.AsyncClient, scope: str, action: str, body: Any) -> str:
    """Start a batch action and return its action-status id."""
    base = f"{WORKSPACE}/items" if scope == WORKSPACE_LOCATION else REPO_ITEMS
    resp = await client.post(
        f"{config.VIYA_ENDPOINT}{base}/batch",
        params={"action": action, "clientId": _CLIENT_ID},
        json=body,
        headers={"Accept": JSON, "Content-Type": JSON},
    )
    raise_for_viya_status(resp)
    location = resp.headers.get("Location") or resp.headers.get("location")
    if not location:
        raise ClinicalError(f"The server accepted the {action} action but named no action status to follow.", "failed")
    return location.rstrip("/").rsplit("/", 1)[-1]


async def wait_action(client: httpx.AsyncClient, status_id: str, timeout: float = 300) -> dict[str, Any]:
    """Poll an action status until it ends or ``timeout`` seconds pass; return the status."""
    deadline = _clock() + timeout
    while True:
        resp = await client.get(
            f"{config.VIYA_ENDPOINT}{ACTION_STATUS}/{status_id}", headers={"Accept": ACCEPT_ACTION_STATUS}
        )
        raise_for_viya_status(resp)
        status = resp.json()
        if (status.get("summary") or {}).get("endTimeStamp") or _clock() >= deadline:
            return status
        await _sleep(_POLL_INTERVAL)


async def run_batch(
    client: httpx.AsyncClient, scope: str, action: str, body: Any, timeout: float = 300
) -> dict[str, Any]:
    """Start a batch action, wait for it, and return it as a tool result."""
    status_id = await start_batch(client, scope, action, body)
    return action_result(await wait_action(client, status_id, timeout))


# --- service detection and plain requests ---------------------------------------------

_available: set[str] = set()
_SERVICES = {
    "repository": ("SAS Clinical Acceleration Repository", f"{REPO}/", None, ACCEPT_API),
    "jobs": ("SAS Clinical Job Management", f"{JOBS}/submissions", {"limit": 1}, JSON),
}


def clear_service_cache() -> None:
    _available.clear()


async def ensure_service(client: httpx.AsyncClient, service: str) -> dict[str, Any] | None:
    """None when ``service`` (``repository`` or ``jobs``) is there, else a ``service_unavailable`` result.

    A successful probe is remembered for the process; a missing service is asked
    again next time, so one that is deployed later is found without a restart.
    """
    if service in _available:
        return None
    title, path, params, accept = _SERVICES[service]
    try:
        resp = await client.get(
            f"{config.VIYA_ENDPOINT}{path}", params=params, headers={"Accept": accept}, follow_redirects=False
        )
    except httpx.HTTPError as exc:
        return {"status": "service_unavailable", "service": title, "message": f"Could not reach {title}: {exc}"}
    if resp.status_code == 200:
        _available.add(service)
        return None
    if resp.status_code in (401, 403):
        return None  # deployed; the call itself reports the authorization problem
    return {
        "status": "service_unavailable",
        "service": title,
        "message": (
            f"{title} is not available on this SAS Viya server (HTTP {resp.status_code} from {path}). "
            "These tools need SAS Clinical Acceleration"
            + (" with SAS Clinical Job Management." if service == "jobs" else ".")
        ),
    }


async def get(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    accept: str = JSON,
    missing: str | None = None,
) -> httpx.Response:
    """GET ``path``; a 404 becomes ``ClinicalError(missing, "not_found")`` when ``missing`` is given."""
    resp = await client.get(f"{config.VIYA_ENDPOINT}{path}", params=params, headers={"Accept": accept})
    if resp.status_code == 404 and missing:
        raise ClinicalError(missing, "not_found")
    raise_for_viya_status(resp)
    return resp


async def collect(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    accept: str = ACCEPT_COLLECTION,
    limit: int | None = None,
    missing: str | None = None,
) -> list[dict[str, Any]]:
    """Every item of a collection, following ``links[rel=next]``, up to ``limit`` items.

    The collection's ``count`` is not used: on several endpoints it is the size
    of the page, not of the collection.
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    resp = await get(client, path, params=params, accept=accept, missing=missing)
    for _ in range(MAX_PAGES):
        page = resp.json()
        if isinstance(page, list):  # some endpoints answer a bare array
            items.extend(page)
            break
        items.extend(page.get("items") or [])
        if limit is not None and len(items) >= limit:
            return items[:limit]
        nxt = next((link.get("href") for link in page.get("links") or [] if link.get("rel") == "next"), None)
        if not nxt or nxt in seen:
            break
        seen.add(nxt)
        resp = await get(client, nxt, accept=accept)
    return items if limit is None else items[:limit]


# --- repository --------------------------------------------------------------------------


async def resolve_repo_item(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    """The repository item at ``path`` (a ``not_found`` ClinicalError when there is none)."""
    resp = await get(
        client,
        f"{REPO}/repository/paths",
        params={"path": path},
        missing=f"Nothing at {path!r} in the Clinical Repository, or you may not see it. "
        "clinical_list_items or clinical_search_repository finds items.",
    )
    return resp.json()


async def repo_item(client: httpx.AsyncClient, item_id: str) -> tuple[dict[str, Any], str | None]:
    """The repository item with its synchronization info, and its ETag."""
    resp = await get(
        client,
        f"{REPO_ITEMS}/{quote(item_id, safe='')}",
        params={"includeSynchronizationInfo": "true"},
        missing=f"Repository item {item_id!r} no longer exists.",
    )
    return resp.json(), resp.headers.get("ETag")


async def repo_children(client: httpx.AsyncClient, item_id: str, limit: int) -> list[dict[str, Any]]:
    return await collect(
        client,
        f"{REPO_ITEMS}/{quote(item_id, safe='')}/children",
        params={"start": 0, "limit": min(limit, 500), "includeSynchronizationInfo": "true"},
        limit=limit,
    )


async def read_repo_content(client: httpx.AsyncClient, item: dict[str, Any], version: str | None = None) -> bytes:
    """The bytes of a repository file, at ``version`` or the latest."""
    item_id = quote(item["id"], safe="")
    if version:
        resp = await client.get(
            f"{config.VIYA_ENDPOINT}{REPO_ITEMS}/{item_id}/versions/{quote(version, safe='')}/content"
        )
        if resp.status_code == 404:
            raise ClinicalError(
                f"{item.get('path')} has no version {version!r}; clinical_list_versions lists them.", "not_found"
            )
    else:
        resp = await client.post(f"{config.VIYA_ENDPOINT}{REPO_ITEMS}/{item_id}/content", json=[""])
    raise_for_viya_status(resp)
    return unwrap_single_file_zip(resp.content, item.get("name") or "")


async def item_access(client: httpx.AsyncClient, item_id: str) -> dict[str, Any]:
    """The caller's permissions and versioning privileges on an item, ``None`` where unreadable."""
    out: dict[str, Any] = {}
    for key in ("permissions", "privileges"):
        resp = await client.get(
            f"{config.VIYA_ENDPOINT}{REPO}/authorizations/{quote(item_id, safe='')}/{key}", headers={"Accept": JSON}
        )
        if resp.status_code == 200:
            out[key] = parse_access_list(resp.json())
        elif resp.status_code == 403 and key == "permissions":
            out[key] = ["READ"]
        elif resp.status_code == 404 and key == "privileges":
            out[key] = []  # errorCode 145191: none granted
        else:
            out[key] = None  # unknown (this release answers 500 for files): the server decides
    return out


# --- workspace -------------------------------------------------------------------------


async def workspace_item(client: httpx.AsyncClient, path: str) -> dict[str, Any] | None:
    """The workspace item at ``path`` with its sync info, or None when there is none."""
    resp = await client.get(
        f"{config.VIYA_ENDPOINT}{WORKSPACE}/items{url_path(path)}",
        params={"includeSynchronizationInfo": "true", "type": "FILE"},
        headers={"Accept": JSON},
    )
    if resp.status_code == 404:
        return None
    raise_for_viya_status(resp)
    return resp.json()


async def workspace_children(client: httpx.AsyncClient, path: str, limit: int) -> list[dict[str, Any]]:
    suffix = "" if path == "/" else url_path(path)
    return await collect(
        client,
        f"{WORKSPACE}/items{suffix}",
        params={"includeSynchronizationInfo": "true", "start": 0, "limit": min(limit, 500)},
        limit=limit,
        missing=(
            f"Nothing at {path!r} in your Clinical Workspace. "
            "clinical_list_items with location='workspace' lists it."
        ),
    )


async def ensure_workspace_folder(client: httpx.AsyncClient, folder: str) -> list[str]:
    """Create ``folder`` and any missing parents in the workspace; return the folders created.

    A file can only be written into a workspace folder that exists, and the
    workspace mirrors the repository's layout, so a new file for
    ``/Org/Study/programs`` first needs those folders in the workspace too.
    """
    created: list[str] = []
    parts = [p for p in folder.split("/") if p]
    for depth in range(1, len(parts) + 1):
        path = "/" + "/".join(parts[:depth])
        if await workspace_item(client, path) is not None:
            continue
        parent, name = split_path(path)
        suffix = "" if parent == "/" else url_path(parent)
        resp = await client.post(
            f"{config.VIYA_ENDPOINT}{WORKSPACE}/folders{suffix}", params={"name": name}, headers={"Accept": JSON}
        )
        raise_for_viya_status(resp)
        created.append(path)
    return created


async def read_workspace_content(client: httpx.AsyncClient, path: str) -> bytes:
    resp = await client.post(f"{config.VIYA_ENDPOINT}{WORKSPACE}/items/content", json={"paths": [path]})
    if resp.status_code == 404:
        raise ClinicalError(f"Nothing at {path!r} in your Clinical Workspace.", "not_found")
    raise_for_viya_status(resp)
    return unwrap_single_file_zip(resp.content, split_path(path)[1])


async def put_workspace_file(
    client: httpx.AsyncClient,
    target: str,
    name: str,
    data: bytes,
    *,
    expand: bool = False,
    action_status_id: str | None = None,
) -> httpx.Response:
    """PUT bytes to the workspace as a multipart ``file`` part.

    ``target`` is the file's own path for a plain write, the folder for an
    expanded zip (the root needs ``path=/``).
    """
    params: dict[str, Any] = {"expand": str(expand).lower()}
    if action_status_id:
        params["actionStatusId"] = action_status_id
    suffix = url_path(target)
    if expand and target == "/":
        suffix, params["path"] = "", "/"
    resp = await client.put(
        f"{config.VIYA_ENDPOINT}{WORKSPACE}/items{suffix}",
        params=params,
        files={"file": (name, data, "application/octet-stream")},
        headers={"Accept": JSON},
    )
    raise_for_viya_status(resp)
    return resp


async def check_in_candidates(client: httpx.AsyncClient, paths: list[str]) -> list[dict[str, Any]]:
    """The files below ``paths`` the server offers for check-in, with their sync info."""
    resp = await client.post(
        f"{config.VIYA_ENDPOINT}{REPO}/synchronization/items",
        params={"type": "CHECKIN"},
        json=paths,
        headers={"Accept": ACCEPT_COLLECTION, "Content-Type": JSON},
    )
    raise_for_viya_status(resp)
    body = resp.json()
    return body.get("items", []) if isinstance(body, dict) else list(body)


def plan_check_in(
    candidates: list[dict[str, Any]],
    selected: Iterable[str],
    user_id: str | None,
    *,
    bump: str,
    new_file_version: str | None,
    comment: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split check-in candidates into the ``CHECK_IN`` body, what goes in, and what is left out.

    Ported from the extension's ``sortCheckInCandidates`` / ``planCheckIn`` /
    ``buildCheckInBody``: a copy that is not checked out cannot add a version
    (an unchanged copy is dropped quietly), a file checked out by someone else
    cannot be checked in, and a selected file the server did not offer is
    reported. New and checked-out unversioned files get ``new_file_version``
    (none keeps them unversioned); versioned files get their next ``bump``.
    """
    specs: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def block(info: dict[str, Any]) -> dict[str, Any] | None:
        if info.get("checkedOut"):
            by = info.get("checkedOutBy")
            if user_id and by and by != user_id:
                return {"reason": "checked_out_by_other", "by": info.get("checkedOutByDisplayName") or by}
            return None
        if info.get("workspaceStatus") in ("IN_SYNC", "OUT_OF_SYNC"):
            return {"reason": "not_checked_out"}
        return None

    for item in candidates:
        info = item.get("synchronizationInfo") or {}
        reason = block(info)
        if reason:
            if reason["reason"] != "not_checked_out" or info.get("workspaceStatus") == "OUT_OF_SYNC":
                skipped.append({"path": item.get("path"), **reason})
            continue
        if not info.get("checkedOut"):
            version, kind = new_file_version, "new"
        elif is_versioned(info.get("fileVersion")):
            version, kind = next_version(bump, info.get("fileVersion")), "new_version"
        else:
            version, kind = new_file_version, "unversioned"
        spec: dict[str, Any] = {"path": item.get("path"), "type": "FILE"}
        if version:
            spec["fileVersion"] = version.upper()
        specs.append(spec)
        included.append({"path": item.get("path"), "kind": kind, "version": version})
    offered = {c.get("path") for c in candidates}
    for path in selected:
        if path not in offered and not any(str(o).startswith(path.rstrip("/") + "/") for o in offered):
            skipped.append({"path": path, "reason": "not_offered"})
    body = {"fileSpecifications": specs, "comment": comment, "includeNewFiles": True}
    return body, included, skipped


VERSIONING_PRIVILEGE_HINT = (
    "Creating a versioned file needs a versioning privilege, which this user may not have: the "
    "server fails such requests with an internal error instead of refusing them. Retry without a "
    "version (the file is then unversioned), or ask a SAS Clinical Acceleration administrator for "
    "the privilege. clinical_get_item shows your versioning privileges where the server reports them."
)


def looks_like_missing_privilege(result: dict[str, Any]) -> bool:
    """Whether a failed action reads like the server's crash on a missing versioning privilege."""
    texts = [str(result.get("message") or "")] + [str(i.get("message") or "") for i in result.get("items") or []]
    return any("Privilege" in t and "null" in t for t in texts)


SKIP_REASONS = {
    "not_checked_out": (
        "a copy of a repository file you have not checked out; only a check-out can add a version. "
        "clinical_check_out with with_content=false keeps your workspace file and makes it eligible."
    ),
    "checked_out_by_other": "checked out by another user; only they can check it in.",
    "not_offered": (
        "the server did not offer it for check-in: it may be unchanged, a folder without files, "
        "or not in your workspace."
    ),
}


# --- datasets into WORK (ported from the extension's xptFormat.ts) ---------------------

_BYTES_PER_LINE = 57
_LINES_PER_CHUNK = 13_000
BYTES_PER_CHUNK = _BYTES_PER_LINE * _LINES_PER_CHUNK
_PLAIN_NAME = re.compile(r"^[\w.-]+$")
_TRANSPORT_HEADERS = {
    "xport5": "HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!",
    "xport8": "HEADER RECORD*******LIBV8   HEADER RECORD!!!!!!!",
    "cport": "**COMPRESSED** **COMPRESSED**",
}


def detect_transport_format(data: bytes) -> str | None:
    text = data[:80].decode("latin-1")
    return next((fmt for fmt, header in _TRANSPORT_HEADERS.items() if text.startswith(header)), None)


def work_path_literal(subdir: str, name: str | None = None) -> str:
    """A directory or file under WORK as a SAS string literal; SAS resolves the WORK path itself."""
    for part in (subdir, name or "x"):
        if not _PLAIN_NAME.match(part):
            raise ClinicalError(f"{part!r} cannot be used as a file name under WORK (letters, digits, _ . - only).")
    return f'"%sysfunc(pathname(work))/{subdir}{"/" + name if name else ""}"'


def work_subdir_code(subdir: str) -> str:
    work_path_literal(subdir)
    return f'data _null_; length d $1024; d=dcreate("{subdir}", pathname("work")); run;'


def base64_write_code(file_literal: str, chunk: bytes, append: bool) -> str:
    """SAS that decodes base64 DATALINES and writes the raw bytes to ``file_literal``."""
    lines = [
        base64.b64encode(chunk[i : i + _BYTES_PER_LINE]).decode("ascii") for i in range(0, len(chunk), _BYTES_PER_LINE)
    ]
    return "\n".join(
        [
            "options nosource nonotes;",
            "data _null_;",
            "  infile datalines truncover;",
            "  input line $76.;",
            "  length raw $57;",
            f"  file {file_literal} recfm=n{' mod' if append else ''};",
            "  n = length(line);",
            "  if n < 4 then return;",
            '  pad = (char(line, n) = "=") + (char(line, n - 1) = "=");',
            '  raw = inputc(substr(line, 1, n), cats("$base64x", n, "."));',
            "  rawlen = n / 4 * 3 - pad;",
            "  put raw $varying57. rawlen;",
            "datalines;",
            *lines,
            ";",
            "run;",
            "options notes source;",
        ]
    )


def upload_to_work_code(subdir: str, name: str, data: bytes) -> list[str]:
    """The SAS programs (one per ~740 KB chunk) that write ``data`` to ``WORK/subdir/name``."""
    target = work_path_literal(subdir, name)
    chunks = [data[i : i + BYTES_PER_CHUNK] for i in range(0, len(data), BYTES_PER_CHUNK)] or [b""]
    programs = [base64_write_code(target, chunk, i > 0) for i, chunk in enumerate(chunks)]
    programs[0] = work_subdir_code(subdir) + "\n" + programs[0]
    return programs


def transport_load_code(fmt: str, subdir: str, name: str, libref: str) -> str:
    """SAS that unpacks a transport file in ``WORK/subdir/name`` into ``libref``."""
    file_literal = work_path_literal(subdir, name)
    target_dir = libref.lower()
    setup = [
        work_subdir_code(target_dir),
        f"libname {libref} {work_path_literal(target_dir)};",
        f"proc datasets lib={libref} kill nolist nowarn; quit;",
    ]
    if fmt == "xport5":
        body = [
            f"libname _XPTIN xport {file_literal};",
            f"proc copy in=_XPTIN out={libref} memtype=data; run;",
            "libname _XPTIN clear;",
        ]
    elif fmt == "xport8":
        body = [f"%xpt2loc(libref={libref}, filespec={file_literal})"]
    else:
        body = [
            f"filename _XPTIN {file_literal};",
            f"proc cimport infile=_XPTIN library={libref} memtype=data; run;",
            "filename _XPTIN clear;",
        ]
    return "\n".join(setup + body)


_SAS_ERROR = re.compile(r"^ERROR(\s+\d+-\d+)?:")


def find_sas_errors(log: str) -> list[str]:
    """ERROR lines of a SAS log, with their indented continuation lines joined on."""
    errors: list[str] = []
    current = False
    for raw in log.splitlines():
        line = raw.strip()
        if _SAS_ERROR.match(line):
            errors.append(line)
            current = True
        elif current and raw[:1].isspace() and line:
            errors[-1] = f"{errors[-1]} {line}"
        else:
            current = False
    return errors


# --- clinical jobs --------------------------------------------------------------------

PARAMETER_TYPES = ("CHARACTER", "CHARACTER_MASKED", "NUMERIC", "DATE", "FILE", "FOLDER")
_SAS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
RUN_MODES = {"run": "RUN", "test": "TEST", "run_and_populate": "RUN_AND_POPULATE"}
UNFINISHED_STATES = ("NOT_STARTED", "STARTED", "RUNNING_TASKS", "CANCELING", "POST_EXECUTION_PUBLISHING")


def run_category(state: str | None) -> str:
    """``running``, ``successful``, ``warnings``, ``canceled`` or ``error`` for a run state.

    A check-in error counts as an error: the audit trail of the run is incomplete.
    """
    value = (state or "").upper()
    if not value.startswith("COMPLETED"):
        return "running"
    if value == "COMPLETED_SUCCESSFUL":
        return "successful"
    if value == "COMPLETED_PROGRAM_WARNINGS":
        return "warnings"
    if value == "COMPLETED_CANCELED":
        return "canceled"
    return "error"


def normalize_submission(sub: dict[str, Any]) -> dict[str, Any]:
    """A submission in one shape, whether the service answered flat fields or the spec's nested job."""
    job = sub.get("job") or {}
    manifest = sub.get("manifest") or {}
    out = {
        "run_id": sub.get("id"),
        "job_name": sub.get("name") or job.get("name"),
        "job_path": sub.get("jobPath") or job.get("path"),
        "location": (sub.get("jobResourceLocation") or job.get("resourceLocation") or "").lower() or None,
        "job_version": None if sub.get("jobVersion") in (None, "-") else sub.get("jobVersion"),
        "state": sub.get("state"),
        "category": run_category(sub.get("state")),
        "kind": run_kind(sub.get("jobRunContext")),
        "run_context": sub.get("jobRunContext"),
        "submitted_by": sub.get("submittedBy"),
        "run_as": sub.get("runAsUser"),
        "submitted": sub.get("submittedTimeStamp"),
        "completed": sub.get("completedTimeStamp"),
        "info": sub.get("info"),
        "manifest_path": manifest.get("path"),
    }
    failures = sub.get("checkinFailures") or []
    if failures:
        out["checkin_failures"] = [
            {"path": f.get("itemIdentifier") or f.get("itemName"), "message": f.get("message")} for f in failures
        ]
    return {k: v for k, v in out.items() if v not in (None, "")}


def run_kind(context: str | None) -> str:
    value = (context or "").upper()
    if value == "INTERACTIVE_TEST":
        return "test"
    if value.startswith("INTERACTIVE_RUN_AND_POPULATE"):
        return "run_and_populate"
    if value == "SCHEDULER":
        return "scheduled"
    return "run"


def rerun_options(context: str | None) -> tuple[str, bool]:
    """The ``(mode, relative_paths)`` that repeats a run of ``context``."""
    value = (context or "").upper()
    if value == "INTERACTIVE_TEST":
        return "test", False
    if value == "INTERACTIVE_RUN_AND_POPULATE_RELATIVE":
        return "run_and_populate", True
    if value.startswith("INTERACTIVE_RUN_AND_POPULATE"):
        return "run_and_populate", False
    return "run", False


def job_reference(path: str, location: str) -> dict[str, str]:
    """The ``job`` object of a run or schedule request: path-based, nothing else."""
    parent, name = split_path(path)
    return {"name": name, "path": path, "location": parent, "resourceLocation": location.upper()}


def resolve_job_path(job_path: str, value: str) -> str:
    """A path in a job (``./x.sas``, ``../data``) as an absolute path."""
    if value.startswith("/"):
        return normalize_path(value)
    return normalize_path(posixpath.normpath(posixpath.join(split_path(job_path)[0], value)))


def encode_parameter_values(
    definitions: list[dict[str, Any]], values: dict[str, Any] | None, job_path: str
) -> list[dict[str, Any]]:
    """``parameterValues`` for a run: each job parameter from ``values`` or its default.

    CHARACTER, CHARACTER_MASKED and NUMERIC are plain strings (the service
    refuses the spec's ``{value}`` form), DATE is midnight UTC, FILE is
    ``{path}`` and FOLDER ``{path, includeSubfolders}``, both absolute.
    """
    given = {str(k).lower(): v for k, v in (values or {}).items()}
    known = {str(p.get("name", "")).lower() for p in definitions}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ClinicalError(
            f"The job has no parameter named {', '.join(unknown)}; its parameters are "
            f"{', '.join(sorted(p.get('name', '') for p in definitions)) or 'none'}."
        )
    out = []
    for param in definitions:
        name = str(param.get("name") or "")
        ptype = (param.get("type") or "CHARACTER").upper()
        value = given.get(name.lower(), param.get("defaultValue"))
        if value in (None, "") or value == {}:
            raise ClinicalError(f"Parameter {name!r} ({ptype}) has no default; pass a value in parameter_values.")
        out.append({"name": name, "type": ptype, "value": _encode_value(name, ptype, value, job_path)})
    return out


def _encode_value(name: str, ptype: str, value: Any, job_path: str) -> Any:
    if ptype in ("FILE", "FOLDER"):
        raw = value if isinstance(value, dict) else {"path": value}
        path = raw.get("path")
        if not path:
            raise ClinicalError(f"Parameter {name!r} ({ptype}) needs a path.")
        encoded: dict[str, Any] = {"path": resolve_job_path(job_path, str(path))}
        if ptype == "FOLDER":
            encoded["includeSubfolders"] = bool(raw.get("includeSubfolders", raw.get("include_subfolders", False)))
        elif raw.get("version"):
            encoded["version"] = str(raw["version"])
        return encoded
    if isinstance(value, dict):
        value = value.get("value", "")
    text = str(value).strip()
    if ptype == "NUMERIC":
        try:
            float(text)
        except ValueError as exc:
            raise ClinicalError(f"Parameter {name!r} is NUMERIC; {value!r} is not a number.") from exc
    if ptype == "DATE":
        if not _DATE.match(text):
            raise ClinicalError(f"Parameter {name!r} is a DATE; pass it as YYYY-MM-DD, not {value!r}.")
        text = f"{text[:10]}T00:00:00Z"
    return text


def validate_job(job: dict[str, Any]) -> list[str]:
    """Problems with a ``.cjob`` definition, as the job editor reports them; empty when fine."""
    problems: list[str] = []
    programs = job.get("programs") or []
    if not programs:
        problems.append("The job runs no programs: add at least one to 'programs'.")
    for i, program in enumerate(programs):
        if not isinstance(program, dict) or not program.get("path"):
            problems.append(f"programs[{i}] needs a 'path'.")
    seen: set[str] = set()
    for i, param in enumerate(job.get("parameters") or []):
        name = str((param or {}).get("name") or "")
        if not _SAS_NAME.match(name):
            problems.append(
                f"parameters[{i}]: {name!r} is not a valid SAS name (letter or _, then up to 31 letters, digits, _)."
            )
        elif name.lower() in seen:
            problems.append(f"parameters[{i}]: {name!r} is used twice (names ignore case).")
        seen.add(name.lower())
        ptype = str((param or {}).get("type") or "").upper()
        if ptype not in PARAMETER_TYPES:
            problems.append(f"parameters[{i}]: type must be one of {', '.join(PARAMETER_TYPES)}.")
    checkin = job.get("publishCheckin") or {}
    version = checkin.get("fileVersion")
    if version not in (None, "", "MAJOR", "MINOR") and not re.match(r"^\d+\.\d+$", str(version)):
        problems.append("publishCheckin.fileVersion must be empty, MAJOR, MINOR or a version such as 1.0.")
    return problems


def program_job_definition(program_path: str, job_path: str) -> dict[str, Any]:
    """The create request of a job that runs one program (the extension's "Run as Clinical Job")."""
    job_parent, job_name = split_path(job_path)
    prog_parent, prog_name = split_path(program_path)
    return {
        "resource": {"name": job_name, "path": job_path, "location": job_parent, "resourceLocation": "WORKSPACE"},
        "programs": [{"path": program_path, "name": prog_name, "location": prog_parent}],
    }


def job_runs_only(job: dict[str, Any], program_path: str, job_path: str) -> bool:
    """Whether ``job`` runs exactly the one program ``program_path``."""
    programs = job.get("programs") or []
    return len(programs) == 1 and resolve_job_path(job_path, str(programs[0].get("path") or "")) == program_path


def fill_checkin_locations(manifest: dict[str, Any], cmnf: dict[str, Any] | None) -> dict[str, Any]:
    """``manifest`` with the check-in locations of its ``.cmnf`` file when the service left them out."""
    spec = dict(manifest.get("checkinSpecification") or {})
    if spec.get("locations") or not cmnf:
        return manifest
    locations = []
    for loc in (cmnf.get("checkinSpecification") or {}).get("locations") or []:
        if isinstance(loc, str):
            entry: dict[str, Any] = {"path": loc}
        elif isinstance(loc, dict):
            path = loc.get("path") or (loc.get("resource") or {}).get("path") or (loc.get("file") or {}).get("path")
            if not path:
                continue
            entry = {"path": path}
            if loc.get("includeSubfolders") is True:
                entry["includeSubfolders"] = True
        else:
            continue
        locations.append(entry)
    if not locations:
        return manifest
    file_spec = {k: v for k, v in (cmnf.get("checkinSpecification") or {}).items() if k != "locations"}
    return {**manifest, "checkinSpecification": {**file_spec, **spec, "locations": locations}}


def _resource(entry: dict[str, Any]) -> dict[str, Any]:
    res = entry.get("resource") or entry.get("file") or entry
    out = {
        "path": res.get("path"),
        "location": (res.get("resourceLocation") or "").lower() or None,
        "version": None if res.get("fileVersion") in (None, "-") else res.get("fileVersion"),
        "size": res.get("size"),
        "modified": res.get("lastModified") or res.get("modifiedTimeStamp"),
        "md5": (entry.get("digest") or res.get("digest") or {}).get("value")
        if isinstance(entry.get("digest") or res.get("digest"), dict)
        else (entry.get("digest") or res.get("digest")),
    }
    return {k: v for k, v in out.items() if v not in (None, "")}


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """A run's manifest as the audit record a reviewer reads: who, what, which versions."""
    sub = manifest.get("jobSubmission") or {}
    logs = {r.get("path") for r in (_resource(e) for e in manifest.get("logs") or []) if r.get("path")}
    results = {r.get("path") for r in (_resource(e) for e in manifest.get("results") or []) if r.get("path")}
    params = []
    for p in manifest.get("parameters") or []:
        value = p.get("value")
        if (p.get("type") or "").upper() == "CHARACTER_MASKED":
            value = "*****"
        params.append({"name": p.get("name"), "type": p.get("type"), "value": value})
    out = {
        "run_id": sub.get("id"),
        "state": sub.get("state"),
        "category": run_category(sub.get("state")),
        "kind": run_kind(sub.get("jobRunContext") or sub.get("context")),
        "location": (manifest.get("type") or "").lower() or None,
        "job": _resource(manifest.get("job") or {}),
        "submitted_by": sub.get("submittedBy") or sub.get("runAsUser"),
        "submitted": sub.get("submittedTimeStamp") or sub.get("submitted"),
        "completed": sub.get("completedTimeStamp") or sub.get("completed"),
        "total_execution_ms": manifest.get("totalExecutionTime"),
        "programs": [
            {
                **_resource(p),
                "result": p.get("executionState"),
                "execution_ms": p.get("executionTime"),
                **({"info": p["executionInfo"]} if p.get("executionInfo") else {}),
            }
            for p in manifest.get("programs") or []
        ],
        "parameters": params,
        "inputs": [_resource(e) for e in manifest.get("inputs") or []],
        "external_inputs": [
            {"name": e.get("name"), "location": e.get("location"), "program": e.get("referencingProgram")}
            for e in manifest.get("externalInputs") or []
        ],
        "outputs": [
            r for r in (_resource(e) for e in manifest.get("outputs") or []) if r.get("path") not in logs | results
        ],
        "external_outputs": [
            {"name": e.get("name"), "location": e.get("location"), "program": e.get("referencingProgram")}
            for e in manifest.get("externalOutputs") or []
        ],
        "logs": sorted(p for p in logs if p),
        "results": sorted(p for p in results if p),
        "manifest_path": _resource(manifest.get("manifest") or {}).get("path"),
        "checkin": manifest.get("checkinSpecification") or None,
    }
    return {k: v for k, v in out.items() if v not in (None, [], "")}


_SCHEDULE_TYPES = ("MINUTELY", "HOURLY", "DAILY", "WEEKLY", "MONTHLY", "YEARLY", "DATELIST")
_DAYS = ("SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT")
_OCCURRENCES = ("FIRST", "SECOND", "THIRD", "FOURTH", "LAST")
_MONTHS = (
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
)


def build_schedule_trigger(trigger: dict[str, Any]) -> dict[str, Any]:
    """A validated schedule ``trigger`` in the shape the service takes.

    ``type`` is MINUTELY, HOURLY, DAILY, WEEKLY, MONTHLY, YEARLY or DATELIST
    (a one-off run is a DATELIST with one date). Hours and minutes travel as
    unpadded strings; ``timeZone`` (IANA, e.g. ``Europe/Berlin``) is required.
    """
    t = {str(k): v for k, v in (trigger or {}).items()}
    ttype = str(t.get("type") or "").upper()
    if ttype not in _SCHEDULE_TYPES:
        raise ClinicalError(f"trigger.type must be one of {', '.join(_SCHEDULE_TYPES)}.")
    zone = t.get("timeZone") or t.get("time_zone")
    if not zone:
        raise ClinicalError(
            "trigger.timeZone is required, as an IANA name such as 'Europe/Berlin' or 'America/New_York'."
        )
    out: dict[str, Any] = {"type": ttype, "timeZone": str(zone)}

    def hm(key: str, high: int) -> str:
        value = t.get(key)
        if value is None or str(value).strip() == "":
            raise ClinicalError(f"trigger.{key} is required for a {ttype} schedule.")
        try:
            number = int(str(value))
        except ValueError as exc:
            raise ClinicalError(f"trigger.{key} must be a whole number.") from exc
        if not 0 <= number <= high:
            raise ClinicalError(f"trigger.{key} must be between 0 and {high}.")
        return str(number)

    if ttype == "DATELIST":
        dates = t.get("dates") or []
        if isinstance(dates, str):
            dates = [dates]
        if not dates or not all(_DATE.match(str(d)) for d in dates):
            raise ClinicalError("A DATELIST schedule needs trigger.dates as a list of YYYY-MM-DD dates.")
        out.update({"dates": [str(d)[:10] for d in dates], "hours": hm("hours", 23), "minutes": hm("minutes", 59)})
        return out
    out["interval"] = max(1, int(t.get("interval") or 1))
    if ttype == "HOURLY":
        out["minutes"] = hm("minutes", 59)
    elif ttype != "MINUTELY":
        out["hours"], out["minutes"] = hm("hours", 23), hm("minutes", 59)
    if ttype == "WEEKLY":
        days = [str(d).upper()[:3] for d in (t.get("daysOfWeek") or [])]
        if not days or any(d not in _DAYS for d in days):
            raise ClinicalError("A WEEKLY schedule needs trigger.daysOfWeek, e.g. ['MON', 'THU'].")
        out["daysOfWeek"] = days
    if ttype in ("MONTHLY", "YEARLY"):
        if t.get("dayOfMonth"):
            out["dayOfMonth"] = int(t["dayOfMonth"])
        else:
            occurrence = str(t.get("occurrence") or "").upper()
            days = [str(d).upper()[:3] for d in (t.get("daysOfWeek") or [])]
            if occurrence not in _OCCURRENCES or len(days) != 1 or days[0] not in _DAYS:
                raise ClinicalError(
                    f"A {ttype} schedule needs trigger.dayOfMonth (e.g. 15), or trigger.occurrence "
                    "(FIRST…FOURTH or LAST) with one day in trigger.daysOfWeek (e.g. ['MON'])."
                )
            out["occurrence"], out["daysOfWeek"] = occurrence, days
    if ttype == "YEARLY":
        month = str(t.get("monthOfYear") or "").upper()
        if month not in _MONTHS:
            raise ClinicalError("A YEARLY schedule needs trigger.monthOfYear, e.g. 'JANUARY'.")
        out["monthOfYear"] = month
    for key in ("start", "end"):
        if t.get(key):
            if not _DATE.match(str(t[key])):
                raise ClinicalError(f"trigger.{key} must be a date, YYYY-MM-DD.")
            out[key] = str(t[key])[:10]
    if t.get("numOccurrences") and int(t["numOccurrences"]) > 0:
        out["numOccurrences"] = int(t["numOccurrences"])
    return out


def summarize_schedule(schedule: dict[str, Any], user_id: str | None = None) -> dict[str, Any]:
    info = schedule.get("jobInfo") or {}
    job = info.get("job") or {}
    trigger = {k: v for k, v in (schedule.get("trigger") or {}).items() if k != "version"}
    if "startDate" in trigger and "start" not in trigger:
        trigger["start"] = trigger.pop("startDate")
    values = []
    for v in info.get("parameterValues") or []:
        value = "*****" if (v.get("type") or "").upper() == "CHARACTER_MASKED" else v.get("value")
        values.append({"name": v.get("name"), "type": v.get("type"), "value": value})
    owner = schedule.get("runAs") or schedule.get("createdBy")
    out = {
        "schedule_id": schedule.get("id"),
        "name": schedule.get("name"),
        "description": schedule.get("description"),
        "status": schedule.get("status"),
        "job_path": job.get("path"),
        "location": (job.get("resourceLocation") or "").lower() or None,
        "job_available": schedule.get("jobAvailable"),
        "owner": owner,
        "yours": None if not user_id else owner == user_id,
        "trigger": trigger,
        "parameter_values": values,
    }
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def schedule_parameter_values(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    """A schedule's echoed values as a run request takes them (``label: "null"`` and ``version`` dropped)."""
    return [
        {"name": v.get("name"), "type": v.get("type"), "value": v.get("value")}
        for v in (schedule.get("jobInfo") or {}).get("parameterValues") or []
    ]


def parse_json_bytes(data: bytes, what: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClinicalError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ClinicalError(f"{what} is not a JSON object.")
    return value
