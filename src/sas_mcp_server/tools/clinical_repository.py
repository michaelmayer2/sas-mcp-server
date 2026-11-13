# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 12 — Clinical Repository (SAS Clinical Acceleration) tools.

SAS Clinical Acceleration keeps clinical study files in a controlled
repository. Changes go through each user's private workspace on the server:
copy or check out files, edit them there, check them back in as new
versions. These tools do what the Positron SAS extension's Clinical
Acceleration views do, under the caller's own identity and CA permissions,
plus repository search, audit history and signature history.

Paths address everything: the same path names a file in the repository and
its copy in the workspace. ``location`` says which of the two a call means.
Electronic signing is deliberately not offered: a signature needs the
signer's own authenticator PIN.
"""

import difflib
import mimetypes
import posixpath
from collections.abc import Awaitable, Callable
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from fastmcp import Context, FastMCP
from fastmcp.utilities.types import File
from pydantic import BeforeValidator

from .. import config
from ..helpers import clinical_helpers as ch
from ..helpers.upload_sources import resolve_source_bytes
from ..user_log import resolve_user_id
from ..viya_client import filter_literal, in_filter, raise_for_viya_status
from ..viya_utils import submit_job, wait_job
from ._common import coerce_str_or_json_list, make_session_helpers

PathList = Annotated[list[str], BeforeValidator(coerce_str_or_json_list)]
_MAX_DIFF_LINES = 2000
_DATASET_EXTENSIONS = (".sas7bdat", ".xpt")


async def _user_id(client: httpx.AsyncClient) -> str | None:
    """The caller's Viya user id (cached per token by user_log)."""
    auth = client.headers.get("Authorization", "")
    return await resolve_user_id(auth.removeprefix("Bearer ").strip() or None)


def _paths(paths: list[str]) -> list[str]:
    out = [ch.normalize_path(p) for p in paths if str(p).strip()]
    if not out:
        ch.fail("Pass at least one path.")
    return out


def _binary_result(data: bytes, name: str, path: str, version: str | None) -> Any:
    """Content too binary for text: an embedded file, or a refusal past the inline limit."""
    if len(data) > config.MAX_EXPORT_INLINE_BYTES:
        return {
            "status": "export_too_large",
            "path": path,
            "size_bytes": len(data),
            "limit_bytes": config.MAX_EXPORT_INLINE_BYTES,
            "message": (
                f"{path} is {len(data):,} bytes, over the {config.MAX_EXPORT_INLINE_BYTES:,}-byte inline limit "
                "(MAX_EXPORT_INLINE_BYTES). Datasets can be loaded into SAS with clinical_load_dataset instead."
            ),
        }
    ext = posixpath.splitext(name)[1].lstrip(".").lower() or "bin"
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    stem = posixpath.splitext(name)[0] + (f"-{version}" if version else "")
    return File(data=data, format=ext, name=stem).to_resource_content(mime_type=mime)


def register(mcp: FastMCP, get_token: Callable[[Context], Awaitable[str]]) -> None:
    """Register Tier 12 (Clinical Repository) tools on *mcp*."""

    viya_session, compute_tool_session = make_session_helpers(get_token)

    async def read_bytes(client: httpx.AsyncClient, path: str, location: str, version: str | None) -> bytes:
        if location == ch.REPOSITORY:
            item = await ch.resolve_repo_item(client, path)
            if item.get("primaryType") not in ("FILE", "JOB_FILE"):
                ch.fail(f"{path} is a {str(item.get('primaryType')).lower()}, not a file.")
            return await ch.read_repo_content(client, item, version)
        if version:
            ch.fail("Versions exist only in the repository; leave version empty for a workspace file.")
        return await ch.read_workspace_content(client, path)

    # --- browse and read ----------------------------------------------------------------

    @mcp.tool()
    async def clinical_list_items(
        ctx: Context, path: str = "/", location: str = "repository", limit: int = 200
    ) -> dict[str, Any]:
        """List the files and folders in a Clinical Repository or Clinical Workspace folder.

        Each item shows its type, version, lock and check-out state (and whether
        you hold the check-out), and in the repository how its workspace copy
        compares (``workspaceStatus``: IN_SYNC, OUT_OF_SYNC, WORKSPACE_FILE_MISSING,
        NOT_SYNCED). Top-level repository items are contexts (organizations,
        studies) that administrators manage.

        Args:
            path: Folder to list, e.g. ``/PositLabs/Study1/programs``; ``/`` is the top.
            location: ``repository`` (the shared, controlled store) or ``workspace``
                (your own area, where files are edited).
            limit: Maximum items (default 200).
        """
        async with viya_session("clinical_list_items", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                where = ch.check_location(location)
                folder = ch.normalize_path(path)
                user = await _user_id(client)
                limit = max(1, min(int(limit), 2000))
                if where == ch.REPOSITORY:
                    item = await ch.resolve_repo_item(client, folder)
                    if item.get("primaryType") not in ("CONTEXT", "FOLDER"):
                        ch.fail(f"{folder} is a file; clinical_get_item shows its details.")
                    children = await ch.repo_children(client, item["id"], limit)
                    items = [ch.repo_item_summary(c, user) for c in children]
                else:
                    children = await ch.workspace_children(client, folder, limit)
                    items = [ch.workspace_item_summary(c, user) for c in children]
            except ch.ClinicalError as exc:
                return exc.result()
            return {"location": where, "path": folder, "items": items, "truncated": len(items) >= limit}

    @mcp.tool()
    async def clinical_get_item(ctx: Context, path: str, location: str = "repository") -> dict[str, Any]:
        """Show the properties of a Clinical Repository or Workspace item.

        For a repository item this includes your permissions on it (such as
        READ, WRITE, DELETE) and your versioning privileges, which explain why
        an action is refused: without WRITE on a file you cannot check it out.
        ``null`` means the server did not say; it still decides each action.

        Args:
            path: The item's path.
            location: ``repository`` or ``workspace``.
        """
        async with viya_session("clinical_get_item", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                where = ch.check_location(location)
                target = ch.normalize_path(path)
                user = await _user_id(client)
                if where == ch.WORKSPACE_LOCATION:
                    item = await ch.workspace_item(client, target)
                    if item is None:
                        ch.fail(f"Nothing at {target!r} in your Clinical Workspace.", "not_found")
                    return {"location": where, **ch.workspace_item_summary(item, user)}
                found = await ch.resolve_repo_item(client, target)
                item, _ = await ch.repo_item(client, found["id"])
                access = await ch.item_access(client, found["id"])
            except ch.ClinicalError as exc:
                return exc.result()
            out = {"location": where, **ch.repo_item_summary(item, user)}
            sync = item.get("synchronizationInfo") or {}
            if sync:
                out["workspace"] = {k: v for k, v in sync.items() if k in ch.SYNC_FIELDS and v not in (None, "")}
            out["your_permissions"] = access["permissions"]
            out["your_versioning_privileges"] = access["privileges"]
            return out

    @mcp.tool()
    async def clinical_read_file(
        ctx: Context, path: str, location: str = "repository", version: str | None = None
    ) -> Any:
        """Read a file from the Clinical Repository (any version) or your Clinical Workspace.

        Text files (programs, logs, listings, ``.cjob``, ``.cmnf``) come back as
        text; other files as an embedded file. To work with a ``.sas7bdat`` or
        ``.xpt`` dataset, ``clinical_load_dataset`` loads it into SAS instead.

        Args:
            path: The file's path.
            location: ``repository`` or ``workspace``.
            version: A repository version such as ``1.2`` (see clinical_list_versions);
                the latest when omitted.
        """
        async with viya_session("clinical_read_file", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                where = ch.check_location(location)
                target = ch.normalize_path(path)
                data = await read_bytes(client, target, where, version)
            except ch.ClinicalError as exc:
                return exc.result()
            name = ch.split_path(target)[1]
            text = None if name.lower().endswith(_DATASET_EXTENSIONS) else ch.decode_text(data)
            if text is None:
                return _binary_result(data, name, target, version)
            if len(data) > config.MAX_EXPORT_INLINE_BYTES:
                return _binary_result(data, name, target, version)
            return {
                "location": where,
                "path": target,
                "version": version or "latest",
                "size": len(data),
                "content": text,
            }

    @mcp.tool()
    async def clinical_download_items(ctx: Context, paths: PathList, location: str = "repository") -> Any:
        """Download files or folders from the Clinical Repository or Workspace as one zip file.

        Args:
            paths: Paths of the files and folders to include.
            location: ``repository`` or ``workspace``.
        """
        async with viya_session("clinical_download_items", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                where = ch.check_location(location)
                targets = _paths(paths)
                if where == ch.REPOSITORY:
                    ids = [(await ch.resolve_repo_item(client, p))["id"] for p in targets]
                    resp = await client.post(
                        f"{config.VIYA_ENDPOINT}{ch.REPO}/repository/files/content",
                        json={"filter": in_filter("id", ids)},
                    )
                    raise_for_viya_status(resp)
                    data = resp.content
                else:
                    status_id = await ch.start_batch(client, where, "DOWNLOAD", targets)
                    resp = await client.post(
                        f"{config.VIYA_ENDPOINT}{ch.WORKSPACE}/items/content",
                        json={"paths": targets, "actionStatusId": status_id},
                    )
                    raise_for_viya_status(resp)
                    data = resp.content
                    outcome = ch.action_result(await ch.wait_action(client, status_id))
                    if outcome["status"] == "failed":
                        return outcome
            except ch.ClinicalError as exc:
                return exc.result()
            if not data.startswith(b"PK\x03\x04"):
                # One workspace file comes back as itself, not zipped.
                return _binary_result(data, ch.split_path(targets[0])[1], targets[0], None)
            return _binary_result(data, "clinical-download.zip", ", ".join(targets), None)

    @mcp.tool()
    async def clinical_list_versions(ctx: Context, path: str) -> dict[str, Any]:
        """List the versions of a versioned Clinical Repository file, newest first.

        Each version shows who created it and when, its comment, and its size.
        ``clinical_read_file`` and ``clinical_compare_files`` take a version.

        Args:
            path: The repository file's path.
        """
        async with viya_session("clinical_list_versions", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                target = ch.normalize_path(path)
                item = await ch.resolve_repo_item(client, target)
                if not item.get("versioned"):
                    return {
                        "path": target,
                        "versioned": False,
                        "versions": [],
                        "message": "This file is not versioned; clinical_set_versioning turns versioning on.",
                    }
                versions = await ch.collect(
                    client, f"{ch.REPO_ITEMS}/{quote(item['id'], safe='')}/versions", params={"limit": 200}
                )
            except ch.ClinicalError as exc:
                return exc.result()
            versions.sort(key=lambda v: ch.version_key(v.get("fileVersion")), reverse=True)
            fields = (
                "fileVersion",
                "comment",
                "createdBy",
                "createdByDisplayName",
                "creationTimeStamp",
                "latest",
                "signingStatus",
            )
            out = []
            for v in versions:
                entry = {k: v[k] for k in fields if v.get(k) not in (None, "")}
                size = v.get("size") or v.get("fileSize")
                if size is not None:
                    entry["size"] = size
                out.append(entry)
            return {"path": target, "versioned": True, "versions": out}

    @mcp.tool()
    async def clinical_compare_files(
        ctx: Context,
        left_path: str,
        right_path: str | None = None,
        left_location: str = "repository",
        right_location: str = "workspace",
        left_version: str | None = None,
        right_version: str | None = None,
        context_lines: int = 3,
    ) -> dict[str, Any]:
        """Compare two text files or versions from the Clinical Repository and Workspace (a unified diff).

        By default it compares the latest repository version of ``left_path``
        with your workspace copy of the same file. Compare two versions with
        ``right_location='repository'`` and both versions set.

        Args:
            left_path: The first file.
            right_path: The second file; the same path as ``left_path`` when omitted.
            left_location: ``repository`` or ``workspace`` (default repository).
            right_location: ``repository`` or ``workspace`` (default workspace).
            left_version: A repository version of the first file; latest when omitted.
            right_version: A repository version of the second file; latest when omitted.
            context_lines: Unchanged lines shown around each change (default 3).
        """
        async with viya_session("clinical_compare_files", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                sides = []
                for path, location, version in (
                    (left_path, left_location, left_version),
                    (right_path or left_path, right_location, right_version),
                ):
                    where = ch.check_location(location)
                    target = ch.normalize_path(path)
                    text = ch.decode_text(await read_bytes(client, target, where, version))
                    label = f"{where}:{target}" + (f"@{version}" if version else "")
                    if text is None:
                        ch.fail(f"{label} is not a text file; only text files can be compared.")
                    sides.append((label, text))
            except ch.ClinicalError as exc:
                return exc.result()
            (left_label, left), (right_label, right) = sides
            diff = list(
                difflib.unified_diff(
                    left.splitlines(),
                    right.splitlines(),
                    left_label,
                    right_label,
                    n=max(0, int(context_lines)),
                    lineterm="",
                )
            )
            out: dict[str, Any] = {"left": left_label, "right": right_label, "identical": not diff}
            if diff:
                out["diff"] = "\n".join(diff[:_MAX_DIFF_LINES])
                if len(diff) > _MAX_DIFF_LINES:
                    out["truncated"] = True
            return out

    # --- search, audit history, signatures ----------------------------------------------

    @mcp.tool()
    async def clinical_search_repository(
        ctx: Context,
        name_contains: str | None = None,
        under_path: str | None = None,
        type_id: str | None = None,
        primary_type: str | None = None,
        checked_out: bool | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search the whole Clinical Repository you can see.

        Combine any of the criteria; all must match.

        Args:
            name_contains: Text the item's name contains, e.g. ``adsl``.
            under_path: Only items whose path starts with this, e.g. ``/Org/Study1/data``.
            type_id: Item type, e.g. ``sasprogram``, ``sasdataset``, ``clinicalJob``,
                ``clinicalJobManifest``, ``pdf``, ``folder``, ``study``.
            primary_type: ``FILE``, ``FOLDER`` or ``CONTEXT``.
            checked_out: Only checked-out (true) or not checked-out (false) files.
            limit: Maximum results (default 50).
        """
        parts = []
        if name_contains:
            parts.append(f"contains(name,'{filter_literal(name_contains)}')")
        if under_path:
            parts.append(f"startsWith(path,'{filter_literal(ch.normalize_path(under_path))}')")
        if type_id:
            parts.append(f"eq(typeId,'{filter_literal(type_id)}')")
        if primary_type:
            parts.append(f"eq(primaryType,'{filter_literal(primary_type.upper())}')")
        if checked_out is not None:
            parts.append(f"eq(checkedOut,{str(bool(checked_out)).lower()})")
        if not parts:
            return {"status": "refused", "message": "Give at least one search criterion."}
        rql = parts[0] if len(parts) == 1 else f"and({','.join(parts)})"
        async with viya_session("clinical_search_repository", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            limit = max(1, min(int(limit), 500))
            items = await ch.collect(
                client, f"{ch.REPO_ITEMS}", params={"filter": rql, "limit": min(limit, 200)}, limit=limit
            )
            user = await _user_id(client)
            return {
                "filter": rql,
                "items": [ch.repo_item_summary(i, user) for i in items],
                "truncated": len(items) >= limit,
            }

    @mcp.tool()
    async def clinical_search_audit_history(
        ctx: Context,
        path: str | None = None,
        name: str | None = None,
        under_path: str | None = None,
        action: str | None = None,
        user: str | None = None,
        since: str | None = None,
        until: str | None = None,
        include_details: bool = False,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search the Clinical Acceleration audit history, newest first.

        Every significant event is recorded: who acted on which item, when,
        and how. You see the events of your own actions; Clinical
        Administrators see everyone's.

        Args:
            path: Only events on this repository item (file or folder).
            name: Only events on items with exactly this name, e.g. ``adsl.sas7bdat``.
            under_path: Only events on items in this folder or below.
            action: The action, e.g. checkedIn, checkedOut, undoCheckedOut, created,
                updated, uploaded, downloaded, deleted, restored, renamed, moved,
                enabledVersioning, disabledVersioning, locked, unlocked, signed, ran.
            user: Only events by this user id.
            since: Only events at or after this time (ISO 8601, e.g. ``2026-11-01`` or ``2026-11-01T08:00:00Z``).
            until: Only events before this time.
            include_details: Also return each event's details (version, client used,
                old and new values). One request per event, so keep ``limit`` small.
            limit: Maximum events (default 50).
        """
        async with viya_session("clinical_search_audit_history", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                parts = []
                if path:
                    item = await ch.resolve_repo_item(client, ch.normalize_path(path))
                    parts.append(f"eq(sourceId,'{filter_literal(item['id'])}')")
                if name:
                    parts.append(f"eq(sourceName,'{filter_literal(name)}')")
                if under_path:
                    parts.append(f"startsWith(sourceLocation,'{filter_literal(ch.normalize_path(under_path))}')")
                if action:
                    parts.append(f"eq(action,'{filter_literal(action)}')")
                if user:
                    parts.append(f"eq(userId,'{filter_literal(user)}')")
                if since:
                    parts.append(f"ge(timestamp,'{filter_literal(_timestamp(since))}')")
                if until:
                    parts.append(f"lt(timestamp,'{filter_literal(_timestamp(until))}')")
                params: dict[str, Any] = {"limit": min(max(1, int(limit)), 200)}
                if parts:
                    params["filter"] = parts[0] if len(parts) == 1 else f"and({','.join(parts)})"
                limit = max(1, min(int(limit), 1000))
                entries = await ch.collect(client, f"{ch.REPO}/audit/entries", params=params, limit=limit)
            except ch.ClinicalError as exc:
                return exc.result()
            fields = (
                "timestamp",
                "userId",
                "userDisplayName",
                "action",
                "actionDisplayName",
                "sourceName",
                "sourceLocation",
                "sourceTypeId",
                "sourcePrimaryType",
                "mode",
            )
            out = []
            for entry in entries:
                row = {k: entry[k] for k in fields if entry.get(k) not in (None, "")}
                if include_details:
                    detail = await client.get(
                        f"{config.VIYA_ENDPOINT}{ch.REPO}/audit/entries/{quote(entry['id'], safe='')}",
                        headers={"Accept": "application/json"},
                    )
                    if detail.status_code == 200:
                        row["details"] = [
                            {k: d[k] for k in ("attributeName", "oldValue", "newValue") if d.get(k) not in (None, "")}
                            for d in detail.json().get("auditEntryDetails") or []
                        ]
                out.append(row)
            return {"filter": params.get("filter"), "entries": out, "truncated": len(out) >= limit}

    @mcp.tool()
    async def clinical_get_signature_history(ctx: Context, path: str) -> dict[str, Any]:
        """List the electronic signatures on a Clinical Repository file: who signed, why, in which role, when.

        Signatures are made in SAS Clinical Acceleration itself with the
        signer's authenticator PIN; these tools only read them. For a
        versioned file, the history covers the latest version, and each
        version lists whether it is signed.

        Args:
            path: The repository file's path.
        """
        async with viya_session("clinical_get_signature_history", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                target = ch.normalize_path(path)
                item = await ch.resolve_repo_item(client, target)
                signatures = await ch.collect(
                    client, f"{ch.REPO_ITEMS}/{quote(item['id'], safe='')}/signatures", params={"limit": 200}
                )
            except ch.ClinicalError as exc:
                return exc.result()
            return {
                "path": target,
                "signing_status": item.get("signingStatus"),
                "file_version": item.get("fileVersion"),
                "signatures": [{k: v for k, v in s.items() if k not in ("links", "version")} for s in signatures],
            }

    # --- workspace actions --------------------------------------------------------------

    @mcp.tool()
    async def clinical_copy_to_workspace(
        ctx: Context, paths: PathList, mode: str = "latest", version: str | None = None
    ) -> dict[str, Any]:
        """Copy Clinical Repository files or folders into your Clinical Workspace without checking them out.

        A copy can be read, run and edited, but a changed copy cannot be checked
        in over the repository file: check the file out for that
        (``clinical_check_out`` with ``with_content=false`` keeps the copy).

        Args:
            paths: Repository paths of files or folders.
            mode: ``latest`` (latest version of each file; a folder copies its files),
                ``folder_structure`` (the folders and subfolders, no files) or
                ``version`` (the given ``version`` of each file).
            version: The version to copy, with ``mode='version'``.
        """
        async with viya_session("clinical_copy_to_workspace", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                targets = _paths(paths)
                if mode == "latest":
                    action, body = "COPY_LATEST_VERSION", ch.batch_body("COPY_LATEST_VERSION", targets)
                elif mode == "folder_structure":
                    action, body = "COPY_FOLDER_STRUCTURE", ch.batch_body("COPY_FOLDER_STRUCTURE", targets)
                elif mode == "version":
                    fixed = ch.validate_version(version, allow_bump=False)
                    if not fixed:
                        ch.fail("mode='version' needs a version such as 1.0.")
                    action, body = (
                        "COPY_SPECIFIC_VERSION",
                        ch.batch_body("COPY_SPECIFIC_VERSION", targets, file_version=fixed),
                    )
                else:
                    ch.fail("mode must be 'latest', 'folder_structure' or 'version'.")
                return await ch.run_batch(client, ch.WORKSPACE_LOCATION, action, body)
            except ch.ClinicalError as exc:
                return exc.result()

    @mcp.tool()
    async def clinical_check_out(ctx: Context, paths: PathList, with_content: bool = True) -> dict[str, Any]:
        """Check out Clinical Repository files to your workspace, reserving them for you to change.

        Only you can then check them in, as new versions. A checked-out file
        needs WRITE permission (clinical_get_item shows yours).

        Args:
            paths: Repository paths of files (or folders, for all their files).
            with_content: Copy the content to the workspace (default). ``false``
                reserves the files and keeps whatever is in your workspace, which
                is how a changed copy made with clinical_copy_to_workspace becomes
                eligible for check-in.
        """
        async with viya_session("clinical_check_out", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                body = ch.batch_body("REPOSITORY_CHECK_OUT", _paths(paths), copy_content=with_content)
                return await ch.run_batch(client, ch.WORKSPACE_LOCATION, "REPOSITORY_CHECK_OUT", body)
            except ch.ClinicalError as exc:
                return exc.result()

    @mcp.tool()
    async def clinical_undo_check_out(ctx: Context, paths: PathList) -> dict[str, Any]:
        """Give up your check-out of Clinical Repository files, discarding the changes you have not checked in.

        Args:
            paths: Paths of files you checked out (or folders, for all of them).
        """
        async with viya_session("clinical_undo_check_out", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                targets = _paths(paths)
                return await ch.run_batch(
                    client, ch.WORKSPACE_LOCATION, "UNDO_CHECKOUT", ch.batch_body("UNDO_CHECKOUT", targets)
                )
            except ch.ClinicalError as exc:
                return exc.result()

    @mcp.tool()
    async def clinical_check_in(
        ctx: Context,
        paths: PathList,
        comment: str,
        bump: str = "minor",
        new_file_version: str | None = None,
    ) -> dict[str, Any]:
        """Check in workspace files to the Clinical Repository as new versions (or new files).

        Checks in what is eligible below ``paths`` and reports what was left out
        and why: a copy you did not check out, or a file someone else checked
        out. The comment goes into the audit record.

        Args:
            paths: Workspace paths of files or folders.
            comment: Why the change was made (required; up to 1024 characters).
            bump: For versioned files: ``minor`` (2.3 → 2.4, default) or ``major`` (2.3 → 3.0).
            new_file_version: First version for new or unversioned files, e.g. ``1.0``;
                they stay unversioned when omitted.
        """
        async with viya_session("clinical_check_in", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                text = ch.validate_comment(comment)
                if not text:
                    ch.fail("A check-in needs a comment: it is the audit record of the change.")
                if bump not in ("minor", "major"):
                    ch.fail("bump must be 'minor' or 'major'.")
                first = ch.validate_version(new_file_version, allow_bump=False)
                targets = _paths(paths)
                candidates = await ch.check_in_candidates(client, targets)
                body, included, skipped = ch.plan_check_in(
                    candidates, targets, await _user_id(client), bump=bump, new_file_version=first, comment=text
                )
                for entry in skipped:
                    entry["explanation"] = ch.SKIP_REASONS.get(entry["reason"], "")
                if not body["fileSpecifications"]:
                    return {"status": "nothing_to_check_in", "checked_in": [], "skipped": skipped}
                result = await ch.run_batch(client, ch.WORKSPACE_LOCATION, "CHECK_IN", body)
            except ch.ClinicalError as exc:
                return exc.result()
            if result["status"] == "failed" and ch.looks_like_missing_privilege(result):
                result["hint"] = ch.VERSIONING_PRIVILEGE_HINT
            return {**result, "checked_in": included, "skipped": skipped}

    @mcp.tool()
    async def clinical_write_file(
        ctx: Context,
        path: str,
        content: str | None = None,
        file_path: str | None = None,
        url: str | None = None,
    ) -> dict[str, Any]:
        """Create or overwrite a file in your Clinical Workspace (a program, a .cjob, data).

        This is how an edited workspace file is saved; the repository changes
        only when the file is checked in. Missing workspace folders on the way
        are created. Give exactly one source.

        Args:
            path: Workspace path of the file, e.g. ``/Org/Study1/programs/adsl.sas``.
            content: The file's text.
            file_path: A file the server reads from its own disk instead.
            url: An http(s) URL the server downloads the file from instead.
        """
        async with viya_session("clinical_write_file", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                target = ch.normalize_path(path)
                name = ch.split_path(target)[1]
                ch.validate_file_name(name)
                data, source_error = await _source(content, file_path, url)
                if source_error:
                    return source_error
                assert data is not None
                existed = await ch.workspace_item(client, target) is not None
                created = [] if existed else await ch.ensure_workspace_folder(client, ch.split_path(target)[0])
                await ch.put_workspace_file(client, target, name, data)
            except ch.ClinicalError as exc:
                return exc.result()
            out = {"status": "updated" if existed else "created", "path": target, "size": len(data)}
            if created:
                out["created_folders"] = created
            return out

    # --- either location ----------------------------------------------------------------

    @mcp.tool()
    async def clinical_create_folder(ctx: Context, path: str, location: str = "workspace") -> dict[str, Any]:
        """Create a folder in your Clinical Workspace or in the Clinical Repository.

        The parent folder must exist.

        Args:
            path: Path of the new folder.
            location: ``workspace`` (default) or ``repository``.
        """
        async with viya_session("clinical_create_folder", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                where = ch.check_location(location)
                target = ch.normalize_path(path)
                parent, name = ch.split_path(target)
                if not name:
                    ch.fail("Give the path of the new folder, not the root.")
                ch.validate_folder_name(name)
                if where == ch.REPOSITORY:
                    container = await ch.resolve_repo_item(client, parent)
                    resp = await client.post(
                        f"{config.VIYA_ENDPOINT}{ch.REPO_ITEMS}/{quote(container['id'], safe='')}/children",
                        params={"name": name, "type": "FOLDER"},
                        headers={"Accept": ch.ACCEPT_ITEM},
                    )
                else:
                    suffix = "" if parent == "/" else ch.url_path(parent)
                    resp = await client.post(
                        f"{config.VIYA_ENDPOINT}{ch.WORKSPACE}/folders{suffix}",
                        params={"name": name},
                        headers={"Accept": "application/json"},
                    )
                if resp.status_code == 404:
                    ch.fail(f"The parent folder {parent!r} does not exist in the {where}.", "not_found")
                raise_for_viya_status(resp)
            except ch.ClinicalError as exc:
                return exc.result()
            return {"status": "created", "location": where, "path": target}

    @mcp.tool()
    async def clinical_rename_item(
        ctx: Context, path: str, new_name: str, location: str = "workspace"
    ) -> dict[str, Any]:
        """Rename a file or folder in your Clinical Workspace or in the Clinical Repository.

        Repository contexts (organizations, studies) are managed by
        administrators and cannot be renamed.

        Args:
            path: The item's current path.
            new_name: The new name only (not a path).
            location: ``workspace`` (default) or ``repository``.
        """
        async with viya_session("clinical_rename_item", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                where = ch.check_location(location)
                target = ch.normalize_path(path)
                if "/" in new_name or not new_name.strip():
                    ch.fail("new_name is the new name only, without a path.")
                parent = ch.split_path(target)[0]
                if where == ch.REPOSITORY:
                    found = await ch.resolve_repo_item(client, target)
                    if found.get("primaryType") == "CONTEXT":
                        ch.fail(f"{target} is a context; administrators manage contexts.")
                    (ch.validate_file_name if found.get("primaryType") != "FOLDER" else ch.validate_folder_name)(
                        new_name
                    )
                    # A fresh ETag right before the PATCH: the item may have changed since it was listed.
                    _, etag = await ch.repo_item(client, found["id"])
                    resp = await client.patch(
                        f"{config.VIYA_ENDPOINT}{ch.REPO_ITEMS}/{quote(found['id'], safe='')}",
                        json={"name": new_name},
                        headers={"Accept": ch.ACCEPT_ITEM, "If-Match": etag or "*"},
                    )
                else:
                    item = await ch.workspace_item(client, target)
                    if item is None:
                        ch.fail(f"Nothing at {target!r} in your Clinical Workspace.", "not_found")
                    (ch.validate_file_name if item.get("type") == "FILE" else ch.validate_folder_name)(new_name)
                    # The body is a bare JSON string literal.
                    resp = await client.post(
                        f"{config.VIYA_ENDPOINT}{ch.WORKSPACE}/items{ch.url_path(target)}",
                        params={"action": "RENAME"},
                        content=f'"{new_name}"'.encode(),
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                    )
                raise_for_viya_status(resp)
            except ch.ClinicalError as exc:
                return exc.result()
            return {"status": "renamed", "location": where, "path": posixpath.join(parent, new_name)}

    @mcp.tool()
    async def clinical_delete_items(ctx: Context, paths: PathList, location: str = "workspace") -> dict[str, Any]:
        """Delete files or folders from your Clinical Workspace or the Clinical Repository.

        In the workspace this removes only your copies; the repository is not
        changed. In the repository, items go to the recycle bin, from which the
        Manage Data Repository application of SAS Clinical Acceleration restores
        them. Contexts (organizations, studies) cannot be deleted.

        Args:
            paths: Paths of the items to delete.
            location: ``workspace`` (default) or ``repository``.
        """
        async with viya_session("clinical_delete_items", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                where = ch.check_location(location)
                targets = _paths(paths)
                if where == ch.WORKSPACE_LOCATION:
                    return await ch.run_batch(client, where, "DELETE", ch.batch_body("DELETE", targets))
                results = []
                for target in targets:
                    item = await ch.resolve_repo_item(client, target)
                    if item.get("primaryType") == "CONTEXT":
                        results.append({"path": target, "status": "refused", "message": "Contexts cannot be deleted."})
                        continue
                    resp = await client.delete(
                        f"{config.VIYA_ENDPOINT}{ch.REPO_ITEMS}/{quote(item['id'], safe='')}",
                        params={"permanent": "false"},
                    )
                    if resp.status_code >= 400:
                        try:
                            raise_for_viya_status(resp)
                        except httpx.HTTPStatusError as exc:
                            results.append({"path": target, "status": "failed", "message": str(exc)})
                            continue
                    results.append({"path": target, "status": "moved_to_recycle_bin"})
            except ch.ClinicalError as exc:
                return exc.result()
            failed = any(r["status"] != "moved_to_recycle_bin" for r in results)
            return {"status": "completed_with_errors" if failed else "completed", "items": results}

    @mcp.tool()
    async def clinical_upload(
        ctx: Context,
        folder: str,
        name: str | None = None,
        location: str = "workspace",
        content: str | None = None,
        file_path: str | None = None,
        url: str | None = None,
        expand: bool = False,
        version: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Upload a file into a Clinical Workspace or Clinical Repository folder, or unpack a zip there.

        Give exactly one source. An upload straight to the repository creates
        an unversioned file unless ``version`` is given, which needs a
        versioning privilege; the usual route for changing repository files is
        the workspace and clinical_check_in.

        Args:
            folder: The folder to upload into (it must exist).
            name: The file's name; taken from ``file_path`` or ``url`` when omitted.
            location: ``workspace`` (default) or ``repository``.
            content: The file's text.
            file_path: A file the server reads from its own disk instead.
            url: An http(s) URL the server downloads the file from instead.
            expand: Unpack a zip file into ``folder`` instead of storing the zip.
            version: Repository only: a version such as ``1.0`` for a new file, or
                ``MAJOR`` / ``MINOR`` for a file that is already versioned.
            comment: Repository only: a comment for the audit record.
        """
        async with viya_session("clinical_upload", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                where = ch.check_location(location)
                parent = ch.normalize_path(folder)
                file_name = name or posixpath.basename((file_path or url or "").split("?", 1)[0])
                if not file_name:
                    ch.fail("Give the file's name.")
                ch.validate_file_name(file_name)
                data, source_error = await _source(content, file_path, url)
                if source_error:
                    return source_error
                assert data is not None
                if where == ch.REPOSITORY:
                    container = await ch.resolve_repo_item(client, parent)
                    params: dict[str, Any] = {"name": file_name, "expand": str(expand).lower()}
                    if text := ch.validate_comment(comment):
                        params["comment"] = text
                    if fixed := ch.validate_version(version):
                        params["fileVersion"] = fixed
                    resp = await client.put(
                        f"{config.VIYA_ENDPOINT}{ch.REPO_ITEMS}/{quote(container['id'], safe='')}/content",
                        params=params,
                        files={"file": (file_name, data, "application/octet-stream")},
                        headers={"Accept": ch.ACCEPT_ITEM},
                    )
                    if resp.status_code >= 500 and "fileVersion" in params:
                        return {
                            "status": "failed",
                            "path": posixpath.join(parent, file_name),
                            "message": ch.VERSIONING_PRIVILEGE_HINT,
                        }
                    raise_for_viya_status(resp)
                    return {
                        "status": "completed",
                        "location": where,
                        "path": posixpath.join(parent, file_name),
                        "size": len(data),
                    }
                if version or comment:
                    ch.fail("version and comment apply to repository uploads; check in to version a workspace file.")
                action = "UPLOAD_EXPAND" if expand else "UPLOAD"
                target = posixpath.join(parent, file_name)
                status_id = await ch.start_batch(client, where, action, ch.batch_body(action, [target]))
                await ch.put_workspace_file(
                    client, parent if expand else target, file_name, data, expand=expand, action_status_id=status_id
                )
                result = ch.action_result(await ch.wait_action(client, status_id))
            except ch.ClinicalError as exc:
                return exc.result()
            return {**result, "location": where, "path": parent if expand else target, "size": len(data)}

    @mcp.tool()
    async def clinical_set_versioning(
        ctx: Context, path: str, enabled: bool, comment: str, version: str = "1.0"
    ) -> dict[str, Any]:
        """Turn versioning on or off for a Clinical Repository file.

        Needs a versioning privilege (clinical_get_item shows yours). Turning
        it off keeps only the current content.

        Args:
            path: The repository file's path.
            enabled: ``true`` to enable versioning, ``false`` to disable it.
            comment: Why (required; it goes into the audit record).
            version: The first version when enabling (default ``1.0``).
        """
        async with viya_session("clinical_set_versioning", ctx) as client:
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                target = ch.normalize_path(path)
                text = ch.validate_comment(comment)
                if not text:
                    ch.fail("A versioning change needs a comment for the audit record.")
                item = await ch.resolve_repo_item(client, target)
                if item.get("primaryType") not in ("FILE", "JOB_FILE"):
                    ch.fail(f"{target} is not a file; only files are versioned.")
                if bool(item.get("versioned")) == enabled:
                    return {"status": "unchanged", "path": target, "versioned": enabled}
                privileges = (await ch.item_access(client, item["id"]))["privileges"]
                if privileges is not None and not {"PRIVILEGE_ENABLE_VERSIONING", "PRIVILEGE_MANAGE_VERSIONING"} & set(
                    privileges
                ):
                    ch.fail(
                        "You have no versioning privilege on this file; ask a SAS Clinical Acceleration administrator."
                    )
                if enabled:
                    first = ch.validate_version(version, allow_bump=False) or "1.0"
                    body = ch.batch_body("ENABLE_VERSIONING", [target], comment=text, file_version=first)
                    action = "ENABLE_VERSIONING"
                else:
                    body, action = ch.batch_body("DISABLE_VERSIONING", [target], comment=text), "DISABLE_VERSIONING"
                result = await ch.run_batch(client, ch.REPOSITORY, action, body)
            except ch.ClinicalError as exc:
                return exc.result()
            if result["status"] == "failed" and ch.looks_like_missing_privilege(result):
                result["hint"] = ch.VERSIONING_PRIVILEGE_HINT
            return {**result, "path": target, "versioned": enabled}

    # --- datasets -----------------------------------------------------------------------

    @mcp.tool()
    async def clinical_load_dataset(
        ctx: Context,
        path: str,
        location: str = "repository",
        libref: str | None = None,
        compute_context_name: str = config.CONTEXT_NAME,
    ) -> dict[str, Any]:
        """Load a Clinical Repository or Workspace dataset (.sas7bdat or .xpt) into your SAS compute session.

        The file is copied into a library under WORK of the reusable compute
        session that ``execute_sas_code`` and the data tools use, so it can be
        queried there right away. A transport file (.xpt, XPORT or CPORT) is
        unpacked into its datasets.

        Args:
            path: The dataset file's path.
            location: ``repository`` (latest version) or ``workspace``.
            libref: Library name to use (up to 8 characters); derived from the file name when omitted.
            compute_context_name: Compute context of the session.
        """
        target = ch.normalize_path(path)
        name = ch.split_path(target)[1]
        stem, ext = posixpath.splitext(name.lower())
        if ext not in _DATASET_EXTENSIONS:
            return {"status": "refused", "message": "Only .sas7bdat and .xpt files can be loaded as datasets."}
        lib = (libref or ("_ca" + "".join(c for c in stem if c.isalnum())[:5])).lower()
        if (
            not lib
            or len(lib) > 8
            or not (lib[0].isalpha() or lib[0] == "_")
            or not all(c.isalnum() or c == "_" for c in lib)
        ):
            return {"status": "refused", "message": f"{lib!r} is not a valid libref (up to 8 letters, digits or _)."}
        async with compute_tool_session("clinical_load_dataset", ctx, compute_context_name) as (client, session_id):
            if err := await ch.ensure_service(client, "repository"):
                return err
            try:
                data = await read_bytes(client, target, ch.check_location(location), None)
                if ext == ".xpt":
                    fmt = ch.detect_transport_format(data)
                    if fmt is None:
                        ch.fail(f"{name} is not a SAS transport file (XPORT or CPORT).")
                    programs = ch.upload_to_work_code("_fxptsrc", f"{lib}.xpt", data)
                    programs.append(ch.transport_load_code(fmt, "_fxptsrc", f"{lib}.xpt", lib))
                else:
                    programs = ch.upload_to_work_code(lib, f"{stem}.sas7bdat", data)
                    programs.append(f"libname {lib} {ch.work_path_literal(lib)};")
                programs.append(
                    "%let _ca_members=;\n"
                    "proc sql noprint; select memname into :_ca_members separated by ' ' "
                    f"from dictionary.tables where libname='{lib.upper()}'; quit;\n"
                    "%put _CA_MEMBERS=&_ca_members;"
                )
                log = ""
                for program in programs:
                    job_id = await submit_job(client, session_id, program)
                    _, log, _ = await wait_job(client, session_id, job_id, poll=0.5)
                    errors = ch.find_sas_errors(log)
                    if errors:
                        return {"status": "failed", "path": target, "errors": errors[:10]}
            except ch.ClinicalError as exc:
                return exc.result()
            members = next(
                (line.split("=", 1)[1].split() for line in log.splitlines() if line.startswith("_CA_MEMBERS=")), []
            )
            return {
                "status": "loaded",
                "path": target,
                "libref": lib.upper(),
                "datasets": [f"{lib.upper()}.{m}" for m in members],
                "compute_context_name": compute_context_name,
            }


def _timestamp(value: str) -> str:
    """A date or timestamp as the ISO timestamp the audit filter compares with."""
    text = value.strip()
    return f"{text}T00:00:00Z" if len(text) == 10 else text


async def _source(
    content: str | None, file_path: str | None, url: str | None
) -> tuple[bytes | None, dict[str, Any] | None]:
    """The bytes of exactly one of ``content`` / ``file_path`` / ``url``."""
    given = [s for s in (content, file_path, url) if s is not None]
    if len(given) != 1:
        return None, {"status": "invalid_source", "message": "Give exactly one of content, file_path or url."}
    if content is not None:
        data = content.encode("utf-8")
        if len(data) > config.MAX_UPLOAD_BYTES:
            return None, {"status": "source_too_large", "source": "content", "limit_bytes": config.MAX_UPLOAD_BYTES}
        return data, None
    return await resolve_source_bytes(file_path, url)
