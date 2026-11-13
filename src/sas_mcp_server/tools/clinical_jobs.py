# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 13 — Clinical Jobs (SAS Clinical Job Management) tools.

A clinical job (a ``.cjob`` file) lists SAS programs to run in order and the
parameters they take. SAS Clinical Acceleration runs it on the server and
writes a manifest (``.cmnf``) next to it: which programs ran and how they
ended, the parameter values, and every file the run read and wrote, with its
version. The manifest is the audit record of the run.

These tools create, run, test and schedule jobs and read their runs and
manifests, as the Positron SAS extension's Clinical Jobs views do. They use the
repository and workspace of Tier 12 for the files themselves.
"""

import json
import posixpath
from collections.abc import Awaitable, Callable
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from fastmcp import Context, FastMCP
from pydantic import BeforeValidator

from .. import config
from ..helpers import clinical_helpers as ch
from ..user_log import resolve_user_id
from ..viya_client import filter_literal, raise_for_viya_status
from ._common import coerce_json_dict, make_session_helpers

JsonDict = Annotated[dict[str, Any], BeforeValidator(coerce_json_dict)]
_JSON = {"Accept": "application/json"}
_MAX_WAIT = 600
_MAX_OUTPUT_CHARS = 200_000


async def _user_id(client: httpx.AsyncClient) -> str | None:
    auth = client.headers.get("Authorization", "")
    return await resolve_user_id(auth.removeprefix("Bearer ").strip() or None)


def _relative(job_path: str, value: str) -> str:
    """An absolute path in a job file as the job-relative path the service stores (``./x.sas``)."""
    if not value.startswith("/"):
        return value
    rel = posixpath.relpath(value, ch.split_path(job_path)[0])
    return rel if rel.startswith("../") else f"./{rel}"


def _job_file(job_path: str, job: dict[str, Any]) -> dict[str, Any]:
    """``job`` as the ``.cjob`` file stores it: program and output paths relative to the job."""
    out = {k: v for k, v in job.items() if k not in ("resource", "links", "owner", "ownerDisplayName", "version")}
    programs = []
    for program in job.get("programs") or []:
        entry = {"path": _relative(job_path, str(program.get("path") or "")), "version": program.get("version")}
        programs.append(entry)
    out["programs"] = programs
    for key in ("logsLocation", "resultsLocation", "manifestLocation", "outputLocation"):
        if isinstance(out.get(key), str):
            out[key] = _relative(job_path, out[key])
    return out


def register(mcp: FastMCP, get_token: Callable[[Context], Awaitable[str]]) -> None:
    """Register Tier 13 (Clinical Jobs) tools on *mcp*."""

    viya_session, _ = make_session_helpers(get_token)

    async def services(client: httpx.AsyncClient) -> dict[str, Any] | None:
        return await ch.ensure_service(client, "repository") or await ch.ensure_service(client, "jobs")

    async def load_job(
        client: httpx.AsyncClient, path: str, location: str, version: str | None = None
    ) -> dict[str, Any]:
        """A job definition: the workspace ``.cjob`` file, or the repository job as the service reads it."""
        if not path.lower().endswith(".cjob"):
            ch.fail(f"{path} is not a clinical job (.cjob).")
        if location == ch.WORKSPACE_LOCATION:
            if version:
                ch.fail("Versions exist only in the repository; leave version empty for a workspace job.")
            # There is no GET for workspace jobs: read the file itself.
            return ch.parse_json_bytes(await ch.read_workspace_content(client, path), path)
        item = await ch.resolve_repo_item(client, path)
        resp = await ch.get(
            client,
            f"{ch.JOBS}/repository/definitions/{quote(item['id'], safe='')}",
            params={"fileVersion": version} if version else None,
            missing=f"{path} has no version {version!r}." if version else f"{path} is not a clinical job.",
        )
        return {k: v for k, v in resp.json().items() if k != "links"}

    async def submission(client: httpx.AsyncClient, run_id: str) -> dict[str, Any]:
        resp = await ch.get(
            client,
            f"{ch.JOBS}/submissions/{quote(run_id, safe='')}",
            missing=f"No clinical job run {run_id!r}; clinical_list_job_runs lists them.",
        )
        return resp.json()

    async def run_state(client: httpx.AsyncClient, run_id: str) -> str:
        resp = await ch.get(client, f"{ch.JOBS}/submissions/{quote(run_id, safe='')}/state")
        try:
            value = resp.json()
        except ValueError:
            value = resp.text
        return str(value).strip().strip('"')

    async def wait_for(client: httpx.AsyncClient, run_id: str, wait_seconds: int) -> dict[str, Any]:
        """Wait up to ``wait_seconds`` for a run to end; return the run as a tool result."""
        deadline = ch._clock() + max(0, min(int(wait_seconds), _MAX_WAIT))
        while ch.run_category(await run_state(client, run_id)) == "running" and ch._clock() < deadline:
            await ch._sleep(ch._POLL_INTERVAL)
        run = ch.normalize_submission(await submission(client, run_id))
        if run["category"] == "running":
            run["message"] = "The run continues on the server; clinical_get_job_run with wait_seconds follows it."
        return run

    def check_mode(mode: str, location: str) -> None:
        if mode not in ch.RUN_MODES:
            ch.fail("mode must be 'run', 'test' or 'run_and_populate'.")
        if mode != "run" and location != ch.WORKSPACE_LOCATION:
            ch.fail(f"mode={mode!r} works only on workspace jobs; the service refuses it for repository jobs.")

    async def submit(
        client: httpx.AsyncClient,
        job_path: str,
        location: str,
        mode: str,
        relative_paths: bool,
        values: list[dict[str, Any]],
    ) -> str:
        check_mode(mode, location)
        body: dict[str, Any] = {"job": ch.job_reference(job_path, location)}
        if values:
            body["parameterValues"] = values
        if mode == "run_and_populate" and relative_paths:
            body["useRelativePaths"] = True  # in the body: as a query parameter it is ignored
        resp = await client.post(
            f"{config.VIYA_ENDPOINT}{ch.JOBS}/submissions",
            params={"action": ch.RUN_MODES[mode]},
            json=body,
            headers={**_JSON, "Content-Type": "application/json"},
        )
        raise_for_viya_status(resp)
        return resp.json()["id"]

    async def manifest_for(client: httpx.AsyncClient, run_id: str) -> dict[str, Any]:
        resp = await client.get(
            f"{config.VIYA_ENDPOINT}{ch.JOBS}/submissions/{quote(run_id, safe='')}/manifest", headers=_JSON
        )
        if resp.status_code == 404:
            ch.fail(
                f"The manifest of run {run_id} is no longer available: a later run of the same job overwrote "
                "its .cmnf file. The service keeps a run's manifest only while that file is unchanged.",
                "manifest_overwritten",
            )
        raise_for_viya_status(resp)
        return resp.json()

    async def read_resource(client: httpx.AsyncClient, resource: dict[str, Any]) -> bytes:
        """The content of a file a manifest names, at the version the run used."""
        path = resource.get("path") or ""
        if (resource.get("resourceLocation") or "").upper() == "REPOSITORY":
            item = {"id": resource.get("id"), "name": resource.get("name") or ch.split_path(path)[1], "path": path}
            if not item["id"]:
                item = await ch.resolve_repo_item(client, path)
            version = resource.get("fileVersion")
            return await ch.read_repo_content(client, item, None if version in (None, "", "-") else version)
        return await ch.read_workspace_content(client, path)

    # --- jobs ----------------------------------------------------------------------------

    @mcp.tool()
    async def clinical_get_job(
        ctx: Context, path: str, location: str = "workspace", version: str | None = None
    ) -> dict[str, Any]:
        """Read a clinical job (.cjob): its programs, parameters, output locations and repository-run settings.

        ``publishSetup`` lists the repository files and folders a repository run
        can read; ``publishCheckin`` says where its outputs are checked in.
        Program and output paths are relative to the job's folder.

        Args:
            path: The job's path.
            location: ``workspace`` (default) or ``repository``.
            version: A repository version of the job; latest when omitted.
        """
        async with viya_session("clinical_get_job", ctx) as client:
            if err := await services(client):
                return err
            try:
                where = ch.check_location(location)
                target = ch.normalize_path(path)
                job = await load_job(client, target, where, version)
            except ch.ClinicalError as exc:
                return exc.result()
            return {"location": where, "path": target, "problems": ch.validate_job(job), "job": job}

    @mcp.tool()
    async def clinical_save_job(ctx: Context, path: str, job: JsonDict) -> dict[str, Any]:
        """Create or replace a clinical job (.cjob) in your Clinical Workspace.

        Read the job with clinical_get_job first and change what you need; the
        saved job replaces the whole file. Fields:

        * ``description``
        * ``programs``: ``[{"path": "./adsl.sas", "version": null}]`` in run
          order; ``version`` empty means the latest.
        * ``parameters``: ``[{"name", "label", "type", "defaultValue"}]``; the
          programs see them as macro variables, so names are SAS names. Types:
          CHARACTER, CHARACTER_MASKED, NUMERIC, DATE, FILE (``{"path"}``),
          FOLDER (``{"path", "includeSubfolders"}``).
        * ``logsLocation``, ``resultsLocation``, ``manifestLocation``: ``./`` is the job's folder.
        * ``publishSetup``: the repository inputs of a repository run,
          ``{"files": [{"path"}], "folders": [{"path", "includeFiles", "includeSubfolders"}]}``.
        * ``publishCheckin``: where outputs are checked in, ``{"enableVersioningOnNewFiles",
          "fileVersion": "MAJOR"|"MINOR"|"", "locations": [{"path"}]}``.

        Args:
            path: Workspace path of the job, ending in ``.cjob``.
            job: The job definition.
        """
        async with viya_session("clinical_save_job", ctx) as client:
            if err := await services(client):
                return err
            try:
                target = ch.normalize_path(path)
                if not target.lower().endswith(".cjob"):
                    ch.fail("A clinical job's file name ends in .cjob.")
                problems = ch.validate_job(job)
                if problems:
                    return {"status": "invalid_job", "problems": problems}
                existed = await ch.workspace_item(client, target) is not None
                if not existed:
                    await ch.ensure_workspace_folder(client, ch.split_path(target)[0])
                    # The service creates the job (owner, default output locations) ...
                    resp = await client.post(
                        f"{config.VIYA_ENDPOINT}{ch.JOBS}/definitions",
                        params={"useRelativePaths": "true"},
                        json={
                            "resource": ch.job_reference(target, ch.WORKSPACE_LOCATION),
                            "programs": [],
                            "parameters": [],
                        },
                        headers={**_JSON, "Content-Type": "application/json"},
                    )
                    raise_for_viya_status(resp)
                    created = ch.parse_json_bytes(await ch.read_workspace_content(client, target), target)
                    job = {**created, **job}
                # ... and the job editor saves edits as the file itself, which this does too.
                content = json.dumps(_job_file(target, job), indent=2).encode("utf-8")
                await ch.put_workspace_file(client, target, ch.split_path(target)[1], content)
            except ch.ClinicalError as exc:
                return exc.result()
            return {"status": "updated" if existed else "created", "path": target}

    @mcp.tool()
    async def clinical_run_job(
        ctx: Context,
        job_path: str | None = None,
        program_path: str | None = None,
        rerun_of: str | None = None,
        location: str = "workspace",
        mode: str = "run",
        relative_paths: bool = True,
        parameter_values: JsonDict | None = None,
        wait_seconds: int = 60,
    ) -> dict[str, Any]:
        """Run a clinical job, run one SAS program as a clinical job, or repeat an earlier run.

        Every run writes a manifest, the audit record of what it read and
        wrote. Give exactly one of ``job_path``, ``program_path`` or ``rerun_of``.

        Modes (``test`` and ``run_and_populate`` work on workspace jobs only):

        * ``run``: a normal run. A repository job runs in a transient workspace
          that holds only its ``publishSetup`` inputs and checks its outputs in.
        * ``test``: runs a workspace job as if from the repository (programs and
          inputs read there) and writes only the log, listing and manifest;
          nothing is checked in. Use it before checking a job in.
        * ``run_and_populate``: runs the workspace job and, if it succeeds,
          rewrites its ``publishSetup`` from the files the programs read and
          wrote, and clears its check-in locations.

        Suggested order before a repository run: run_and_populate, check in the
        job, its programs and inputs (clinical_check_in), test, then run from
        the repository.

        Args:
            job_path: The ``.cjob`` to run.
            program_path: A workspace ``.sas`` program to run as a job: the job
                ``<program>.cjob`` next to it is reused, or created.
            rerun_of: A run id to repeat with the same parameter values and mode.
            location: Where ``job_path`` is: ``workspace`` (default) or ``repository``.
            mode: ``run`` (default), ``test`` or ``run_and_populate``.
            relative_paths: For run_and_populate: write ``./`` and ``../`` paths
                (default) instead of absolute ones.
            parameter_values: ``{"name": value}``; parameters left out use their
                defaults. DATE as ``YYYY-MM-DD``; FILE and FOLDER as a path (for
                FOLDER also ``{"path", "includeSubfolders"}``), in the repository for tests.
            wait_seconds: How long to wait for the run to end (default 60, at most
                600). The run continues on the server either way.
        """
        async with viya_session("clinical_run_job", ctx) as client:
            if err := await services(client):
                return err
            try:
                if sum(x is not None for x in (job_path, program_path, rerun_of)) != 1:
                    ch.fail("Give exactly one of job_path, program_path or rerun_of.")
                where = ch.check_location(location)
                created_job = False
                if rerun_of:
                    previous = await submission(client, rerun_of)
                    normalized = ch.normalize_submission(previous)
                    target, where = normalized["job_path"], normalized["location"]
                    mode, relative_paths = ch.rerun_options(previous.get("jobRunContext"))
                    values = [
                        {"name": p.get("name"), "type": p.get("type"), "value": p.get("value")}
                        for p in (await manifest_for(client, rerun_of)).get("parameters") or []
                    ]
                else:
                    check_mode(mode, where)
                    if program_path:
                        if where != ch.WORKSPACE_LOCATION or mode != "run":
                            ch.fail("program_path runs a workspace program with mode='run'.")
                        target, created_job = await program_job(client, ch.normalize_path(program_path))
                    else:
                        target = ch.normalize_path(job_path)
                    job = await load_job(client, target, where)
                    values = ch.encode_parameter_values(job.get("parameters") or [], parameter_values, target)
                run_id = await submit(client, target, where, mode, relative_paths, values)
                result = await wait_for(client, run_id, wait_seconds)
            except ch.ClinicalError as exc:
                return exc.result()
            if created_job:
                result["created_job"] = target
            return result

    async def program_job(client: httpx.AsyncClient, program: str) -> tuple[str, bool]:
        """The workspace job that runs ``program`` alone: ``<name>.cjob``, ``<name>-2.cjob`` …, created if needed."""
        if not program.lower().endswith(".sas"):
            ch.fail(f"{program} is not a SAS program (.sas).")
        if await ch.workspace_item(client, program) is None:
            ch.fail(f"{program} is not in your Clinical Workspace; copy or check it out first.", "not_found")
        stem = program[: -len(".sas")]
        for n in range(1, 51):
            candidate = f"{stem}.cjob" if n == 1 else f"{stem}-{n}.cjob"
            try:
                existing = ch.parse_json_bytes(await ch.read_workspace_content(client, candidate), candidate)
            except ch.ClinicalError as exc:
                if exc.status != "not_found":
                    continue  # not a job file: try the next name
                resp = await client.post(
                    f"{config.VIYA_ENDPOINT}{ch.JOBS}/definitions",
                    params={"useRelativePaths": "true"},
                    json=ch.program_job_definition(program, candidate),
                    headers={**_JSON, "Content-Type": "application/json"},
                )
                raise_for_viya_status(resp)
                return candidate, True
            if ch.job_runs_only(existing, program, candidate):
                return candidate, False
        ch.fail(f"Found no free job name next to {program} (tried up to -50).")

    @mcp.tool()
    async def clinical_get_job_run(ctx: Context, run_id: str, wait_seconds: int = 0) -> dict[str, Any]:
        """Show the state of a clinical job run, optionally waiting for it to end.

        ``category`` is running, successful, warnings, canceled or error. A
        ``*_CHECKIN_ERROR`` state means a repository run ended but some outputs
        were not checked in; ``checkin_failures`` lists them.

        Args:
            run_id: The run's id.
            wait_seconds: Wait up to this long for the run to end (at most 600).
        """
        async with viya_session("clinical_get_job_run", ctx) as client:
            if err := await services(client):
                return err
            try:
                return await wait_for(client, run_id, wait_seconds)
            except ch.ClinicalError as exc:
                return exc.result()

    @mcp.tool()
    async def clinical_list_job_runs(
        ctx: Context, limit: int = 25, start: int = 0, all_users: bool = False, job_path: str | None = None
    ) -> dict[str, Any]:
        """List clinical job runs, newest first: tests, run-and-populate and scheduled runs included.

        Args:
            limit: Maximum runs (default 25).
            start: Offset, to page further back.
            all_users: Also list other users' runs that the server shows you (default: only yours).
            job_path: Only runs of this job.
        """
        async with viya_session("clinical_list_job_runs", ctx) as client:
            if err := await services(client):
                return err
            parts = []
            if not all_users:
                user = await _user_id(client)
                if user:
                    parts.append(f"eq(submittedBy,'{filter_literal(user)}')")
            if job_path:
                parts.append(f"eq(jobPath,'{filter_literal(ch.normalize_path(job_path))}')")
            params: dict[str, Any] = {"start": max(0, int(start)), "limit": max(1, min(int(limit), 200))}
            if parts:
                params["filter"] = parts[0] if len(parts) == 1 else f"and({','.join(parts)})"
            resp = await ch.get(client, f"{ch.JOBS}/submissions", params=params)
            page = resp.json()
            runs = [ch.normalize_submission(s) for s in page.get("items") or []]
            more = any(link.get("rel") == "next" for link in page.get("links") or [])
            return {"runs": runs, "start": params["start"], "has_more": more}

    @mcp.tool()
    async def clinical_cancel_job_run(ctx: Context, run_id: str) -> dict[str, Any]:
        """Cancel a clinical job run that has not ended.

        Args:
            run_id: The run's id.
        """
        async with viya_session("clinical_cancel_job_run", ctx) as client:
            if err := await services(client):
                return err
            resp = await client.put(
                f"{config.VIYA_ENDPOINT}{ch.JOBS}/submissions/{quote(run_id, safe='')}/state",
                params={"value": "canceled"},  # lowercase, or the service refuses it
                headers=_JSON,
            )
            if resp.status_code == 404:
                return {"status": "not_found", "message": f"No clinical job run {run_id!r}."}
            if resp.status_code == 409:
                # The run has ended (or not started): the service refuses with its reason.
                try:
                    reason = resp.json().get("message")
                except ValueError:
                    reason = None
                return {
                    "status": "not_cancelable",
                    "run_id": run_id,
                    "message": reason or "The run is not in a state that can be canceled.",
                }
            raise_for_viya_status(resp)
            return {"status": "canceling", "run_id": run_id}

    @mcp.tool()
    async def clinical_get_job_manifest(
        ctx: Context, run_id: str | None = None, cmnf_path: str | None = None, location: str = "workspace"
    ) -> dict[str, Any]:
        """The manifest of a clinical job run: the audit record of what it ran, read and wrote.

        Lists the job, how the run ended, who ran it and when, the parameter
        values (masked ones hidden), each program with its result and run time,
        and every input and output with its path, version, size and MD5
        checksum, plus where outputs are checked in. Use it to answer audit
        questions such as "which version of adsl did this run read?".

        The service keeps a run's manifest only while its ``.cmnf`` file is
        unchanged; the next run of the same job overwrites the file. Export
        what must be kept (clinical_read_file on the .cmnf, or this result).

        Args:
            run_id: The run's id.
            cmnf_path: Or the path of a ``.cmnf`` file (the latest run of its job).
            location: Where ``cmnf_path`` is: ``workspace`` (default) or ``repository``.
        """
        async with viya_session("clinical_get_job_manifest", ctx) as client:
            if err := await services(client):
                return err
            try:
                if (run_id is None) == (cmnf_path is None):
                    ch.fail("Give exactly one of run_id or cmnf_path.")
                cmnf: dict[str, Any] | None = None
                if cmnf_path:
                    where = ch.check_location(location)
                    target = ch.normalize_path(cmnf_path)
                    if where == ch.REPOSITORY:
                        item = await ch.resolve_repo_item(client, target)
                        data = await ch.read_repo_content(client, item)
                    else:
                        data = await ch.read_workspace_content(client, target)
                    cmnf = ch.parse_json_bytes(data, target)
                    run_id = (cmnf.get("jobSubmission") or {}).get("id")
                    if not run_id:
                        ch.fail(f"{target} names no run.")
                assert run_id is not None
                manifest = await manifest_for(client, run_id)
                if cmnf is None and not (manifest.get("checkinSpecification") or {}).get("locations"):
                    resource = (manifest.get("manifest") or {}).get("resource") or {}
                    if resource.get("path"):
                        try:
                            cmnf = ch.parse_json_bytes(await read_resource(client, resource), resource["path"])
                        except (ch.ClinicalError, httpx.HTTPError):
                            cmnf = None  # the manifest stands without the locations
                manifest = ch.fill_checkin_locations(manifest, cmnf)
                summary = ch.summarize_manifest(manifest)
                run = ch.normalize_submission(await submission(client, run_id))
            except ch.ClinicalError as exc:
                return exc.result()
            if run.get("checkin_failures"):
                summary["checkin_failures"] = run["checkin_failures"]
            return summary

    @mcp.tool()
    async def clinical_get_job_output(ctx: Context, run_id: str, kind: str = "log") -> dict[str, Any]:
        """Read the log or the results (listing) a clinical job run wrote.

        Args:
            run_id: The run's id.
            kind: ``log`` (default) or ``results``.
        """
        async with viya_session("clinical_get_job_output", ctx) as client:
            if err := await services(client):
                return err
            try:
                if kind not in ("log", "results"):
                    ch.fail("kind must be 'log' or 'results'.")
                manifest = await manifest_for(client, run_id)
                entries = manifest.get("logs" if kind == "log" else "results") or []
                files = []
                budget = _MAX_OUTPUT_CHARS
                for entry in entries:
                    resource = entry.get("resource") or entry.get("file") or {}
                    text = ch.decode_text(await read_resource(client, resource)) or "(not text)"
                    files.append({"path": resource.get("path"), "content": text[:budget]})
                    budget -= min(len(text), budget)
                    if budget <= 0:
                        files[-1]["truncated"] = True
                        break
            except ch.ClinicalError as exc:
                return exc.result()
            if not files:
                return {"status": "not_found", "message": f"Run {run_id} lists no {kind} files."}
            errors = ch.find_sas_errors("\n".join(f["content"] for f in files)) if kind == "log" else []
            return {"run_id": run_id, "kind": kind, "files": files, **({"errors": errors[:20]} if errors else {})}

    # --- schedules -----------------------------------------------------------------------

    @mcp.tool()
    async def clinical_list_schedules(ctx: Context, all_users: bool = False) -> dict[str, Any]:
        """List the schedules that run clinical jobs at set times.

        Args:
            all_users: Also list other users' schedules that the server shows you
                (default: only those that run as you).
        """
        async with viya_session("clinical_list_schedules", ctx) as client:
            if err := await services(client):
                return err
            # The service does not page this list and returns every user's schedules.
            schedules = await ch.collect(
                client, f"{ch.JOBS}/schedules", params={"limit": 1000}, accept="application/json"
            )
            user = await _user_id(client)
            out = [ch.summarize_schedule(s, user) for s in schedules]
            if not all_users and user:
                out = [s for s in out if s.get("yours")]
            return {"schedules": out}

    @mcp.tool()
    async def clinical_get_schedule(ctx: Context, schedule_id: str, next_runs: int = 5) -> dict[str, Any]:
        """Show a clinical job schedule with its next run times as the server calculates them.

        Check the next runs after creating or changing a schedule: they show
        whether it does what was meant.

        Args:
            schedule_id: The schedule's id.
            next_runs: How many upcoming run times to list (default 5).
        """
        async with viya_session("clinical_get_schedule", ctx) as client:
            if err := await services(client):
                return err
            try:
                base = f"{ch.JOBS}/schedules/{quote(schedule_id, safe='')}"
                schedule = (await ch.get(client, base, missing=f"No schedule {schedule_id!r}.")).json()
                future = await ch.get(client, f"{base}/future", params={"limit": max(1, min(int(next_runs), 50))})
            except ch.ClinicalError as exc:
                return exc.result()
            times = future.json()
            if isinstance(times, dict):  # a collection, or a bare array: both occur
                times = times.get("items") or []
            return {**ch.summarize_schedule(schedule, await _user_id(client)), "next_runs": times}

    @mcp.tool()
    async def clinical_save_schedule(
        ctx: Context,
        name: str,
        job_path: str,
        trigger: JsonDict,
        location: str = "repository",
        parameter_values: JsonDict | None = None,
        description: str | None = None,
        schedule_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a schedule that runs a clinical job at set times under your user, or replace one.

        ``trigger`` examples (``timeZone`` is required, an IANA name):

        * once: ``{"type": "DATELIST", "dates": ["2026-12-01"], "hours": 6, "minutes": 0,
          "timeZone": "Europe/Berlin"}``
        * daily at 02:30: ``{"type": "DAILY", "hours": 2, "minutes": 30, "timeZone": "America/New_York"}``
        * Mondays and Thursdays: ``{"type": "WEEKLY", "daysOfWeek": ["MON", "THU"], "hours": 7, "minutes": 0, ...}``
        * first Monday monthly: ``{"type": "MONTHLY", "occurrence": "FIRST", "daysOfWeek": ["MON"], ...}``
        * every 15 minutes: ``{"type": "MINUTELY", "interval": 15, ...}``
        * hourly at :05: ``{"type": "HOURLY", "minutes": 5, ...}``

        Repeating schedules also take ``interval``, ``start`` and ``end`` dates
        (YYYY-MM-DD) and ``numOccurrences``. YEARLY needs ``monthOfYear``.
        clinical_get_schedule shows the resulting next runs.

        Args:
            name: The schedule's name.
            job_path: The ``.cjob`` the schedule runs.
            trigger: When it runs (see above).
            location: Where the job is: ``repository`` (default) or ``workspace``.
            parameter_values: ``{"name": value}`` for the job's parameters; defaults otherwise.
            description: Optional description.
            schedule_id: Replace this existing schedule instead of creating one.
        """
        async with viya_session("clinical_save_schedule", ctx) as client:
            if err := await services(client):
                return err
            try:
                where = ch.check_location(location)
                target = ch.normalize_path(job_path)
                job = await load_job(client, target, where)
                values = ch.encode_parameter_values(job.get("parameters") or [], parameter_values, target)
                body: dict[str, Any] = {
                    "name": name,
                    "jobInfo": {"job": ch.job_reference(target, where), "parameterValues": values},
                    "trigger": ch.build_schedule_trigger(trigger),
                }
                if description:
                    body["description"] = description
                headers = {**_JSON, "Content-Type": "application/json"}
                if schedule_id:
                    resp = await client.patch(
                        f"{config.VIYA_ENDPOINT}{ch.JOBS}/schedules/{quote(schedule_id, safe='')}",
                        json=body,
                        headers=headers,
                    )
                    if resp.status_code == 404:
                        ch.fail(f"No schedule {schedule_id!r}.", "not_found")
                else:
                    resp = await client.post(f"{config.VIYA_ENDPOINT}{ch.JOBS}/schedules", json=body, headers=headers)
                raise_for_viya_status(resp)
                saved = resp.json()
            except ch.ClinicalError as exc:
                return exc.result()
            return {
                "status": "updated" if schedule_id else "created",
                "schedule": ch.summarize_schedule(saved, await _user_id(client)),
            }

    @mcp.tool()
    async def clinical_set_schedule_state(ctx: Context, schedule_id: str, enabled: bool) -> dict[str, Any]:
        """Enable or disable a clinical job schedule without deleting it.

        Args:
            schedule_id: The schedule's id.
            enabled: ``true`` to resume its runs, ``false`` to stop them.
        """
        async with viya_session("clinical_set_schedule_state", ctx) as client:
            if err := await services(client):
                return err
            value = "ENABLED" if enabled else "DISABLED"  # uppercase, or the service refuses it
            resp = await client.put(
                f"{config.VIYA_ENDPOINT}{ch.JOBS}/schedules/{quote(schedule_id, safe='')}/state",
                params={"value": value},
                headers=_JSON,
            )
            if resp.status_code == 404:
                return {"status": "not_found", "message": f"No schedule {schedule_id!r}."}
            raise_for_viya_status(resp)
            return {"status": value.lower(), "schedule_id": schedule_id}

    @mcp.tool()
    async def clinical_delete_schedule(ctx: Context, schedule_id: str) -> dict[str, Any]:
        """Delete a clinical job schedule. The job itself is kept.

        Args:
            schedule_id: The schedule's id.
        """
        async with viya_session("clinical_delete_schedule", ctx) as client:
            if err := await services(client):
                return err
            resp = await client.delete(f"{config.VIYA_ENDPOINT}{ch.JOBS}/schedules/{quote(schedule_id, safe='')}")
            if resp.status_code == 404:
                return {"status": "not_found", "message": f"No schedule {schedule_id!r}."}
            raise_for_viya_status(resp)
            return {"status": "deleted", "schedule_id": schedule_id}

    @mcp.tool()
    async def clinical_run_schedule_now(ctx: Context, schedule_id: str, wait_seconds: int = 0) -> dict[str, Any]:
        """Run a schedule's job now, with the schedule's parameter values.

        Args:
            schedule_id: The schedule's id.
            wait_seconds: How long to wait for the run to end (at most 600).
        """
        async with viya_session("clinical_run_schedule_now", ctx) as client:
            if err := await services(client):
                return err
            try:
                schedule = (
                    await ch.get(
                        client,
                        f"{ch.JOBS}/schedules/{quote(schedule_id, safe='')}",
                        missing=f"No schedule {schedule_id!r}.",
                    )
                ).json()
                job = (schedule.get("jobInfo") or {}).get("job") or {}
                if not job.get("path"):
                    ch.fail("The schedule names no job.")
                location = (job.get("resourceLocation") or "REPOSITORY").lower()
                run_id = await submit(
                    client, job["path"], location, "run", False, ch.schedule_parameter_values(schedule)
                )
                return await wait_for(client, run_id, wait_seconds)
            except ch.ClinicalError as exc:
                return exc.result()
