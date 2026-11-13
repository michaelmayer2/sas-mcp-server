# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upload sources the server reads itself: a file on its own disk or a URL.

Shared by every tool that takes ``file_path`` / ``url`` in place of inline
content (Tier 2's ``upload_data`` and ``upload_file``, the Clinical Repository
upload tools), so the ``MAX_UPLOAD_BYTES`` cap and the ``ALLOW_LOCAL_FILE_UPLOAD``
switch are enforced the same way everywhere.
"""

from pathlib import Path
from typing import Any

import httpx

from ..config import MAX_UPLOAD_BYTES, SSL_VERIFY
from ..env import env_bool


def source_too_large(source: str, size_bytes: int | None) -> dict[str, Any]:
    """Structured refusal for an upload source over ``MAX_UPLOAD_BYTES``."""
    error: dict[str, Any] = {
        "status": "source_too_large",
        "source": source,
        "limit_bytes": MAX_UPLOAD_BYTES,
        "message": (
            f"The {source} source exceeds the {MAX_UPLOAD_BYTES:,}-byte upload "
            "limit (MAX_UPLOAD_BYTES; the default matches SAS Viya's 100 MB "
            "file-upload cap, so a larger payload would be refused upstream "
            "anyway). For large data, load it via a path-based caslib and "
            "promote_table_to_memory, or ask the administrator to raise both limits."
        ),
    }
    if size_bytes is not None:
        error["size_bytes"] = size_bytes
    return error


async def resolve_source_bytes(file_path: str | None, url: str | None) -> tuple[bytes | None, dict[str, Any] | None]:
    """Materialize the upload bytes server-side from ``file_path`` or ``url``.

    Assumes exactly one of the two is set (the caller's exactly-one-source
    guard). The bytes are read off the server's disk or fetched over HTTP, never
    routed through the model context. Either source is refused past
    ``MAX_UPLOAD_BYTES`` — the payload is held in memory in full while it is
    re-posted to Viya, so the cap is what keeps one oversized source from
    exhausting the process. Returns ``(bytes, None)`` or ``(None, error)``.
    """
    if file_path:
        if not env_bool("ALLOW_LOCAL_FILE_UPLOAD", True):
            return None, {
                "status": "file_upload_disabled",
                "message": (
                    "Server-side file reads are disabled "
                    "(ALLOW_LOCAL_FILE_UPLOAD=false). Use url or the upload_inline_data tool."
                ),
            }
        path = Path(file_path).expanduser()
        if not path.is_file():
            return None, {
                "status": "file_not_found",
                "file_path": file_path,
                "message": (
                    f"No readable file at '{file_path}' on the server host. In "
                    "stdio mode that host is your local machine; pass an absolute path."
                ),
            }
        try:
            size = path.stat().st_size
            if size > MAX_UPLOAD_BYTES:
                return None, source_too_large("file_path", size)
            return path.read_bytes(), None
        except OSError as exc:
            return None, {
                "status": "file_unreadable",
                "file_path": file_path,
                "message": str(exc),
            }
    # url — fetch with a plain client (no Viya bearer on an external URL).
    assert url is not None
    try:
        async with (
            httpx.AsyncClient(timeout=120.0, follow_redirects=True, verify=SSL_VERIFY) as fetch_client,
            fetch_client.stream("GET", url) as fetch_resp,
        ):
            # Plain raise_for_status on purpose: this is an arbitrary caller-supplied
            # URL, not Viya, so there is no vnd.sas.error+json body worth quoting
            # and folding a third-party page into the message adds only noise.
            fetch_resp.raise_for_status()
            # Refuse on the declared size when there is one, but never trust
            # it: the cap is enforced again while the body streams in, so an
            # absent or lying Content-Length cannot blow past the limit.
            declared = fetch_resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
                return None, source_too_large("url", int(declared))
            buf = bytearray()
            async for chunk in fetch_resp.aiter_bytes():
                buf += chunk
                if len(buf) > MAX_UPLOAD_BYTES:
                    return None, source_too_large("url", None)
            return bytes(buf), None
    except httpx.HTTPError as exc:
        return None, {"status": "fetch_failed", "url": url, "message": str(exc)}
