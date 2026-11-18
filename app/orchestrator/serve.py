"""Start the orchestrator over HTTP, or HTTPS when a certificate is configured.

    python -m app.orchestrator.serve          # run the server (port 8000)
    python -m app.orchestrator.serve check    # container health check (works for HTTP and HTTPS)

HTTPS is enabled when both STELLA_TLS_CERTFILE and STELLA_TLS_KEYFILE point to PEM
files. Browsers allow the microphone only on https:// or http://localhost, so HTTPS
is what lets testers on other machines use the page without an SSH tunnel.
Used by compose.gpu.yaml; compose.yaml (CPU laptop) keeps its plain uvicorn command.
"""

import os
import ssl
import sys
import urllib.request
from pathlib import Path

import uvicorn

PORT = 8000


def tls_files() -> tuple[str, str] | None:
    cert = os.environ.get("STELLA_TLS_CERTFILE", "").strip()
    key = os.environ.get("STELLA_TLS_KEYFILE", "").strip()
    if not cert and not key:
        return None
    if not (cert and key):
        raise SystemExit("Set both STELLA_TLS_CERTFILE and STELLA_TLS_KEYFILE, or neither.")
    missing = [path for path in (cert, key) if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"TLS file(s) not found in the container: {', '.join(missing)} (is ./certs mounted?)")
    return cert, key


def run() -> None:
    files = tls_files()
    options = {"host": "0.0.0.0", "port": PORT}
    if files:
        options.update(ssl_certfile=files[0], ssl_keyfile=files[1])
    print(f"Stella orchestrator on {'https' if files else 'http'}://0.0.0.0:{PORT}", flush=True)
    uvicorn.run("app.orchestrator.main:app", **options)


def check() -> None:
    https = tls_files() is not None
    # The certificate is usually self-signed and issued for the host name, not "localhost".
    context = ssl._create_unverified_context() if https else None  # noqa: S323 - local health check only
    url = f"{'https' if https else 'http'}://localhost:{PORT}/health"
    urllib.request.urlopen(url, timeout=10, context=context)


if __name__ == "__main__":
    check() if sys.argv[1:] == ["check"] else run()
