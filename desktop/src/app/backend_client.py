"""Thin HTTP client for the Adouga backend.

Stateless except for the auth token, which is held in memory only — there is
no on-disk persistence for the demo.
"""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional


class BackendError(RuntimeError):
    """Raised when the backend returns a non-2xx response or is unreachable."""


class BackendClient:
    def __init__(self, base_url: str = "http://localhost:8008"):
        self._lock = threading.Lock()
        self._base_url = base_url.rstrip("/")
        self._token: Optional[str] = None
        self._username: Optional[str] = None

    # ---- properties ---------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._base_url

    def set_base_url(self, url: str) -> None:
        with self._lock:
            self._base_url = url.rstrip("/")

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None

    @property
    def username(self) -> Optional[str]:
        return self._username

    # ---- auth ---------------------------------------------------------

    def login(self, username: str, password: str, timeout: float = 5.0) -> None:
        """OAuth2 password flow → JWT. Raises BackendError on failure."""
        body = urllib.parse.urlencode(
            {"username": username, "password": password, "grant_type": "password"}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/auth/login",
            data=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        payload = self._send(req, timeout=timeout)
        token = payload.get("access_token")
        if not token:
            raise BackendError("No access_token in login response")
        with self._lock:
            self._token = token
            self._username = username

    def register(self, username: str, password: str, timeout: float = 5.0) -> None:
        """Create a user. Idempotent-ish: 409 is treated as success."""
        body = json.dumps({"username": username, "password": password}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/auth/register",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            self._send(req, timeout=timeout)
        except BackendError as e:
            if "409" not in str(e):
                raise

    def logout(self) -> None:
        with self._lock:
            self._token = None
            self._username = None

    # ---- predictions --------------------------------------------------

    def post_prediction(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: dict,
        timestamp: Optional[float] = None,
        timeout: float = 5.0,
    ) -> dict:
        """POST /predictions. Requires prior login()."""
        if not self._token:
            raise BackendError("Not authenticated")
        ts = (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            if timestamp is not None
            else datetime.now(tz=timezone.utc)
        )
        body = json.dumps(
            {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "probabilities": {k: float(v) for k, v in probabilities.items()},
                "timestamp": ts.isoformat(),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/predictions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
        )
        return self._send(req, timeout=timeout)

    # ---- internal -----------------------------------------------------

    @staticmethod
    def _send(req: urllib.request.Request, timeout: float) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8") or "{}"
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = ""
            raise BackendError(f"HTTP {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise BackendError(f"Connection error: {e.reason}") from e
        except TimeoutError as e:
            raise BackendError("Request timed out") from e
