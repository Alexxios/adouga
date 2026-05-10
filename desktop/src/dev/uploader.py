"""Yandex Disk uploader for ML training session archives.

Reads the OAuth token from the ``YADISK_TOKEN`` environment variable
(or a ``.env`` file in the current working directory / project root).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yadisk
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_REMOTE_BASE_DIR = "/adouga/ml_sessions"
_ENV_VAR = "YADISK_TOKEN"


def _resolve_token() -> str:
    """Load token from environment, falling back to .env file."""
    # PyInstaller onefile bundles extract data to a temp dir
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        load_dotenv(Path(bundle_dir) / ".env")
    else:
        load_dotenv()
    token = os.environ.get(_ENV_VAR)
    if not token:
        raise EnvironmentError(
            f"{_ENV_VAR} is not set. "
            "Add it to your environment or to a .env file."
        )
    return token


class YaDiskUploader:
    """Uploads ZIP session archives to Yandex Disk.

    Parameters
    ----------
    token:
        OAuth token for Yandex Disk.  When *None* the token is loaded from
        the ``YADISK_TOKEN`` environment variable (or ``.env`` file).
    remote_base:
        Remote directory on Yandex Disk where sessions are stored.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        remote_base: str = _REMOTE_BASE_DIR,
        tester_id: str = "",
    ) -> None:
        self._token = token if token is not None else _resolve_token()
        self._remote_base = remote_base.rstrip("/")
        self._tester_id = tester_id
        logger.info("YaDiskUploader initialised — remote base: %s", self._remote_base)

    def set_tester_id(self, tester_id: str) -> None:
        """Update the tester identifier used for remote path namespacing."""
        self._tester_id = tester_id
        logger.info("Tester ID set: %r", tester_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Return *True* if the token is valid and Yandex Disk is reachable."""
        try:
            with yadisk.Client(token=self._token) as client:
                ok = client.check_token()
            logger.debug("Yandex Disk connection check: %s", "ok" if ok else "failed")
            return bool(ok)
        except Exception:
            logger.warning("Yandex Disk connection check raised an exception", exc_info=True)
            return False

    def upload(self, local_path: Path, overwrite: bool = True) -> str:
        """Upload *local_path* to Yandex Disk.

        Parameters
        ----------
        local_path:
            Path to the local ZIP archive.
        overwrite:
            Replace the remote file if it already exists.

        Returns
        -------
        str
            Remote path of the uploaded file.
        """
        local_path = Path(local_path)
        if self._tester_id:
            remote_path = f"{self._remote_base}/{self._tester_id}/{local_path.name}"
        else:
            remote_path = f"{self._remote_base}/{local_path.name}"
        logger.info("Uploading %s → %s", local_path, remote_path)

        with yadisk.Client(token=self._token) as client:
            self._ensure_remote_dir(client, remote_path)
            with open(local_path, "rb") as fh:
                client.upload(fh, remote_path, overwrite=overwrite)

        size_kb = local_path.stat().st_size / 1024
        logger.info("Upload complete: %s (%.1f KB) → %s", local_path.name, size_kb, remote_path)
        return remote_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_remote_dir(self, client: yadisk.Client, remote_path: str) -> None:
        """Create all parent directories for *remote_path* if absent."""
        dir_path = remote_path.rsplit("/", 1)[0]
        parts = dir_path.strip("/").split("/")
        for depth in range(1, len(parts) + 1):
            path = "/" + "/".join(parts[:depth])
            try:
                if not client.exists(path):
                    client.mkdir(path)
                    logger.debug("Created remote directory: %s", path)
            except Exception:
                logger.exception("Failed to create remote directory %s", path)
                raise
