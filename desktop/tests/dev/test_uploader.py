"""Tests for src.dev.uploader — YaDiskUploader."""

import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dev.uploader import YaDiskUploader, _ENV_VAR, _REMOTE_BASE_DIR, _resolve_token


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FAKE_TOKEN = "fake_oauth_token_for_tests"


def _make_zip(tmp_path: Path, name: str = "session.zip") -> Path:
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("metadata.json", '{"sample_count": 1}')
    return p


def _mock_client(exists: bool = True, check_token: bool = True):
    """Return a context-manager-compatible yadisk.Client mock."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    m.exists.return_value = exists
    m.check_token.return_value = check_token
    return m


# ---------------------------------------------------------------------------
# _resolve_token
# ---------------------------------------------------------------------------

def test_resolve_token_reads_from_env(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, _FAKE_TOKEN)
    with patch("src.dev.uploader.load_dotenv"):
        assert _resolve_token() == _FAKE_TOKEN


def test_resolve_token_raises_when_missing(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    with patch("src.dev.uploader.load_dotenv"):
        with pytest.raises(EnvironmentError, match=_ENV_VAR):
            _resolve_token()


# ---------------------------------------------------------------------------
# YaDiskUploader — init
# ---------------------------------------------------------------------------

def test_uploader_explicit_token():
    assert YaDiskUploader(token=_FAKE_TOKEN)._token == _FAKE_TOKEN


def test_uploader_default_remote_base():
    assert YaDiskUploader(token=_FAKE_TOKEN)._remote_base == _REMOTE_BASE_DIR


def test_uploader_custom_remote_base():
    assert YaDiskUploader(token=_FAKE_TOKEN, remote_base="/custom/dir")._remote_base == "/custom/dir"


def test_uploader_trailing_slash_stripped():
    assert YaDiskUploader(token=_FAKE_TOKEN, remote_base="/custom/dir/")._remote_base == "/custom/dir"


def test_uploader_loads_token_from_env(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, _FAKE_TOKEN)
    with patch("src.dev.uploader.load_dotenv"):
        assert YaDiskUploader()._token == _FAKE_TOKEN


def test_uploader_raises_without_token(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    with patch("src.dev.uploader.load_dotenv"):
        with pytest.raises(EnvironmentError):
            YaDiskUploader()


# ---------------------------------------------------------------------------
# YaDiskUploader — check_connection
# ---------------------------------------------------------------------------

def test_check_connection_true():
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    with patch("src.dev.uploader.yadisk.Client", return_value=_mock_client(check_token=True)):
        assert uploader.check_connection() is True


def test_check_connection_false_on_bad_token():
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    with patch("src.dev.uploader.yadisk.Client", return_value=_mock_client(check_token=False)):
        assert uploader.check_connection() is False


def test_check_connection_false_on_exception():
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    with patch("src.dev.uploader.yadisk.Client", side_effect=RuntimeError("network error")):
        assert uploader.check_connection() is False


# ---------------------------------------------------------------------------
# YaDiskUploader — _ensure_remote_dir
# ---------------------------------------------------------------------------

def test_ensure_remote_dir_creates_missing():
    uploader = YaDiskUploader(token=_FAKE_TOKEN, remote_base="/adouga/ml_sessions")
    client = MagicMock()
    client.exists.return_value = False
    uploader._ensure_remote_dir(client, "/adouga/ml_sessions/file.zip")
    assert client.mkdir.call_count == 2  # /adouga and /adouga/ml_sessions


def test_ensure_remote_dir_skips_existing():
    uploader = YaDiskUploader(token=_FAKE_TOKEN, remote_base="/adouga/ml_sessions")
    client = MagicMock()
    client.exists.return_value = True
    uploader._ensure_remote_dir(client, "/adouga/ml_sessions/file.zip")
    client.mkdir.assert_not_called()


def test_ensure_remote_dir_raises_on_mkdir_failure():
    uploader = YaDiskUploader(token=_FAKE_TOKEN, remote_base="/adouga/ml_sessions")
    client = MagicMock()
    client.exists.return_value = False
    client.mkdir.side_effect = RuntimeError("quota exceeded")
    with pytest.raises(RuntimeError, match="quota exceeded"):
        uploader._ensure_remote_dir(client, "/adouga/ml_sessions/file.zip")


# ---------------------------------------------------------------------------
# YaDiskUploader — upload
# ---------------------------------------------------------------------------

def test_upload_returns_remote_path(tmp_path):
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    zip_file = _make_zip(tmp_path, "session_test.zip")
    with patch("src.dev.uploader.yadisk.Client", return_value=_mock_client()):
        remote = uploader.upload(zip_file)
    assert remote == f"{_REMOTE_BASE_DIR}/session_test.zip"


def test_upload_calls_yadisk_upload(tmp_path):
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    zip_file = _make_zip(tmp_path, "session_test.zip")
    client = _mock_client()
    with patch("src.dev.uploader.yadisk.Client", return_value=client):
        uploader.upload(zip_file)
    client.upload.assert_called_once()
    assert "session_test.zip" in client.upload.call_args[0][1]


def test_upload_overwrite_true(tmp_path):
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    zip_file = _make_zip(tmp_path)
    client = _mock_client()
    with patch("src.dev.uploader.yadisk.Client", return_value=client):
        uploader.upload(zip_file, overwrite=True)
    assert client.upload.call_args[1]["overwrite"] is True


def test_upload_overwrite_false(tmp_path):
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    zip_file = _make_zip(tmp_path)
    client = _mock_client()
    with patch("src.dev.uploader.yadisk.Client", return_value=client):
        uploader.upload(zip_file, overwrite=False)
    assert client.upload.call_args[1]["overwrite"] is False


def test_upload_raises_if_file_missing(tmp_path):
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    with patch("src.dev.uploader.yadisk.Client", return_value=_mock_client()):
        with pytest.raises(FileNotFoundError):
            uploader.upload(tmp_path / "ghost.zip")


def test_upload_propagates_yadisk_errors(tmp_path):
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    zip_file = _make_zip(tmp_path)
    client = _mock_client()
    client.upload.side_effect = RuntimeError("disk full")
    with patch("src.dev.uploader.yadisk.Client", return_value=client):
        with pytest.raises(RuntimeError, match="disk full"):
            uploader.upload(zip_file)


def test_upload_accepts_string_path(tmp_path):
    uploader = YaDiskUploader(token=_FAKE_TOKEN)
    zip_file = _make_zip(tmp_path, "session.zip")
    with patch("src.dev.uploader.yadisk.Client", return_value=_mock_client()):
        remote = uploader.upload(str(zip_file))
    assert "session.zip" in remote


# ---------------------------------------------------------------------------
# Integration (require real YADISK_TOKEN env var)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get(_ENV_VAR), reason=f"{_ENV_VAR} not set")
def test_real_check_connection():
    assert YaDiskUploader().check_connection() is True


@pytest.mark.skipif(not os.environ.get(_ENV_VAR), reason=f"{_ENV_VAR} not set")
def test_real_upload_and_cleanup(tmp_path):
    import yadisk
    zip_file = _make_zip(tmp_path, "integration_test.zip")
    uploader = YaDiskUploader(remote_base="/adouga/ml_sessions/_test")
    remote = uploader.upload(zip_file, overwrite=True)
    with yadisk.Client(token=uploader._token) as client:
        assert client.exists(remote)
        client.remove(remote)
