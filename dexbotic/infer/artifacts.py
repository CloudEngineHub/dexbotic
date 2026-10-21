"""Model-independent integrity checks and transactional artifact publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | Path, *, filename_key: str = "file") -> dict:
    """Describe one immutable artifact using its basename, size, and digest."""
    if not isinstance(filename_key, str) or not filename_key:
        raise ValueError("filename_key must be a nonempty string")
    path = Path(path)
    return {
        filename_key: path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_file_record(record: dict, *, filename_key: str) -> None:
    if not isinstance(record, dict):
        raise ValueError("Invalid inference file record.")
    filename = record.get(filename_key)
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(
            "File records must reference a plain filename in the engine directory."
        )
    if filename in (".", ".."):
        raise ValueError("Invalid engine or manifest filename.")
    if type(record.get("bytes")) is not int or record["bytes"] <= 0:
        raise ValueError("file bytes must be a positive integer")
    if not isinstance(record.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", record["sha256"]
    ):
        raise ValueError("File record requires a SHA256 digest.")


def read_verified_file(path: str | Path, record: dict) -> bytes:
    """Read once, verify, then hand these exact bytes to the deserializer."""
    path = Path(path)
    validate_file_record(
        record, filename_key="engine" if "engine" in record else "file"
    )
    if path.is_symlink():
        raise ValueError(f"Engine or manifest file cannot be a symlink: {path.name}.")
    payload = path.read_bytes()
    if (
        len(payload) != record["bytes"]
        or hashlib.sha256(payload).hexdigest() != record["sha256"]
    ):
        raise ValueError(f"Inference file integrity mismatch: {path.name}.")
    return payload


@contextmanager
def engine_directory_lock(engine_dir: str | Path, *, exclusive: bool = False):
    """Coordinate engine publication with readers loading all layers."""
    directory = Path(engine_dir).expanduser()
    if directory.is_symlink():
        raise ValueError("Inference engine directory cannot be a symlink.")
    directory = directory.resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent / f".{directory.name}.lock"
    try:
        handle = lock_path.open("r+b" if exclusive else "rb")
    except FileNotFoundError:
        try:
            handle = lock_path.open("a+b")
        except OSError as exc:
            raise OSError(
                exc.errno,
                "Missing engine directory lock cannot be created; keep the directory and "
                "its sibling lock together for read-only deployment.",
                str(lock_path),
            ) from exc
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cleanup_engine_directory(path: str | Path, *, purpose: str) -> None:
    """Ignore ordinary filesystem cleanup errors without changing success."""
    path = Path(path)
    try:
        if not path.exists():
            return
        shutil.rmtree(path)
    except OSError as exc:
        logger.warning("Failed to remove %s %s: %s", purpose, path, exc)


def _recovery_path_exists(path: Path, primary: BaseException) -> bool | None:
    """Probe recovery state without replacing the operation's primary error."""
    try:
        return path.exists()
    except OSError as probe_error:
        logger.error("Failed to inspect engine recovery path %s: %s", path, probe_error)
        if hasattr(primary, "add_note"):
            primary.add_note(
                f"Could not inspect engine recovery path {path}: {probe_error!r}"
            )
        return None


def publish_engine_generation(
    staging: str | Path,
    engine_dir: str | Path,
    backup: str | Path,
) -> None:
    """Publish a complete directory generation and preserve recoverability.

    A rename that moved ``staging`` into ``engine_dir`` is the commit point,
    even if Python observes an exception immediately after the syscall. Before
    that point, an existing generation is restored from ``backup``. Rollback
    failures retain the backup and annotate, but never replace, the publication
    exception.
    """
    staging = Path(staging)
    engine_dir = Path(engine_dir)
    backup = Path(backup)
    if backup.exists():
        raise FileExistsError(f"Engine backup path already exists: {backup}")

    try:
        if engine_dir.exists():
            engine_dir.rename(backup)
        staging.rename(engine_dir)
    except BaseException as publish_error:
        # os.rename is atomic, but an injected exception or asynchronous
        # interruption can be observed just after it succeeds. The new target
        # plus an absent staging directory proves that publication committed.
        engine_exists = _recovery_path_exists(engine_dir, publish_error)
        staging_exists = _recovery_path_exists(staging, publish_error)
        if engine_exists is True and staging_exists is False:
            logger.warning(
                "Engine directory rename reported %s after publication committed at %s",
                type(publish_error).__name__,
                engine_dir,
            )
            raise
        else:
            backup_exists = _recovery_path_exists(backup, publish_error)
            if backup_exists is True and engine_exists is False:
                try:
                    backup.rename(engine_dir)
                except BaseException as rollback_error:
                    logger.error(
                        "Failed to restore engines %s from backup %s: %s",
                        engine_dir,
                        backup,
                        rollback_error,
                    )
                    if hasattr(publish_error, "add_note"):
                        publish_error.add_note(
                            "Engine directory rollback reported an error; inspect both "
                            f"{engine_dir} and {backup}. Neither path is cleaned "
                            f"by this failure handler: {rollback_error!r}"
                        )
            raise

    # Once the commit point is reached, failure to retire the old generation
    # must not turn usable newly published engines into a reported failure.
    cleanup_engine_directory(backup, purpose="retired engine backup")
