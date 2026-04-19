"""Safe local reconstruction of per-brain transfer/process/verify/cleanup orchestration.

This module does not connect to OMERO or institutional infrastructure.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import shutil

def process_one_brain(source: Path, temporary_root: Path, output_root: Path,
                      processor: Callable[[Path, Path], None], *, cleanup: bool = False) -> Path:
    """Stage one brain, run a supplied processor, verify nonempty output, optionally clean staging.

    Never removes source data. Caller is responsible for secure access and storage quotas.
    """
    source=Path(source).resolve(); temporary_root=Path(temporary_root).resolve()
    output_root=Path(output_root).resolve()
    if not source.is_dir() or not any(source.iterdir()):
        raise ValueError("Source brain directory missing or empty")
    if temporary_root == output_root or source in (temporary_root, output_root):
        raise ValueError("Input, staging and output roots must be distinct")
    if temporary_root in source.parents or output_root in source.parents:
        raise ValueError("Unsafe directory placement")
    stage=temporary_root/source.name
    dest=output_root/source.name
    if stage.exists() or dest.exists():
        raise FileExistsError("Refusing to overwrite existing stage or results")
    stage.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, stage)
    try:
        processor(stage, dest)
        if not any(f.is_file() and f.stat().st_size > 0 for f in dest.rglob('*')):
            raise RuntimeError("Processor produced no nonempty files; stage retained for debugging")
        if cleanup:
            shutil.rmtree(stage)
        return dest
    except Exception:
        # Preserve staging and any partial outputs for diagnosis; never silently delete inputs.
        raise
