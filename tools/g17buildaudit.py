"""Compatibility entry for the library's committed-source audit.

Public functions retain identity. This module remains present in sys.modules so
cold audits record the compatibility entry as well as its implementation.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.g17.audit import committed_hashes, sha, verified_build, verified_reference
