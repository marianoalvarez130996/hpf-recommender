"""Rutas del repositorio para los scripts de evaluación."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "python" / "tests")]
