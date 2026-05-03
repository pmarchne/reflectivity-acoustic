"""
Custom build hook: compiles the Fortran/f2py extensions automatically
when running `pip install -e .` or `pip install .`.
"""

import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py


FORTRAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "fortran")

FORT_FLAGS = "-O3 -march=native -ffast-math -funroll-loops -fopenmp -fopenmp-simd"

MODULES = [
    ("reflectivity",     "reflectivity.f90"),
    ("reflectivity_adj", "reflectivity_adj.f90"),
]


def compile_fortran_extensions():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

    for module, source in MODULES:
        print(f"[setup.py] compiling Fortran extension: {module}")
        subprocess.check_call(
            [
                sys.executable, "-m", "numpy.f2py",
                "--backend", "meson",
                "-c", "-m", module, source,
                f"--f90flags={FORT_FLAGS}",
                f"--opt={FORT_FLAGS}",
                "-lgomp",
            ],
            cwd=FORTRAN_DIR,
            env=env,
        )


class BuildPyWithFortran(build_py):
    """Extend build_py to compile Fortran extensions before copying sources."""

    def run(self):
        compile_fortran_extensions()
        super().run()


setup(
    cmdclass={"build_py": BuildPyWithFortran},
)
