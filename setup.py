from __future__ import annotations

import subprocess
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

ROOT = Path(__file__).parent.resolve()


class BuildRvo2Ext(_build_ext):
    """Builds the native RVO2 library before compiling Cython bindings."""

    def run(self) -> None:
        build_dir = Path(self.build_temp) / "rvo2"
        build_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "cmake",
            str(ROOT),
            "-DCMAKE_CXX_FLAGS=-fPIC",
            "-DRVO_BUILD_EXAMPLES=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        subprocess.check_call(cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

        for ext in self.extensions:
            ext.library_dirs = [str(build_dir / "src")]

        super().run()


extensions = [
    Extension(
        "rvo2",
        ["src/rvo2.pyx"],
        include_dirs=["src"],
        libraries=["RVO"],
        extra_compile_args=["-fPIC"],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    cmdclass={"build_ext": BuildRvo2Ext},
)
