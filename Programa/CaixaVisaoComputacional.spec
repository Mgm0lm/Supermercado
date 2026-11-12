# -*- mode: python ; coding: utf-8 -*-
import sys
from importlib.util import find_spec

from PyInstaller.utils.hooks import collect_all

if sys.version_info[:3] == (3, 12, 0):
    raise SystemExit("Para empacotar, use Python 3.12.1 ou mais recente da série 3.12.")


def runtime_module(name):
    return not {"test", "tests", "benchmarks", "conftest"}.intersection(name.split("."))


datas = [('export_classifier.py', '.')]
binaries = []
hiddenimports = ['scipy.stats._continuous_distns', 'scipy.stats._discrete_distns']
for package in ('mediapipe', 'sklearn', 'scipy', 'joblib', 'rapidocr', 'rapidocr_onnxruntime'):
    if find_spec(package) is None:
        continue
    package_datas, package_binaries, package_imports = collect_all(
        package,
        filter_submodules=runtime_module,
        exclude_datas=['**/test/**', '**/tests/**', '**/benchmarks/**'],
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CaixaVisaoComputacional',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CaixaVisaoComputacional',
)
