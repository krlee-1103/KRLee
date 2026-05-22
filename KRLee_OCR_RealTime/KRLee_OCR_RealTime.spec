# -*- mode: python ; coding: utf-8 -*-
"""
KRLee_OCR_RealTime PyInstaller 빌드 스펙
- rapidocr_onnxruntime 모델(.onnx) + config.yaml 번들
- onnxruntime DLL 번들
- HIKROBOT MVS Runtime DLL 자동 탐색·포함
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── rapidocr 데이터 (모델 + config) ─────────────────────────────────────────
try:
    rapidocr_datas = collect_data_files('rapidocr_onnxruntime', include_py_files=False)
except Exception as e:
    print(f"[경고] rapidocr 데이터 수집 실패: {e}")
    rapidocr_datas = []

# ── onnxruntime 데이터 & DLL ─────────────────────────────────────────────────
try:
    ort_datas = collect_data_files('onnxruntime')
except Exception:
    ort_datas = []

try:
    ort_bins = collect_dynamic_libs('onnxruntime')
except Exception:
    ort_bins = []

# ── HIKROBOT MVS Runtime DLL 자동 탐색 ──────────────────────────────────────
mvs_bins = []
for _mvs_dir in [
    r'C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64',
    r'C:\Program Files\Common Files\MVS\Runtime\Win64_x64',
    r'C:\Program Files (x86)\MVS\Runtime\Win64_x64',
    r'C:\Program Files\MVS\Runtime\Win64_x64',
]:
    if os.path.isdir(_mvs_dir):
        for _f in os.listdir(_mvs_dir):
            if _f.lower().endswith('.dll'):
                mvs_bins.append((os.path.join(_mvs_dir, _f), '.'))
        print(f"[정보] MVS Runtime DLL {len(mvs_bins)}개 포함: {_mvs_dir}")
        break

# ── MvImport 포함 여부 ───────────────────────────────────────────────────────
mvimport_datas = []
if os.path.isdir('MvImport'):
    mvimport_datas = [('MvImport', 'MvImport')]

# ────────────────────────────────────────────────────────────────────────────
a = Analysis(
    ['KRLee_OCR_RealTime.py'],
    pathex=['.'],
    binaries=mvs_bins + ort_bins,
    datas=mvimport_datas + rapidocr_datas + ort_datas,
    hiddenimports=[
        # rapidocr
        'rapidocr_onnxruntime',
        'rapidocr_onnxruntime.main',
        'rapidocr_onnxruntime.ch_ppocr_det',
        'rapidocr_onnxruntime.ch_ppocr_rec',
        'rapidocr_onnxruntime.ch_ppocr_cls',
        'rapidocr_onnxruntime.utils',
        # onnxruntime
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi._pybind_state',
        # image / numeric
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        # PyQt5 플러그인
        'PyQt5.sip',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'scipy',
        'IPython', 'jupyter', 'notebook',
        'tensorflow', 'torch',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='KRLee_OCR_RealTime',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime*.dll',
        'msvcp*.dll',
        'onnxruntime*.dll',
        'Mv*.dll',
        'Qt5*.dll',
    ],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
