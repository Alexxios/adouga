# -*- mode: python ; coding: utf-8 -*-
import sys

# Platform-specific hidden imports
if sys.platform == 'win32':
    platform_imports = [
        'pynput.keyboard._win32',
        'pynput.mouse._win32',
        'mss.windows',
    ]
    platform_excludes = ['pyobjc', 'Quartz', 'Cocoa']
elif sys.platform == 'darwin':
    platform_imports = [
        'pynput.keyboard._darwin',
        'pynput.mouse._darwin',
        'mss.darwin',
    ]
    platform_excludes = []
else:  # Linux
    platform_imports = [
        'pynput.keyboard._xorg',
        'pynput.mouse._xorg',
        'mss.linux',
    ]
    platform_excludes = ['pyobjc', 'Quartz', 'Cocoa']

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('../ml/models/model.onnx', 'ml/models'),  # bundle ONNX model
    ],
    hiddenimports=[
        'pynput',
        'pynput.keyboard',
        'pynput.mouse',
        'psutil',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'mss',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.figure',
        'GPUtil',
        'setuptools',
        'pkg_resources.extern',
        'numpy',
        'onnxruntime',
    ] + platform_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'yadisk',
        'dotenv',
    ] + platform_excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Adouga',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)
