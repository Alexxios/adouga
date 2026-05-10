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
    ['src/dev_main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('.env', '.'),  # bundle YADISK_TOKEN for the built executable
    ],
    hiddenimports=[
        'pynput',
        'pynput.keyboard',
        'pynput.mouse',
        'psutil',
        'yadisk',
        'yadisk.sessions',
        'yadisk.sessions.requests_session',
        'dotenv',
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
    ] + platform_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'onnxruntime',
        'onnxruntime_gpu',
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
    name='AdougaDev',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)
