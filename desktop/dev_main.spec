# -*- mode: python ; coding: utf-8 -*-

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
        'pynput.keyboard._win32',
        'pynput.mouse',
        'pynput.mouse._win32',
        'psutil',
        'yadisk',
        'dotenv',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'mss',
        'mss.windows',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.figure',
        'GPUtil',
        'setuptools',
        'pkg_resources.extern',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pyobjc',
        'Quartz',
        'Cocoa',
        'onnxruntime',
        'onnxruntime_gpu',
    ],
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
