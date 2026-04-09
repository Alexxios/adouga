# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['src/dev_main.py'],
    pathex=[],
    binaries=[],
    datas=[],
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
    [],
    exclude_binaries=True,
    name='AdougaDev',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AdougaDev',
)
