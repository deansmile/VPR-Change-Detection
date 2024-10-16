# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['image_annotator.py'],
    pathex=['D:/VPR-Change-Detection/annotation/annotation_tool'],
    binaries=[('C:\\Users\\vgohil\\anaconda3\\envs\\vpr\\Library\\bin\\mkl_intel_thread.2.dll', '.'),
              ('C:\\Users\\vgohil\\anaconda3\\envs\\vpr\\Library\\bin\\mkl_core.2.dll', '.')],
    datas=[],
    hiddenimports=['os', 'cv2', 'tkinter', 'PIL.ImageTk', 'PIL.Image', 'cffi'],
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
    a.binaries,
    a.datas,
    [],
    name='image_annotator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
