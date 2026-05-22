@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   KRLee_OCR_RealTime  --  EXE Build Script
echo ============================================================
echo.

:: ── [0] Python 확인 ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 PATH에 등록되어 있지 않습니다.
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo  Python: %%v
echo.

:: ── [1] 의존성 패키지 설치 ───────────────────────────────────────────────────
echo [1/5] 의존성 패키지 설치 중...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [오류] 패키지 설치 실패
    pause & exit /b 1
)
echo       완료.
echo.

:: ── [2] RapidOCR 모델 사전 다운로드 ─────────────────────────────────────────
echo [2/5] RapidOCR 모델 확인 / 다운로드 중...
echo       (모델이 이미 있으면 건너뜁니다)
python -c ^
"from rapidocr_onnxruntime import RapidOCR; import numpy as np; ^
e=RapidOCR(); e(np.zeros((64,200,3),dtype='uint8')); print('  모델 준비 완료 OK')"
if errorlevel 1 (
    echo [경고] RapidOCR 모델 확인 실패 - 빌드는 계속합니다.
)
echo.

:: ── [3] HIKROBOT MVS SDK 탐색 ───────────────────────────────────────────────
echo [3/5] HIKROBOT MVS SDK 탐색 중...

set "SDK_SRC="
if exist "C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport" (
    set "SDK_SRC=C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
) else if exist "C:\Program Files\MVS\Development\Samples\Python\MvImport" (
    set "SDK_SRC=C:\Program Files\MVS\Development\Samples\Python\MvImport"
)

if defined SDK_SRC (
    echo       SDK 발견: !SDK_SRC!
    xcopy /E /I /Y "!SDK_SRC!" "MvImport\" >nul
    echo       MvImport 복사 완료.
) else if exist "MvImport\MvCameraControl_class.py" (
    echo       로컬 MvImport 사용.
) else (
    echo       [경고] MVS SDK 없음 - 카메라 기능 비활성화 상태로 빌드합니다.
)
echo.

:: ── [4] PyInstaller EXE 빌드 ─────────────────────────────────────────────────
echo [4/5] PyInstaller EXE 빌드 중...
echo       (첫 빌드는 수 분 소요될 수 있습니다)
echo.

if exist "dist\KRLee_OCR_RealTime.exe" del /q "dist\KRLee_OCR_RealTime.exe"

pyinstaller KRLee_OCR_RealTime.spec ^
    --distpath "dist" ^
    --workpath "build_tmp" ^
    --noconfirm ^
    --clean

if errorlevel 1 (
    echo.
    echo [오류] EXE 빌드 실패 - 위 오류 메시지를 확인하세요.
    pause & exit /b 1
)
echo.

:: ── [5] 배포 폴더 구성 ───────────────────────────────────────────────────────
echo [5/5] 배포 폴더 구성 중...

:: MVS Runtime DLL → dist\ 복사
set "RUNTIME_DIR="
if exist "C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64" (
    set "RUNTIME_DIR=C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
) else if exist "C:\Program Files\Common Files\MVS\Runtime\Win64_x64" (
    set "RUNTIME_DIR=C:\Program Files\Common Files\MVS\Runtime\Win64_x64"
) else if exist "C:\Program Files (x86)\MVS\Runtime\Win64_x64" (
    set "RUNTIME_DIR=C:\Program Files (x86)\MVS\Runtime\Win64_x64"
) else if exist "C:\Program Files\MVS\Runtime\Win64_x64" (
    set "RUNTIME_DIR=C:\Program Files\MVS\Runtime\Win64_x64"
)

if defined RUNTIME_DIR (
    echo       MVS Runtime DLL 복사: !RUNTIME_DIR!
    copy /Y "!RUNTIME_DIR!\*.dll" "dist\" >nul 2>&1
    echo       완료.
)

:: build_tmp 정리
if exist "build_tmp" (
    echo       임시 빌드 폴더 정리 중...
    rmdir /s /q "build_tmp" >nul 2>&1
)

:: 결과 파일 크기 표시
if exist "dist\KRLee_OCR_RealTime.exe" (
    for %%F in ("dist\KRLee_OCR_RealTime.exe") do (
        set /a "SIZE_MB=%%~zF / 1048576"
        echo.
        echo ============================================================
        echo   빌드 완료!
        echo   실행 파일 : dist\KRLee_OCR_RealTime.exe  (!SIZE_MB! MB)
        echo ============================================================
    )
) else (
    echo [오류] dist\KRLee_OCR_RealTime.exe 가 생성되지 않았습니다.
    pause & exit /b 1
)

echo.
pause
