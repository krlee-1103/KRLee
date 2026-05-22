#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KRLee_OCR_RealTime - HIKROBOT Camera & Image OCR Integrated Application
Real-Time OCR with Live Video & Image File support.
"""

import sys
import os
import time
import cv2
import numpy as np
import tempfile
import json
import subprocess
import shutil
import winreg
from datetime import datetime
from ctypes import c_ubyte

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QMessageBox, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSpinBox, QDoubleSpinBox, QComboBox, QFrame, QSizePolicy,
    QTabWidget, QScrollArea, QToolBar, QFileDialog, QListWidget,
    QTextEdit, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPalette, QPainter

# ── GPU(CUDA) DLL Path Registration & Early Preload ──────────────────────────
_cuda_dll_handles: list = []   # ctypes 핸들을 GC 방지용으로 보관

def _register_cuda_dll_paths():
    """nvidia-* Python 패키지의 CUDA DLL 경로를 PATH 및 add_dll_directory 에 등록."""
    nvidia_dir = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if not os.path.exists(nvidia_dir):
        return
    for root, _dirs, files in os.walk(nvidia_dir):
        if any(f.endswith(".dll") for f in files):
            if root not in os.environ.get("PATH", ""):
                os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(root)
            except (AttributeError, OSError):
                pass

_register_cuda_dll_paths()
# ※ PyQt5 가 임포트된 상태에서 onnxruntime 은 DLL_INIT_FAILED(1114) 로 임포트 불가.
#   GPU 추론은 Qt 없는 별도 프로세스(OcrServerProcess)에서만 가능.
#   서버 프로세스 스크립트(_OCR_SERVER_SCRIPT)에서 ctypes 선행 로드 + CUDA 초기화를 수행한다.

# Add MvImport to sys.path
HIKROBOT_AVAILABLE = False
for _p in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport"),
    r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport",
    r"C:\Program Files\MVS\Development\Samples\Python\MvImport",
]:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from MvCameraControl_class import (
        MvCamera, MV_CC_DEVICE_INFO_LIST, MV_ACCESS_Exclusive,
        MV_USB_DEVICE, MV_GIGE_DEVICE, MVCC_INTVALUE,
        MVCC_FLOATVALUE, MVCC_ENUMVALUE,
        MV_FRAME_OUT_INFO_EX,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_RGB8_Packed,
    )
    HIKROBOT_AVAILABLE = True
except ImportError:
    pass

# ── Helper functions ──────────────────────────────────────────────────────────
def _b2s(arr) -> str:
    return bytes(arr).split(b"\x00")[0].decode("utf-8", "replace").strip()

def get_device_info(dev_info) -> dict:
    tl = dev_info.nTLayerType
    info = {"type": tl, "type_str": "Unknown",
            "vendor": "", "model": "", "serial": "", "version": ""}
    if tl == MV_USB_DEVICE:
        info["type_str"] = "USB3 Vision"
        u = dev_info.SpecialInfo.stUsb3VInfo
        info["vendor"]  = _b2s(u.chManufacturerName) or _b2s(u.chVendorName)
        info["model"]   = _b2s(u.chModelName)
        info["serial"]  = _b2s(u.chSerialNumber)
        info["version"] = _b2s(u.chDeviceVersion)
    elif tl == MV_GIGE_DEVICE:
        info["type_str"] = "GigE Vision"
        g = dev_info.SpecialInfo.stGigEInfo
        info["vendor"]  = _b2s(g.chManufacturerName)
        info["model"]   = _b2s(g.chModelName)
        info["serial"]  = _b2s(g.chSerialNumber)
        info["version"] = _b2s(g.chDeviceVersion)
    return info

def enumerate_cameras():
    if not HIKROBOT_AVAILABLE:
        raise RuntimeError(
            "HIKROBOT MVS SDK를 찾을 수 없습니다.\n"
            "MvImport 패키지를 확인하거나 HIKROBOT MVS가 정상 설치되었는지 확인하세요."
        )
    dl = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, dl)
    if ret != 0:
        raise RuntimeError(f"장치 검색 실패 (코드: 0x{ret:08X})")
    infos = [get_device_info(dl.pDeviceInfo[i].contents) for i in range(dl.nDeviceNum)]
    return dl, infos

def read_roi_constraints(cam) -> dict:
    result = {}
    for name in ("Width", "Height", "OffsetX", "OffsetY"):
        p = MVCC_INTVALUE()
        ret = cam.MV_CC_GetIntValue(name, p)
        if ret == 0:
            result[name] = {
                "cur": p.nCurValue,
                "min": p.nMin,
                "max": p.nMax,
                "inc": max(1, p.nInc),
            }
    if "Width" in result and "OffsetX" in result:
        result["sensor_w"] = result["Width"]["max"] + result["OffsetX"]["cur"]
    if "Height" in result and "OffsetY" in result:
        result["sensor_h"] = result["Height"]["max"] + result["OffsetY"]["cur"]
    return result

def _to_bgr(buf, info):
    try:
        w, h = info.nWidth, info.nHeight
        raw = np.frombuffer(bytes(buf[:info.nFrameLen]), dtype=np.uint8)
        pt = info.enPixelType
        if pt == PixelType_Gvsp_Mono8:
            return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2BGR)
        elif pt == PixelType_Gvsp_BayerRG8:
            return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_BayerRG2BGR)
        elif pt == PixelType_Gvsp_BayerGB8:
            return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_BayerGB2BGR)
        elif pt == PixelType_Gvsp_BayerGR8:
            return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_BayerGR2BGR)
        elif pt == PixelType_Gvsp_BayerBG8:
            return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_BayerBG2BGR)
        elif pt == PixelType_Gvsp_RGB8_Packed:
            return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2BGR)
    except Exception:
        return None

# ── Styling definitions ───────────────────────────────────────────────────────
def _bs(bg, hov, press, dis="#444"):
    return (f"QPushButton{{background:{bg};color:white;border:none;border-radius:6px;"
            f"padding:8px 12px;font-size:14px;font-weight:bold;}}"
            f"QPushButton:hover{{background:{hov};}}"
            f"QPushButton:pressed{{background:{press};}}"
            f"QPushButton:disabled{{background:{dis};color:#777;}}")

S_GREEN  = _bs("#43A047", "#2E7D32", "#1B5E20")
S_GRAY   = _bs("#616161", "#424242", "#212121")
S_BLUE   = _bs("#1E88E5", "#1565C0", "#0D47A1")
S_ORANGE = _bs("#FB8C00", "#E65100", "#BF360C")
S_RED    = _bs("#C62828", "#B71C1C", "#7F0000")
S_PURPLE = _bs("#6A1B9A", "#4A148C", "#38006b")
S_CYAN   = _bs("#00838F", "#006064", "#004D40")

# ── GrabThread ────────────────────────────────────────────────────────────────
class GrabThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera, payload_size: int):
        super().__init__()
        self._camera = camera
        self._payload_size = payload_size
        self._running = False

    def run(self):
        self._running = True
        buf = (c_ubyte * self._payload_size)()
        info = MV_FRAME_OUT_INFO_EX()

        while self._running:
            ret = self._camera.MV_CC_GetOneFrameTimeout(buf, self._payload_size, info, 1000)
            if ret != 0:
                self.msleep(5)
                continue
            frame = _to_bgr(buf, info)
            if frame is not None:
                self.frame_ready.emit(frame)

    def stop(self):
        self._running = False
        if self._camera:
            self._camera.MV_CC_StopGrabbing()
        self.wait(2000)

# ── OcrWorker ────────────────────────────────────────────────────────────────
# ── 상주 OCR 서버 스크립트 ───────────────────────────────────────────────────
# 별도 Python 프로세스에서 실행됨.
# 시작 시 모델을 한 번만 로드하고, 이후 stdin→stdout 파이프로 요청을 처리.
_OCR_SERVER_SCRIPT = r"""
import sys, json, os, cv2

# ① nvidia DLL 경로 등록 (PATH + add_dll_directory)
_nv = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia')
_cuda_handles = []
if os.path.exists(_nv):
    for _r, _, _fs in os.walk(_nv):
        if any(f.endswith('.dll') for f in _fs):
            if _r not in os.environ.get('PATH', ''):
                os.environ['PATH'] = _r + os.pathsep + os.environ.get('PATH', '')
            try:
                os.add_dll_directory(_r)
            except Exception:
                pass

# ② ctypes 선행 로드 — onnxruntime CUDA provider 의존 DLL 을 프로세스 캐시에 등록
#    이 프로세스에는 Qt 가 없으므로 ctypes 로딩 후 onnxruntime 임포트가 정상 동작함
import ctypes as _ct
_key_dlls = [
    'cudart64_12.dll',   'cublas64_12.dll',   'cublasLt64_12.dll',
    'cudnn64_9.dll',     'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll',
    'cufft64_11.dll',    'curand64_10.dll',   'cusolver64_11.dll',
    'cusparse64_12.dll', 'nvrtc64_120_0.dll',
]
if os.path.exists(_nv):
    for _r, _, _fs in os.walk(_nv):
        if any(f.endswith('.dll') for f in _fs):
            for _dn in _key_dlls:
                _dp = os.path.join(_r, _dn)
                if os.path.exists(_dp):
                    try:
                        _cuda_handles.append(_ct.CDLL(_dp))
                    except OSError:
                        pass

# ③ RapidOCR 임포트
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print(json.dumps({'s': 'no_pkg'}), flush=True)
    sys.exit(1)

# ④ CUDA EP 가용 여부 확인
try:
    import onnxruntime as ort
    _cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
except Exception:
    _cuda = False

# ⑤ 엔진 초기화
engine = RapidOCR(det_use_cuda=_cuda, rec_use_cuda=_cuda, cls_use_cuda=_cuda)

# ⑥ 실제 CUDA 사용 여부 검증 (session.get_providers 로 확인)
_actual_cuda = False
if _cuda:
    try:
        _det_prov = engine.text_det.infer.session.get_providers()
        _actual_cuda = 'CUDAExecutionProvider' in _det_prov
    except Exception:
        _actual_cuda = False

print(json.dumps({'s': 'ready', 'cuda': _actual_cuda}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    try:
        img = cv2.imread(req['p'])
        result, _ = engine(img)
        out = []
        if result:
            for item in result:
                bbox = [[int(p[0]), int(p[1])] for p in item[0]]
                conf = float(item[2]) if item[2] is not None else 0.0
                out.append({'b': bbox, 't': item[1], 'c': conf})
        print(json.dumps({'s': 'ok', 'r': out}), flush=True)
    except Exception as e:
        print(json.dumps({'s': 'err', 'm': str(e)}), flush=True)
"""


# ── OcrServerProcess ─────────────────────────────────────────────────────────
class OcrServerProcess:
    """
    상주 OCR 서버 프로세스 래퍼.
    외부 Python 인터프리터에서 _OCR_SERVER_SCRIPT 를 실행한 뒤
    stdin/stdout 파이프로 OCR 요청/응답을 주고받는다.
    모델은 프로세스 시작 시 단 한 번만 로드된다.
    """

    def __init__(self):
        self._proc = None
        self.use_cuda = False   # 서버 시작 후 실제 CUDA 사용 여부

    def start(self, python_cmd: list) -> bool:
        """서버 시작 및 'ready' 응답 대기. 성공 시 True, use_cuda 도 설정."""
        try:
            self._proc = subprocess.Popen(
                python_cmd + ['-c', _OCR_SERVER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
                creationflags=0x08000000,   # CREATE_NO_WINDOW
            )
            line = self._proc.stdout.readline()
            if not line:
                return False
            data = json.loads(line)
            if data.get('s') == 'ready':
                self.use_cuda = bool(data.get('cuda', False))
                return True
            return False
        except Exception:
            return False

    def run_ocr(self, img_path: str) -> list:
        """이미지 경로를 서버로 전송하고 결과 리스트를 반환."""
        req = json.dumps({'p': img_path}) + '\n'
        self._proc.stdin.write(req)
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        data = json.loads(line)
        if data.get('s') != 'ok':
            raise RuntimeError(data.get('m', 'OCR 서버 오류'))
        return [
            ([[p[0], p[1]] for p in item['b']], item['t'], item['c'])
            for item in data.get('r', [])
        ]

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None


# ── OcrWorker ────────────────────────────────────────────────────────────────
class OcrWorker(QThread):
    """
    OCR 실행 워커 – 3단계 속도 최적화

    1순위: 인-프로세스 캐시 엔진  ← rapidocr 가 현재 Python 환경에 설치된 경우
           모델 1회 로드 후 재사용. 추론만 실행 ≈ 0.3~1 초

    2순위: 상주 서버 프로세스      ← 외부 Python 환경에 rapidocr 가 설치된 경우
           서버 프로세스가 모델을 메모리에 유지. stdin/stdout 파이프로 통신 ≈ 0.5~1.5 초

    3순위: 단발 서브프로세스       ← 최후 폴백 (매번 모델 로드 → 느림)
    """

    finished = pyqtSignal(list)          # [(bbox, text, conf), ...]
    error    = pyqtSignal(str)
    status   = pyqtSignal(str)
    timing   = pyqtSignal(float, float)  # (전체 처리시간 ms, 순수 인식 ms)

    # ── 클래스 레벨 공유 상태 ────────────────────────────────────────────────
    _engine        = None    # 1순위: in-process RapidOCR 인스턴스
    _engine_failed = False

    _server        = None    # 2순위: OcrServerProcess 인스턴스
    _server_failed = False

    _use_cuda      = False   # 실제 CUDA EP 사용 여부 (로드 후 확정)
    _gpu_name      = ""      # GPU 이름 (winreg에서 읽음)

    # 긴 변 최대 크기 (픽셀) – 초과 시 비율 유지 축소 후 추론
    MAX_SIDE = 2560

    # ── 엔진/서버 초기화 (백그라운드 스레드에서 호출) ────────────────────────
    @classmethod
    def _detect_gpu_name(cls):
        """레지스트리에서 NVIDIA GPU 이름을 읽어 반환."""
        try:
            for slot in ('0000', '0001', '0002', '0003'):
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    fr'SYSTEM\CurrentControlSet\Control\Class'
                    fr'\{{4d36e968-e325-11ce-bfc1-08002be10318}}\{slot}'
                )
                desc = winreg.QueryValueEx(key, 'DriverDesc')[0]
                winreg.CloseKey(key)
                if 'NVIDIA' in desc or 'GeForce' in desc or 'Quadro' in desc or 'RTX' in desc:
                    return desc
        except Exception:
            pass
        return ""

    @classmethod
    def load_engine(cls):
        """1순위 in-process 시도 → 실패 시 2순위 서버 기동."""
        # 진단 로그 파일 (앱 디렉터리에 생성)
        _log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cuda_init.log"
        )

        def _log(msg: str):
            try:
                with open(_log_path, "a", encoding="utf-8") as _f:
                    _f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            except Exception:
                pass

        # 1순위: 현재 Python 환경에서 직접 임포트
        if not cls._engine_failed and cls._engine is None:
            try:
                # onnxruntime 과 CUDA DLL 은 이미 모듈 레벨(메인 스레드)에서
                # 임포트/선행 로드 완료. 여기서는 RapidOCR 만 임포트하면 된다.
                _log(f"preloaded cuda handles={len(_cuda_dll_handles)}")
                from rapidocr_onnxruntime import RapidOCR

                # CUDA EP 가용 여부 확인
                try:
                    import onnxruntime as ort
                    avail_providers = ort.get_available_providers()
                    use_cuda = 'CUDAExecutionProvider' in avail_providers
                    _log(f"ort.get_device()={ort.get_device()}  "
                         f"providers={avail_providers}  use_cuda={use_cuda}")
                except Exception as _ep_err:
                    use_cuda = False
                    _log(f"get_available_providers 실패: {_ep_err}")

                # ── CUDA 모드 우선 시도 ──────────────────────────────────────
                if use_cuda:
                    try:
                        cls._engine = RapidOCR(
                            det_use_cuda=True,
                            rec_use_cuda=True,
                            cls_use_cuda=True,
                        )
                        # ── 실제 CUDA 사용 여부 검증 ─────────────────────────
                        # OrtInferSession.use_cuda 는 세션 생성 전에 설정되므로
                        # 내부 폴백이 발생해도 True 로 남을 수 있다.
                        # 실제 사용 EP 는 session.get_providers() 로 확인해야 한다.
                        try:
                            det_prov = cls._engine.text_det.infer.session.get_providers()
                            actual_cuda = "CUDAExecutionProvider" in det_prov
                            _log(f"RapidOCR 초기화 완료  "
                                 f"text_det session_providers={det_prov}  "
                                 f"actual_cuda={actual_cuda}")
                        except Exception as _ver_err:
                            # 속성 접근 실패 시 infer.use_cuda 로 차선 확인
                            try:
                                actual_cuda = bool(cls._engine.text_det.infer.use_cuda)
                            except Exception:
                                actual_cuda = True   # 알 수 없으면 낙관적으로 True
                            _log(f"CUDA 검증 중 예외(차선 use_cuda={actual_cuda}): {_ver_err}")

                        cls._use_cuda = actual_cuda
                        if actual_cuda:
                            cls._gpu_name = cls._detect_gpu_name()
                            return   # GPU 모드 성공
                        else:
                            # 엔진은 이미 (CPU 로) 초기화됐으므로 그대로 사용
                            _log("CUDA 요청했으나 실제 CPU 동작 중 "
                                 "(config.yaml 또는 라이브러리 문제)")
                            return   # CPU 폴백 – 엔진은 살아있음

                    except Exception as cuda_err:
                        # CUDA 초기화 자체가 예외 → CPU 폴백
                        cls._engine = None
                        cls._use_cuda = False
                        import traceback as _tb
                        _log(f"CUDA 초기화 예외:\n{_tb.format_exc()}")

                # ── CPU 모드 폴백 ────────────────────────────────────────────
                _log("CPU 모드로 엔진 초기화")
                cls._engine = RapidOCR(
                    det_use_cuda=False,
                    rec_use_cuda=False,
                    cls_use_cuda=False,
                )
                cls._use_cuda = False
                return   # CPU 모드 성공

            except Exception:
                import traceback as _tb
                _log(f"엔진 로드 치명적 오류:\n{_tb.format_exc()}")
                cls._engine_failed = True

        # 2순위: 외부 Python 인터프리터에서 상주 서버 기동
        if cls._engine is None and not cls._server_failed and cls._server is None:
            cmd = cls._find_python()
            if cmd:
                srv = OcrServerProcess()
                if srv.start(cmd):
                    cls._server = srv
                    # 서버가 보고한 실제 CUDA 사용 여부를 클래스 상태에 반영
                    cls._use_cuda = srv.use_cuda
                    if srv.use_cuda:
                        cls._gpu_name = cls._detect_gpu_name()
                else:
                    cls._server_failed = True
            else:
                cls._server_failed = True

    # ── 인스턴스 ─────────────────────────────────────────────────────────────
    def __init__(self, image: np.ndarray):
        super().__init__()
        self._image = image.copy()
        self._recog_ms = 0.0  # 순수 인식(추론) 소요 시간 ms

    @classmethod
    def _preprocess(cls, img: np.ndarray) -> np.ndarray:
        """긴 변이 MAX_SIDE 초과 시 비율 유지 축소."""
        h, w = img.shape[:2]
        if max(h, w) <= cls.MAX_SIDE:
            return img
        scale = cls.MAX_SIDE / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    # ── 실행 진입점 ──────────────────────────────────────────────────────────
    def run(self):
        self.load_engine()  # 아직 준비 안 됐으면 이 호출에서 완료
        # ↑ load_engine은 최초 1회만 오래 걸리고 이후에는 즉시 반환.
        #   모델 로딩 시간은 OCR 처리 시간에 포함하지 않는다.
        _t_total = time.perf_counter()   # ← 엔진 준비 완료 후부터 측정 시작

        if self._engine is not None:
            self._run_inprocess()
        elif self._server is not None and self._server.is_alive():
            self._run_via_server()
        else:
            self._run_subprocess()   # 3순위 폴백

        total_ms = (time.perf_counter() - _t_total) * 1000
        # total_ms 는 항상 _recog_ms 를 포함하므로 total >= recog 가 보장됨
        self.timing.emit(total_ms, self._recog_ms)

    # ── 1순위: 인-프로세스 ───────────────────────────────────────────────────
    def _run_inprocess(self):
        try:
            self.status.emit("문자 영역 탐지 및 인식 중 …")
            img = self._preprocess(self._image)   # 전처리: 타이밍 범위 밖
            _t0 = time.perf_counter()             # ← 전처리 완료 후 순수 추론 측정 시작
            result, _ = self._engine(img)
            self._recog_ms = (time.perf_counter() - _t0) * 1000
            results = []
            if result:
                for item in result:
                    bbox = [[int(p[0]), int(p[1])] for p in item[0]]
                    conf = float(item[2]) if item[2] is not None else 0.0
                    results.append((bbox, item[1], conf))
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(f"OCR 처리 중 오류가 발생했습니다:\n{e}")

    # ── 2순위: 상주 서버 프로세스 ────────────────────────────────────────────
    def _run_via_server(self):
        tmp_img = None
        try:
            self.status.emit("문자 영역 탐지 및 인식 중 …")
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                tmp_img = f.name
            img = self._preprocess(self._image)   # 전처리: 타이밍 범위 밖
            cv2.imwrite(tmp_img, img)              # 파일 쓰기: 타이밍 범위 밖
            _t0 = time.perf_counter()             # ← 서버 파이프 I/O + 추론 측정 시작
            results = self._server.run_ocr(tmp_img)
            self._recog_ms = (time.perf_counter() - _t0) * 1000
            self.finished.emit(results)
        except Exception as e:
            # 서버 프로세스가 비정상 종료된 경우 다음 호출에서 재시작 허용
            OcrWorker._server = None
            OcrWorker._server_failed = False
            self.error.emit(f"OCR 서버 오류:\n{e}\n\n다시 시도하면 서버가 재시작됩니다.")
        finally:
            if tmp_img and os.path.exists(tmp_img):
                try:
                    os.unlink(tmp_img)
                except OSError:
                    pass

    # ── 3순위: 단발 서브프로세스 (폴백) ──────────────────────────────────────
    @staticmethod
    def _find_python():
        py = shutil.which('py')
        if py:
            return [py, '-3']
        for name in ('python', 'python3'):
            p = shutil.which(name)
            if p and 'WindowsApps' not in p:
                return [p]
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for sub in (r'SOFTWARE\Python\PythonCore',
                        r'SOFTWARE\WOW6432Node\Python\PythonCore'):
                try:
                    with winreg.OpenKey(hive, sub) as k:
                        for i in range(winreg.QueryInfoKey(k)[0]):
                            ver = winreg.EnumKey(k, i)
                            try:
                                with winreg.OpenKey(k, ver + r'\InstallPath') as ip:
                                    base = winreg.QueryValueEx(ip, '')[0].rstrip('\\')
                                    exe  = os.path.join(base, 'python.exe')
                                    if os.path.isfile(exe):
                                        return [exe]
                            except OSError:
                                pass
                except OSError:
                    pass
        for base in (
            os.path.expanduser(r'~\AppData\Local\Programs\Python\Python312'),
            os.path.expanduser(r'~\AppData\Local\Programs\Python\Python311'),
            os.path.expanduser(r'~\AppData\Local\Programs\Python\Python310'),
            r'C:\Python312', r'C:\Python311', r'C:\Python310',
        ):
            exe = os.path.join(base, 'python.exe')
            if os.path.isfile(exe):
                return [exe]
        return None

    def _run_subprocess(self):
        tmp_img = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                tmp_img = f.name
            cv2.imwrite(tmp_img, self._preprocess(self._image))
            self.status.emit("RapidOCR 엔진 로드 중 … (단발 서브프로세스 – 느림)")

            script = (
                "import sys, json, cv2, os\n"
                "try:\n"
                "    from rapidocr_onnxruntime import RapidOCR\n"
                "except ImportError:\n"
                "    print(json.dumps({'err':'no_rapidocr'}))\n"
                "    sys.exit(0)\n"
                "_nv = os.path.join(sys.prefix,'Lib','site-packages','nvidia')\n"
                "if os.path.exists(_nv):\n"
                "    for _r,_,_fs in os.walk(_nv):\n"
                "        if any(f.endswith('.dll') for f in _fs) and _r not in os.environ.get('PATH',''):\n"
                "            os.environ['PATH']=_r+os.pathsep+os.environ.get('PATH','')\n"
                "try:\n"
                "    import onnxruntime as ort\n"
                "    _use_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()\n"
                "except Exception:\n"
                "    _use_cuda = False\n"
                "engine = RapidOCR(det_use_cuda=_use_cuda,rec_use_cuda=_use_cuda,cls_use_cuda=_use_cuda)\n"
                f"img = cv2.imread({tmp_img!r})\n"
                "result, _ = engine(img)\n"
                "out = []\n"
                "if result:\n"
                "    for item in result:\n"
                "        bbox=[[int(p[0]),int(p[1])] for p in item[0]]\n"
                "        conf=float(item[2]) if item[2] is not None else 0.0\n"
                "        out.append({'bbox':bbox,'text':item[1],'conf':conf})\n"
                "print(json.dumps({'ok':out}))\n"
            )
            cmd = self._find_python()
            if cmd is None:
                self.error.emit(
                    "시스템에 설치된 Python을 찾을 수 없습니다.\n\n"
                    "    pip install rapidocr-onnxruntime"
                )
                return
            self.status.emit("문자 영역 탐지 및 인식 중 …")
            _t0 = time.perf_counter()
            proc = subprocess.run(
                cmd + ['-c', script],
                capture_output=True, text=True, timeout=120,
                creationflags=0x08000000
            )
            self._recog_ms = (time.perf_counter() - _t0) * 1000
            stdout = proc.stdout.strip()
            if not stdout:
                self.error.emit(f"OCR 오류:\n{proc.stderr.strip() or '알 수 없는 오류'}")
                return
            data = json.loads(stdout.splitlines()[-1])
            if data.get('err') == 'no_rapidocr':
                self.error.emit(
                    "RapidOCR가 설치되어 있지 않습니다.\n\n"
                    "    pip install rapidocr-onnxruntime"
                )
                return
            results = [
                ([[p[0], p[1]] for p in item['bbox']], item['text'], item['conf'])
                for item in data.get('ok', [])
            ]
            self.finished.emit(results)
        except subprocess.TimeoutExpired:
            self.error.emit("OCR 시간 초과.\n첫 실행 시 모델 다운로드 때문일 수 있습니다.")
        except Exception as e:
            self.error.emit(f"OCR 처리 중 오류:\n{e}")
        finally:
            if tmp_img and os.path.exists(tmp_img):
                try:
                    os.unlink(tmp_img)
                except OSError:
                    pass


# ── EnginePrewarmThread ───────────────────────────────────────────────────────
class EnginePrewarmThread(QThread):
    """앱 시작과 동시에 백그라운드에서 OCR 엔진 / 서버를 미리 준비."""
    status_msg  = pyqtSignal(str)
    cuda_ready  = pyqtSignal(bool, str)   # (use_cuda, gpu_name)

    def run(self):
        self.status_msg.emit("OCR 엔진 초기화 중 …")
        OcrWorker.load_engine()

        if OcrWorker._engine is not None:
            if OcrWorker._use_cuda:
                gpu = OcrWorker._gpu_name or "NVIDIA GPU"
                self.status_msg.emit(f"OCR 엔진 준비 완료  [GPU: {gpu}]")
            else:
                # CUDA EP는 있었지만 초기화 실패 → CPU 폴백인지 원래 CPU인지 확인
                try:
                    import onnxruntime as ort
                    has_cuda_ep = 'CUDAExecutionProvider' in ort.get_available_providers()
                except Exception:
                    has_cuda_ep = False
                if has_cuda_ep:
                    self.status_msg.emit(
                        "OCR 엔진 준비 완료  [CPU 모드]  "
                        "※ CUDA EP 감지됐으나 초기화 실패 — stderr 로그 확인"
                    )
                else:
                    self.status_msg.emit("OCR 엔진 준비 완료  [CPU 모드]")
        elif OcrWorker._server is not None:
            cuda_str = " (CUDA)" if OcrWorker._use_cuda else ""
            self.status_msg.emit(f"OCR 엔진 준비 완료  [상주 서버{cuda_str}]")
        else:
            self.status_msg.emit("OCR 준비 완료  [단발 서브프로세스 – 느림]")

        self.cuda_ready.emit(OcrWorker._use_cuda, OcrWorker._gpu_name)

# ── ZoomableImageLabel ────────────────────────────────────────────────────────
class ZoomableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._src_img = None
        self._zoom = 1.0
        self._offset = [0, 0]
        self._drag_pos = None
        self._zoom_min = 0.05
        self._zoom_max = 20.0
        self._highlighted_bbox = None  # bboxes for list selection highlight

    def set_image(self, img: np.ndarray):
        self._src_img = img
        self._fit()

    def set_highlighted_bbox(self, bbox):
        self._highlighted_bbox = bbox
        self._render()

    def _fit(self):
        if self._src_img is None:
            return
        h, w = self._src_img.shape[:2]
        lw = max(self.width(), 1)
        lh = max(self.height(), 1)
        self._zoom = min(lw / w, lh / h)
        self._offset = [0, 0]
        self._render()

    def _render(self):
        if self._src_img is None:
            return
        h, w = self._src_img.shape[:2]
        nw = max(1, int(w * self._zoom))
        nh = max(1, int(h * self._zoom))
        lw, lh = max(self.width(), 1), max(self.height(), 1)

        ox = max(-(nw - 20), min(lw - 20, self._offset[0]))
        oy = max(-(nh - 20), min(lh - 20, self._offset[1]))
        self._offset = [ox, oy]

        interp = cv2.INTER_LINEAR if self._zoom >= 1.0 else cv2.INTER_AREA
        resized = cv2.resize(self._src_img, (nw, nh), interpolation=interp)

        # Highlight overlay if selected
        if self._highlighted_bbox is not None:
            # draw highlight box scaled to resized frame size
            pts = np.array([
                [int(p[0] * self._zoom), int(p[1] * self._zoom)] for p in self._highlighted_bbox
            ], dtype=np.int32)
            cv2.polylines(resized, [pts], True, (0, 255, 255), 3)  # thick yellow box

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qi.copy())

        canvas = QPixmap(lw, lh)
        canvas.fill(QColor("#0d0d0d"))
        painter = QPainter(canvas)
        cx = (lw - nw) // 2 + ox
        cy = (lh - nh) // 2 + oy
        painter.drawPixmap(cx, cy, pix)
        painter.end()
        self.setPixmap(canvas)

    def wheelEvent(self, event):
        if self._src_img is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(self._zoom_min, min(self._zoom_max, self._zoom * factor))

        pos = event.pos()
        lw, lh = max(self.width(), 1), max(self.height(), 1)
        h, w = self._src_img.shape[:2]
        nw_old = max(1, int(w * self._zoom))
        nh_old = max(1, int(h * self._zoom))
        cx_old = (lw - nw_old) // 2 + self._offset[0]
        cy_old = (lh - nh_old) // 2 + self._offset[1]
        img_x = (pos.x() - cx_old) / self._zoom
        img_y = (pos.y() - cy_old) / self._zoom

        self._zoom = new_zoom
        nw_new = max(1, int(w * self._zoom))
        nh_new = max(1, int(h * self._zoom))
        self._offset[0] = int(pos.x() - ((lw - nw_new) // 2 + img_x * self._zoom))
        self._offset[1] = int(pos.y() - ((lh - nh_new) // 2 + img_y * self._zoom))
        self._render()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            diff = event.pos() - self._drag_pos
            self._offset[0] += diff.x()
            self._offset[1] += diff.y()
            self._drag_pos = event.pos()
            self._render()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self._fit()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

# ── CameraSelectDialog ────────────────────────────────────────────────────────
class CameraSelectDialog(QDialog):
    def __init__(self, parent=None, allow_cancel=True):
        super().__init__(parent)
        self.setWindowTitle("카메라 선택")
        self.setMinimumSize(760, 310)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._selected_index  = -1
        self._device_list_obj = None
        self._device_infos    = []

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 14)

        top = QHBoxLayout()
        title = QLabel("연결된 카메라 목록")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet("color:#aaa;")
        btn_ref = QPushButton("🔄")
        btn_ref.setToolTip("새로고침")
        btn_ref.setFixedWidth(44)
        btn_ref.setStyleSheet(
            "QPushButton{background:#455A64;color:white;border:none;"
            "border-radius:4px;padding:5px 10px;font-size:15px;}"
            "QPushButton:hover{background:#37474F;}"
        )
        btn_ref.clicked.connect(self._refresh)
        top.addWidget(title); top.addWidget(self._lbl_count)
        top.addStretch(); top.addWidget(btn_ref)
        lay.addLayout(top)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["No.", "제조사", "모델명", "시리얼 번호", "연결 방식", "펌웨어 버전"]
        )
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        for col, w in ((0, 40), (1, 90), (3, 120), (4, 110), (5, 190)):
            self._table.setColumnWidth(col, w)
        self._table.setStyleSheet(
            "QTableWidget{background:#1e1e1e;color:#ddd;gridline-color:#333;"
            "border:1px solid #444;}"
            "QTableWidget::item:selected{background:#1565C0;}"
            "QHeaderView::section{background:#2d2d2d;color:#ccc;"
            "padding:6px;border:none;border-bottom:1px solid #444;}"
        )
        self._table.doubleClicked.connect(self._accept)
        self._table.itemSelectionChanged.connect(
            lambda: self._btn_ok.setEnabled(
                bool(self._table.selectionModel().selectedRows())))
        lay.addWidget(self._table)

        hint = QLabel("더블클릭하거나 선택 후 [카메라 연결] 버튼을 누르세요.")
        hint.setStyleSheet("color:#888;font-size:11px;")
        lay.addWidget(hint)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#444;"); lay.addWidget(sep)

        btn_box = QHBoxLayout()
        self._btn_ok = QPushButton("🔌 연결")
        self._btn_ok.setToolTip("Connect Camera")
        self._btn_ok.setFixedHeight(38)
        self._btn_ok.setStyleSheet(
            "QPushButton{background:#1E88E5;color:white;border:none;"
            "border-radius:5px;font-size:14px;font-weight:bold;padding:6px 16px;}"
            "QPushButton:hover{background:#1565C0;}"
            "QPushButton:disabled{background:#555;color:#888;}"
        )
        self._btn_ok.clicked.connect(self._accept)
        self._btn_ok.setEnabled(False)
        
        btn_cancel = QPushButton("취소")
        btn_cancel.setFixedHeight(38)
        btn_cancel.setStyleSheet(
            "QPushButton{background:#424242;color:#ccc;border:none;"
            "border-radius:5px;font-size:14px;padding:6px 16px;}"
            "QPushButton:hover{background:#303030;color:white;}"
        )
        btn_cancel.clicked.connect(self.reject)
        
        btn_box.addStretch()
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(self._btn_ok)
        lay.addLayout(btn_box)

        self._refresh()

    def _refresh(self):
        self._table.setRowCount(0)
        self._device_infos = []; self._device_list_obj = None
        self._btn_ok.setEnabled(False)
        try:
            dl, infos = enumerate_cameras()
        except RuntimeError as e:
            self._lbl_count.setText("SDK 오류")
            QMessageBox.critical(self, "SDK 오류", str(e)); return
        self._device_list_obj = dl; self._device_infos = infos
        self._lbl_count.setText(f"({len(infos)}대 발견)")
        for i, info in enumerate(infos):
            self._table.insertRow(i)
            for col, text in enumerate([
                str(i + 1), info["vendor"] or "Hikrobot",
                info["model"] or "Unknown", info["serial"] or "-",
                info["type_str"], info["version"] or "-",
            ]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(i, col, item)
        if not infos:
            self._lbl_count.setText("(연결된 카메라 없음)")
        elif len(infos) == 1:
            self._table.selectRow(0)

    def _accept(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows: return
        self._selected_index = rows[0].row(); self.accept()

    @property
    def selected_index(self): return self._selected_index
    @property
    def selected_info(self):
        if 0 <= self._selected_index < len(self._device_infos):
            return self._device_infos[self._selected_index]
        return None
    @property
    def device_list(self): return self._device_list_obj

# ── ROIPanel ─────────────────────────────────────────────────────────────────
class ROIPanel(QWidget):
    apply_requested   = pyqtSignal(int, int, int, int)
    restore_requested = pyqtSignal()

    _FIELDS = [
        ("OffsetX", "Offset X"),
        ("OffsetY", "Offset Y"),
        ("Width",   "Width"),
        ("Height",  "Height"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sensor_w = 0
        self._sensor_h = 0
        self._inc_ox   = 1
        self._inc_oy   = 1
        self._inc_w    = 1
        self._inc_h    = 1
        self._spins: dict = {}
        self._build_ui()
        self.set_inactive()

    def _build_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(0)
        vbox.setContentsMargins(10, 10, 10, 10)

        for name, label in self._FIELDS:
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#d0d0d0;font-size:12px;font-weight:bold;margin-top:8px;margin-bottom:2px;")
            vbox.addWidget(lbl)

            row = QHBoxLayout(); row.setSpacing(6)
            spin = QSpinBox()
            spin.setRange(0, 99999)
            spin.setValue(0)
            spin.setFixedHeight(28)
            spin.setStyleSheet(
                "QSpinBox{background:#252525;color:#eee;border:1px solid #4a4a4a;"
                "border-radius:3px;padding:2px 4px;font-size:12px;}"
                "QSpinBox::up-button,QSpinBox::down-button{width:16px;background:#353535;border:none;}"
                "QSpinBox:disabled{color:#505050;border-color:#333;background:#1e1e1e;}"
            )
            rlbl = QLabel("-")
            rlbl.setStyleSheet("color:#ffffff;font-size:11px;font-weight:bold;min-width:72px;")
            rlbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(spin, stretch=1)
            row.addWidget(rlbl)
            vbox.addLayout(row)
            self._spins[name] = (spin, rlbl)

        self._spins["Width"] [0].valueChanged.connect(self._update_offset_ranges)
        self._spins["Height"][0].valueChanged.connect(self._update_offset_ranges)
        vbox.addSpacing(14)

        self._btn_apply = QPushButton("Apply ROI")
        self._btn_apply.setFixedHeight(34)
        self._btn_apply.setStyleSheet(S_BLUE)
        self._btn_apply.clicked.connect(self._on_apply)
        vbox.addWidget(self._btn_apply)
        vbox.addSpacing(6)

        self._btn_restore = QPushButton("Restore Max. ROI")
        self._btn_restore.setFixedHeight(34)
        self._btn_restore.setStyleSheet(S_GRAY)
        self._btn_restore.clicked.connect(self.restore_requested.emit)
        vbox.addWidget(self._btn_restore)
        vbox.addSpacing(12)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background:#2a2a2a;max-height:1px;")
        vbox.addWidget(sep2)
        vbox.addSpacing(8)

        cur_title = QLabel("Current ROI")
        cur_title.setStyleSheet("color:#aaa;font-size:11px;font-weight:bold;margin-bottom:4px;")
        vbox.addWidget(cur_title)

        self._lbl_cur = QLabel("–")
        self._lbl_cur.setStyleSheet(
            "color:#ffffff;font-size:13px;font-weight:bold;background:#181818;"
            "border-radius:4px;padding:8px;line-height:160%;"
        )
        self._lbl_cur.setWordWrap(True)
        self._lbl_cur.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        vbox.addWidget(self._lbl_cur)

        self._lbl_sensor = QLabel("")
        self._lbl_sensor.setStyleSheet("color:#e0e0e0;font-size:11px;font-weight:bold;margin-top:4px;")
        vbox.addWidget(self._lbl_sensor)
        vbox.addStretch()

    def update_constraints(self, constraints: dict):
        self._sensor_w = constraints.get("sensor_w", 0)
        self._sensor_h = constraints.get("sensor_h", 0)

        self._inc_ox = constraints.get("OffsetX", {}).get("inc", 1)
        self._inc_oy = constraints.get("OffsetY", {}).get("inc", 1)
        self._inc_w  = constraints.get("Width",   {}).get("inc", 1)
        self._inc_h  = constraints.get("Height",  {}).get("inc", 1)

        for name in ("Width", "Height", "OffsetX", "OffsetY"):
            if name not in constraints:
                continue
            c = constraints[name]
            spin, rlbl = self._spins[name]
            spin.blockSignals(True)
            spin.setRange(c["min"], c["max"])
            spin.setSingleStep(c["inc"])
            spin.setValue(c["cur"])
            spin.blockSignals(False)

        self._update_offset_ranges()
        self._btn_apply.setEnabled(True)
        self._btn_restore.setEnabled(True)
        for name in self._spins:
            self._spins[name][0].setEnabled(True)
        self._refresh_cur_label()

        if self._sensor_w and self._sensor_h:
            self._lbl_sensor.setText(f"Sensor: {self._sensor_w} × {self._sensor_h}")
        else:
            self._lbl_sensor.setText("")

    def _update_offset_ranges(self):
        if self._sensor_w == 0 or self._sensor_h == 0:
            return
        w = self._spins["Width"] [0].value()
        h = self._spins["Height"][0].value()

        max_ox = max(0, self._sensor_w - w)
        max_oy = max(0, self._sensor_h - h)

        ox_spin, ox_rlbl = self._spins["OffsetX"]
        oy_spin, oy_rlbl = self._spins["OffsetY"]

        ox_spin.blockSignals(True)
        ox_spin.setRange(0, max_ox)
        ox_spin.setSingleStep(self._inc_ox)
        if ox_spin.value() > max_ox:
            ox_spin.setValue(max_ox)
        ox_spin.blockSignals(False)

        oy_spin.blockSignals(True)
        oy_spin.setRange(0, max_oy)
        oy_spin.setSingleStep(self._inc_oy)
        if oy_spin.value() > max_oy:
            oy_spin.setValue(max_oy)
        oy_spin.blockSignals(False)

        ox_rlbl.setText(f"0 ~ {max_ox}")
        oy_rlbl.setText(f"0 ~ {max_oy}")

        self._spins["Width"] [1].setText(f"max: {self._sensor_w}")
        self._spins["Height"][1].setText(f"max: {self._sensor_h}")

    def _refresh_cur_label(self):
        ox = self._spins["OffsetX"][0].value()
        oy = self._spins["OffsetY"][0].value()
        w  = self._spins["Width"] [0].value()
        h  = self._spins["Height"][0].value()
        self._lbl_cur.setText(
            f"Offset X : {ox}\nOffset Y : {oy}\nWidth    : {w}\nHeight   : {h}"
        )

    def set_inactive(self):
        self._btn_apply.setEnabled(False)
        self._btn_restore.setEnabled(False)
        for name, (spin, rlbl) in self._spins.items():
            spin.setEnabled(False)
            rlbl.setText("-")
        self._lbl_cur.setText("–")
        self._lbl_sensor.setText("")

    def freeze(self, message: str = "취득 중 변경 불가"):
        self._btn_apply.setEnabled(False)
        self._btn_restore.setEnabled(False)
        for name, (spin, _) in self._spins.items():
            spin.setEnabled(False)
        self._lbl_cur.setText(self._lbl_cur.text() + f"\n\n⚠ {message}")

    def _on_apply(self):
        ox = self._spins["OffsetX"][0].value()
        oy = self._spins["OffsetY"][0].value()
        w  = self._spins["Width"] [0].value()
        h  = self._spins["Height"][0].value()
        self._refresh_cur_label()
        self.apply_requested.emit(ox, oy, w, h)

# ── CameraSettingsPanel ────────────────────────────────────────────────────────
class CameraSettingsPanel(QWidget):
    _EXP_AUTO_LABELS  = ["Off", "Once", "Continuous"]
    _GAIN_AUTO_LABELS = ["Off", "Once", "Continuous"]
    _TRIGGER_LABELS   = ["Off", "On"]
    _USERSET_LABELS   = ["Default", "UserSet1", "UserSet2", "UserSet3"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._camera = None
        self._acquiring = False
        self._build_ui()
        self._refresh_enabled_states()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")

        inner_widget = QWidget()
        inner_widget.setStyleSheet("background:transparent;")
        vbox = QVBoxLayout(inner_widget)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 6, 0)

        self._add_section_label(vbox, "Exposure")
        self._combo_exp_auto = self._add_combo(vbox, "Exposure Auto", self._EXP_AUTO_LABELS)
        self._combo_exp_auto.currentIndexChanged.connect(self._on_exp_auto_changed)

        self._spin_exp = self._add_double_spin(vbox, "Exposure Time (us)", 15.0, 9999500.0)
        self._spin_exp.editingFinished.connect(self._on_exp_time_changed)

        vbox.addSpacing(10)
        self._add_hsep(vbox)
        vbox.addSpacing(6)

        self._add_section_label(vbox, "Gain")
        self._combo_gain_auto = self._add_combo(vbox, "Gain Auto", self._GAIN_AUTO_LABELS)
        self._combo_gain_auto.currentIndexChanged.connect(self._on_gain_auto_changed)

        self._spin_gain = self._add_double_spin(vbox, "Gain (dB)", 0.0, 24.0, step=0.01)
        self._spin_gain.editingFinished.connect(self._on_gain_changed)

        vbox.addSpacing(10)
        self._add_hsep(vbox)
        vbox.addSpacing(6)

        self._add_section_label(vbox, "Trigger")
        self._combo_trigger = self._add_combo(vbox, "Trigger Mode", self._TRIGGER_LABELS)
        self._combo_trigger.currentIndexChanged.connect(self._on_trigger_changed)

        vbox.addSpacing(10)
        self._add_hsep(vbox)
        vbox.addSpacing(6)

        self._add_section_label(vbox, "User Set")
        self._combo_userset = self._add_combo(vbox, "User Set Selector", self._USERSET_LABELS)
        self._combo_userset.currentIndexChanged.connect(self._on_userset_changed)

        note = QLabel("선택 시 로드됨  |  영상 취득 중 사용 불가")
        note.setStyleSheet("color:#666;font-size:10px;margin-top:2px;")
        vbox.addWidget(note)
        vbox.addStretch()

        scroll.setWidget(inner_widget)
        outer.addWidget(scroll)

    def _add_section_label(self, layout, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#f0c070;font-size:11px;font-weight:bold;margin-top:6px;margin-bottom:4px;")
        layout.addWidget(lbl)

    def _add_hsep(self, layout):
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background:#2a2a2a;max-height:1px;")
        layout.addWidget(sep)

    def _add_combo(self, layout, label_text, items):
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color:#b0b0b0;font-size:12px;margin-bottom:2px;")
        layout.addWidget(lbl)
        combo = QComboBox()
        combo.addItems(items)
        combo.setFixedHeight(28)
        combo.setStyleSheet(
            "QComboBox{background:#252525;color:#eee;border:1px solid #4a4a4a;"
            "border-radius:3px;padding:2px 8px;font-size:12px;}"
            "QComboBox::drop-down{border:none;width:20px;}"
            "QComboBox:disabled{color:#505050;border-color:#333;background:#1e1e1e;}"
            "QComboBox QAbstractItemView{background:#2a2a2a;color:#eee;"
            "selection-background-color:#1565C0;border:1px solid #4a4a4a;}"
        )
        layout.addWidget(combo)
        layout.addSpacing(6)
        return combo

    def _add_double_spin(self, layout, label_text, min_val, max_val, step=1.0):
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color:#b0b0b0;font-size:12px;margin-bottom:2px;")
        layout.addWidget(lbl)
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setFixedHeight(28)
        spin.setStyleSheet(
            "QDoubleSpinBox{background:#252525;color:#eee;border:1px solid #4a4a4a;"
            "border-radius:3px;padding:2px 4px;font-size:12px;}"
            "QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{width:16px;background:#353535;border:none;}"
            "QDoubleSpinBox:disabled{color:#505050;border-color:#333;background:#1e1e1e;}"
        )
        layout.addWidget(spin)
        layout.addSpacing(6)
        return spin

    def set_camera(self, cam):
        self._camera = cam
        if cam is not None:
            self._load_from_camera()
        self._refresh_enabled_states()

    def on_acquiring(self, acquiring: bool):
        self._acquiring = acquiring
        self._refresh_enabled_states()

    def _load_from_camera(self):
        self._load_enum_combo("ExposureAuto",    self._combo_exp_auto)
        self._load_float_spin("ExposureTime",    self._spin_exp)
        self._load_enum_combo("GainAuto",        self._combo_gain_auto)
        self._load_float_spin("Gain",            self._spin_gain)
        self._load_enum_combo("TriggerMode",     self._combo_trigger)
        self._load_enum_combo("UserSetSelector", self._combo_userset)

    def _load_float_spin(self, name, spin):
        if not HIKROBOT_AVAILABLE or not self._camera:
            return
        try:
            p = MVCC_FLOATVALUE()
            if self._camera.MV_CC_GetFloatValue(name, p) == 0:
                spin.blockSignals(True)
                spin.setRange(float(p.fMin), float(p.fMax))
                spin.setValue(float(p.fCurValue))
                spin.blockSignals(False)
        except Exception:
            pass

    def _load_enum_combo(self, name, combo):
        if not HIKROBOT_AVAILABLE or not self._camera:
            return
        try:
            p = MVCC_ENUMVALUE()
            if self._camera.MV_CC_GetEnumValue(name, p) == 0:
                combo.blockSignals(True)
                idx = int(p.nCurValue)
                if 0 <= idx < combo.count():
                    combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        except Exception:
            pass

    def _refresh_enabled_states(self):
        has_cam   = self._camera is not None
        acq       = self._acquiring
        exp_auto  = self._combo_exp_auto.currentIndex() != 0
        gain_auto = self._combo_gain_auto.currentIndex() != 0

        self._combo_exp_auto .setEnabled(has_cam)
        self._spin_exp       .setEnabled(has_cam and not exp_auto)
        self._combo_gain_auto.setEnabled(has_cam)
        self._spin_gain      .setEnabled(has_cam and not gain_auto)
        self._combo_trigger  .setEnabled(has_cam)
        self._combo_userset  .setEnabled(has_cam and not acq)

    def _on_exp_auto_changed(self, idx):
        if self._camera:
            self._camera.MV_CC_SetEnumValue("ExposureAuto", idx)
        self._refresh_enabled_states()

    def _on_exp_time_changed(self):
        if self._camera:
            self._camera.MV_CC_SetFloatValue("ExposureTime", self._spin_exp.value())

    def _on_gain_auto_changed(self, idx):
        if self._camera:
            self._camera.MV_CC_SetEnumValue("GainAuto", idx)
        self._refresh_enabled_states()

    def _on_gain_changed(self):
        if self._camera:
            self._camera.MV_CC_SetFloatValue("Gain", self._spin_gain.value())

    def _on_trigger_changed(self, idx):
        if self._camera:
            self._camera.MV_CC_SetEnumValue("TriggerMode", idx)

    def _on_userset_changed(self, idx):
        if self._camera:
            self._camera.MV_CC_SetEnumValue("UserSetSelector", idx)
            self._camera.MV_CC_SetCommandValue("UserSetLoad")

# ── Main Window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._camera = None
        self._grab_thread = None
        self._ocr_worker = None
        self._current_frame = None
        self._annotated_frame = None
        self._ocr_results = []
        self._is_acquiring = False
        self._is_paused = False
        self._continuous_ocr = False  # 연속 OCR 실행 여부
        self._ocr_busy = False        # OCR 워커 실행 중 여부

        self._sel_dl = None
        self._sel_idx = -1
        self._sel_info = {}

        self._fps_frames = 0
        self._fps_time = time.time()

        self._build_ui()
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { font-family: 'Malgun Gothic', 'Segoe UI'; color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #333; background: #1e1e1e; }
            QTabBar::tab { background: #2a2a2a; color: #888; padding: 6px 14px; border: 1px solid #333; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #1e1e1e; color: #fff; border-bottom: 1px solid #1e1e1e; }
            QTabBar::tab:hover { background: #333; }
        """)

        # ── 앱 시작 시 OCR 엔진 백그라운드 사전 워밍 ─────────────────────────
        self._prewarm = EnginePrewarmThread()
        self._prewarm.status_msg.connect(self._status.showMessage)
        self._prewarm.cuda_ready.connect(self._on_cuda_ready)
        self._prewarm.start()

    def _build_ui(self):
        self.setWindowTitle("KRLee_OCR_RealTime - 실시간 OCR 검출 엔진")
        self.setMinimumSize(1200, 720)

        # ── Toolbar ───────────────────────────────────────────────────
        tb = QToolBar("OCR 도구모음")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setContentsMargins(6, 3, 6, 3)
        tb.setStyleSheet(
            "QToolBar{background:#181818;border-bottom:2px solid #2d2d2d;"
            "spacing:4px;padding:3px 6px;}"
            "QToolBar::separator{background:#3a3a3a;width:1px;margin:6px 4px;}"
        )
        self.addToolBar(Qt.TopToolBarArea, tb)

        # ① 카메라 선택 – 보라색
        self._btn_sel = self._make_icon_btn(
            "\U0001f4f7", "카메라 선택", "#6A1B9A", "#7B1FA2", "#4A148C")
        self._btn_sel.clicked.connect(self._open_cam_sel)
        tb.addWidget(self._btn_sel)

        # ② 취득 시작 / 정지 – 초록색
        self._btn_acq = self._make_icon_btn(
            "▶", "취득 시작", "#2E7D32", "#388E3C", "#1B5E20")
        self._btn_acq.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self._btn_acq.setEnabled(False)
        self._btn_acq.clicked.connect(self._toggle_acq)
        tb.addWidget(self._btn_acq)

        # ③ 스냅샷 저장 – 다크 회색
        self._btn_snap = self._make_icon_btn(
            "\U0001f4f8", "스냅샷 저장", "#424242", "#4f4f4f", "#2a2a2a")
        self._btn_snap.setEnabled(False)
        self._btn_snap.clicked.connect(self._save_snapshot)
        tb.addWidget(self._btn_snap)

        # ④ 라이브 일시정지 – 청록색  (⏸ 아이콘)
        self._btn_pause = self._make_icon_btn(
            "⏸", "라이브 일시정지", "#00695C", "#00796B", "#004D40")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._toggle_pause)
        tb.addWidget(self._btn_pause)

        # ⑥ 파일 열기 – 틸 색상
        self._btn_open = self._make_icon_btn(
            "\U0001f4c1", "파일 열기", "#00838F", "#0097A7", "#006064")
        self._btn_open.clicked.connect(self._open_file)
        tb.addWidget(self._btn_open)

        tb.addSeparator()

        # ⑦ OCR 모드 버튼
        self._btn_run_ocr = self._make_mode_btn("OCR", "OCR 실행")
        self._btn_run_ocr.setEnabled(False)
        self._btn_run_ocr.clicked.connect(self._start_ocr)
        tb.addWidget(self._btn_run_ocr)

        # ⑧ OBD 모드 버튼 (향후 확장)
        self._btn_obd = self._make_mode_btn("OBD", "OBD 분석 (준비 중)")
        self._btn_obd.setEnabled(False)
        tb.addWidget(self._btn_obd)

        # ⑨ SEG 모드 버튼 (향후 확장)
        self._btn_seg = self._make_mode_btn("SEG", "세그멘테이션 (준비 중)")
        self._btn_seg.setEnabled(False)
        tb.addWidget(self._btn_seg)

        # Spacer & status labels (right-aligned)
        sp = QWidget()
        sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(sp)

        # GPU / CPU 상태 뱃지
        self._lbl_gpu_badge = QLabel("OCR: 초기화 중…")
        self._lbl_gpu_badge.setStyleSheet(
            "color:#888;font-size:11px;font-weight:bold;"
            "background:#1e1e1e;border:1px solid #3a3a3a;"
            "border-radius:4px;padding:3px 8px;margin-right:6px;")
        tb.addWidget(self._lbl_gpu_badge)

        self._lbl_cam_info = QLabel("연결된 카메라 없음")
        self._lbl_cam_info.setStyleSheet(
            "color:#7ec8e3;font-size:12px;font-weight:bold;margin-right:12px;")
        tb.addWidget(self._lbl_cam_info)

        # ── Central widget ────────────────────────────────────────────
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setSpacing(6)
        outer.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle{background:#2a2a2a;width:4px;}")

        # Left part: Viewer
        viewer_box = QWidget()
        vl = QVBoxLayout(viewer_box)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)

        self._viewer = ZoomableImageLabel()
        self._viewer.setMinimumSize(600, 480)
        self._viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._viewer.setStyleSheet("QLabel{background:#0d0d0d;border:2px solid #2a2a2a;border-radius:4px;}")
        self._viewer.setText("대기 중 …\n\n[파일 열기] 또는 [카메라 선택]을 진행해 주세요.")
        self._viewer.setFont(QFont("Consolas", 12))
        vl.addWidget(self._viewer)

        splitter.addWidget(viewer_box)

        # Right part: Control panel
        right_panel = QTabWidget()
        right_panel.setFixedWidth(320)
        right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Tab 1: OCR Results
        ocr_widget = QWidget()
        ov = QVBoxLayout(ocr_widget)
        ov.setContentsMargins(10, 10, 10, 10)
        ov.setSpacing(8)

        ov.addWidget(self._make_section_label("인식 결과 목록 (선택 시 해당 영역 강조)"))
        self._result_list = QListWidget()
        self._result_list.setFont(QFont("Malgun Gothic", 11))
        self._result_list.setStyleSheet(
            "QListWidget{background:#141414;border:1px solid #333;color:#ddd;outline:0;border-radius:4px;}"
            "QListWidget::item{padding:5px 8px;border-bottom:1px solid #222;}"
            "QListWidget::item:selected{background:#1565C0;color:white;}"
            "QListWidget::item:hover{background:#252525;}"
        )
        self._result_list.itemSelectionChanged.connect(self._on_result_selection_changed)
        ov.addWidget(self._result_list, stretch=2)

        # ── OCR 처리 속도 표시 패널 ──────────────────────────────────────────
        timing_frame = QFrame()
        timing_frame.setFrameShape(QFrame.NoFrame)
        timing_frame.setStyleSheet(
            "QFrame{background:#0e1a24;border:1px solid #1e3a50;"
            "border-radius:5px;padding:0px;}"
        )
        tf_layout = QVBoxLayout(timing_frame)
        tf_layout.setContentsMargins(10, 6, 10, 6)
        tf_layout.setSpacing(4)

        timing_title = QLabel("⏱  OCR TackTime")
        timing_title.setStyleSheet(
            "color:#7ec8e3;font-size:11px;font-weight:bold;"
            "background:transparent;border:none;"
        )
        tf_layout.addWidget(timing_title)

        timing_row = QHBoxLayout()
        timing_row.setSpacing(8)

        self._lbl_time_total = QLabel("전체: --")
        self._lbl_time_total.setAlignment(Qt.AlignCenter)
        self._lbl_time_total.setStyleSheet(
            "color:#e0e0e0;font-size:12px;font-weight:bold;"
            "background:#132030;border:1px solid #1e3a50;"
            "border-radius:4px;padding:4px 6px;"
        )

        self._lbl_time_recog = QLabel("인식: --")
        self._lbl_time_recog.setAlignment(Qt.AlignCenter)
        self._lbl_time_recog.setStyleSheet(
            "color:#a5d6a7;font-size:12px;font-weight:bold;"
            "background:#132030;border:1px solid #1e3a50;"
            "border-radius:4px;padding:4px 6px;"
        )

        timing_row.addWidget(self._lbl_time_total, stretch=1)
        timing_row.addWidget(self._lbl_time_recog, stretch=1)
        tf_layout.addLayout(timing_row)

        # 전처리 시간 = 전체 - 인식
        self._lbl_time_pre = QLabel("전처리: --")
        self._lbl_time_pre.setAlignment(Qt.AlignCenter)
        self._lbl_time_pre.setStyleSheet(
            "color:#888;font-size:10px;background:transparent;border:none;"
        )
        tf_layout.addWidget(self._lbl_time_pre)
        ov.addWidget(timing_frame)

        ov.addWidget(self._make_section_label("전체 텍스트 (편집 가능)"))
        self._text_edit = QTextEdit()
        self._text_edit.setFont(QFont("Malgun Gothic", 12))
        self._text_edit.setStyleSheet(
            "QTextEdit{background:#141414;color:#e8e8e8;border:1px solid #333;border-radius:4px;line-height:140%;}"
        )
        ov.addWidget(self._text_edit, stretch=1)

        btn_copy = QPushButton("📋 전체 텍스트 복사")
        btn_copy.setStyleSheet(S_BLUE)
        btn_copy.clicked.connect(self._copy_text)
        ov.addWidget(btn_copy)

        right_panel.addTab(ocr_widget, "OCR 결과")

        # Tab 2: Camera control
        cam_ctrl_widget = QWidget()
        cv_layout = QVBoxLayout(cam_ctrl_widget)
        cv_layout.setContentsMargins(0, 0, 0, 0)
        
        self._settings_panel = CameraSettingsPanel()
        cv_layout.addWidget(self._settings_panel)
        right_panel.addTab(cam_ctrl_widget, "카메라 제어")

        # Tab 3: ROI settings
        roi_widget = QWidget()
        rv_layout = QVBoxLayout(roi_widget)
        rv_layout.setContentsMargins(0, 0, 0, 0)
        
        self._roi_panel = ROIPanel()
        self._roi_panel.apply_requested.connect(self._apply_roi)
        self._roi_panel.restore_requested.connect(self._restore_max_roi)
        rv_layout.addWidget(self._roi_panel)
        right_panel.addTab(roi_widget, "ROI 설정")

        splitter.addWidget(right_panel)
        splitter.setSizes([850, 320])
        outer.addWidget(splitter)

        # Status Bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status = sb
        self._lbl_fps = QLabel("FPS: --")
        self._lbl_fps.setStyleSheet("color:#aaa;padding:0 8px;")
        sb.addPermanentWidget(self._lbl_fps)
        sb.showMessage("준비 완료")

        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

    # ── Button factory helpers ────────────────────────────────────────────────
    def _make_icon_btn(self, icon: str, tooltip: str, bg: str, hover: str,
                       press: str = "#1a1a1a", size: int = 40) -> QPushButton:
        """정사각형 아이콘 전용 버튼 생성"""
        btn = QPushButton(icon)
        btn.setFixedSize(size, size)
        btn.setToolTip(tooltip)
        btn.setFont(QFont("Segoe UI Emoji", 16))
        btn.setStyleSheet(
            f"QPushButton{{background:{bg};color:white;border:none;"
            f"border-radius:6px;font-size:16px;padding:0;}}"
            f"QPushButton:hover{{background:{hover};}}"
            f"QPushButton:pressed{{background:{press};}}"
            f"QPushButton:disabled{{background:#303030;color:#555;}}"
        )
        return btn

    def _make_mode_btn(self, label: str, tooltip: str) -> QPushButton:
        """OCR / OBD / SEG 텍스트 모드 버튼 생성"""
        btn = QPushButton(label)
        btn.setFixedSize(50, 40)
        btn.setToolTip(tooltip)
        btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        btn.setStyleSheet(
            "QPushButton{background:#2e2e2e;color:#666;border:none;"
            "border-radius:6px;font-size:11px;font-weight:bold;padding:0;}"
            "QPushButton:hover{background:#3a3a3a;color:#ccc;}"
            "QPushButton:pressed{background:#1a1a1a;color:white;}"
            "QPushButton:enabled{background:#3a3a3a;color:#ddd;}"
            "QPushButton:disabled{background:#252525;color:#454545;}"
        )
        return btn

    def _make_section_label(self, text):
        l = QLabel(text)
        l.setStyleSheet("color:#7ec8e3;font-size:11px;font-weight:bold;")
        return l

    # ── Snapshot & Pause ─────────────────────────────────────────────────────
    def _save_snapshot(self):
        """현재 프레임을 파일로 저장 (스냅샷)"""
        if self._current_frame is None:
            QMessageBox.warning(self, "경고", "저장할 이미지가 없습니다.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "스냅샷 저장", f"snapshot_{ts}.png",
            "PNG 이미지 (*.png);;JPEG 이미지 (*.jpg)"
        )
        if not path:
            return
        try:
            cv2.imwrite(path, self._current_frame)
            self._status.showMessage(f"스냅샷 저장 완료: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", f"이미지 저장 중 오류:\n{e}")

    def _toggle_pause(self):
        """라이브 뷰 일시정지 / 재개 토글"""
        if not self._is_acquiring:
            return
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._btn_pause.setText("▶")
            self._btn_pause.setToolTip("라이브 재개")
            self._status.showMessage("라이브 뷰 일시정지됨")
        else:
            self._btn_pause.setText("⏸")
            self._btn_pause.setToolTip("라이브 일시정지")
            self._status.showMessage("라이브 뷰 재개됨")

    # ── File I/O ───────────────────────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 파일 열기", "",
            "이미지 파일 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;모든 파일 (*)"
        )
        if not path:
            return
        self._stop_acq()
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "열기 실패", f"이미지를 열 수 없습니다:\n{path}")
            return
        self._current_frame = img
        self._annotated_frame = img.copy()
        self._viewer.set_image(img)
        self._btn_run_ocr.setEnabled(True)
        self._result_list.clear()
        self._text_edit.clear()
        self._status.showMessage(f"이미지 불러옴: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")

    # ── Camera Selection & Open/Close ──────────────────────────────────────
    def _open_cam_sel(self):
        if self._is_acquiring:
            self._stop_acq()
        self._close_camera()
        dlg = CameraSelectDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self._apply_sel(dlg)

    def _apply_sel(self, dlg: CameraSelectDialog):
        self._sel_dl = dlg.device_list
        self._sel_idx = dlg.selected_index
        self._sel_info = dlg.selected_info or {}

        model = self._sel_info.get("model", "Unknown")
        serial = self._sel_info.get("serial", "")

        sn = f" (S/N: {serial})" if serial else ""
        self._lbl_cam_info.setText(f"카메라: {model}{sn}")
        self._status.showMessage(f"카메라 연결 중: {model} …")
        QApplication.processEvents()

        if self._open_camera():
            self._btn_acq.setEnabled(True)
            self._viewer.setText("카메라 대기 중 …\n\n[ 취득 시작 ] 버튼을 눌러 비디오를 활성화하세요.")
            self._status.showMessage(f"카메라 선택 완료: {model} {serial}")
        else:
            self._btn_acq.setEnabled(False)
            self._lbl_cam_info.setText("연결 실패")
            self._status.showMessage("카메라 열기 실패")

    def _open_camera(self) -> bool:
        self._close_camera()
        cam = MvCamera()
        if cam.MV_CC_CreateHandle(self._sel_dl.pDeviceInfo[self._sel_idx].contents) != 0:
            QMessageBox.critical(self, "오류", "카메라 핸들 생성 실패")
            return False
        if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
            QMessageBox.critical(self, "오류", "카메라 열기 실패\n다른 프로그램에서 사용 중인지 확인하세요.")
            cam.MV_CC_DestroyHandle()
            return False

        self._camera = cam
        constraints = read_roi_constraints(cam)
        self._roi_panel.update_constraints(constraints)
        self._settings_panel.set_camera(cam)
        return True

    def _close_camera(self):
        if self._camera:
            self._camera.MV_CC_CloseDevice()
            self._camera.MV_CC_DestroyHandle()
            self._camera = None
        self._roi_panel.set_inactive()
        self._settings_panel.set_camera(None)

    # ── Acquisition ──────────────────────────────────────────────────────────
    def _toggle_acq(self):
        if not self._is_acquiring:
            self._start_acq()
        else:
            self._stop_acq()

    def _start_acq(self):
        if not self._camera:
            return
        param = MVCC_INTVALUE()
        if self._camera.MV_CC_GetIntValue("PayloadSize", param) != 0:
            QMessageBox.critical(self, "오류", "PayloadSize 읽기 실패")
            return
        if self._camera.MV_CC_StartGrabbing() != 0:
            QMessageBox.critical(self, "오류", "영상 취득 시작 실패")
            return

        t = GrabThread(self._camera, param.nCurValue)
        t.frame_ready.connect(self._on_frame)
        t.start()
        self._grab_thread = t
        self._is_acquiring = True
        self._is_paused = False
        self._continuous_ocr = True   # 연속 OCR 시작
        self._ocr_busy = False

        self._roi_panel.freeze()
        self._settings_panel.on_acquiring(True)
        # 아이콘 버튼: 취득 중 → ⏹ 정지 아이콘으로 변경
        self._btn_acq.setText("⏹")
        self._btn_acq.setToolTip("취득 정지")
        self._btn_acq.setStyleSheet(
            "QPushButton{background:#555;color:white;border:none;border-radius:6px;"
            "font-size:16px;padding:0;}"
            "QPushButton:hover{background:#666;}"
            "QPushButton:pressed{background:#333;}"
        )
        self._btn_sel.setEnabled(False)
        self._btn_open.setEnabled(False)
        self._btn_snap.setEnabled(True)
        self._btn_pause.setEnabled(True)
        # 연속 OCR 모드이므로 개별 OCR 버튼 비활성화
        self._btn_run_ocr.setEnabled(False)
        self._btn_run_ocr.setToolTip("취득 중 연속 OCR 자동 수행 중")
        self._status.showMessage("영상 취득 중 … (연속 OCR 실행 중)")

    def _stop_acq(self):
        if self._grab_thread:
            self._grab_thread.stop()
            self._grab_thread = None
        self._is_acquiring = False
        self._is_paused = False
        self._continuous_ocr = False  # 연속 OCR 중단
        self._ocr_busy = False

        if self._camera:
            constraints = read_roi_constraints(self._camera)
            self._roi_panel.update_constraints(constraints)
        self._settings_panel.on_acquiring(False)

        # 아이콘 버튼: 정지 후 → ▶ 시작 아이콘으로 복구
        self._btn_acq.setText("▶")
        self._btn_acq.setToolTip("취득 시작")
        self._btn_acq.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self._btn_acq.setStyleSheet(
            "QPushButton{background:#2E7D32;color:white;border:none;border-radius:6px;"
            "font-size:16px;padding:0;}"
            "QPushButton:hover{background:#388E3C;}"
            "QPushButton:pressed{background:#1B5E20;}"
            "QPushButton:disabled{background:#303030;color:#555;}"
        )
        self._btn_sel.setEnabled(True)
        self._btn_open.setEnabled(True)
        self._btn_snap.setEnabled(False)
        self._btn_pause.setText("⏸")
        self._btn_pause.setToolTip("라이브 일시정지")
        self._btn_pause.setEnabled(False)
        # 현재 프레임이 있으면 단발 OCR 버튼 복구
        self._btn_run_ocr.setToolTip("OCR 실행")
        if self._current_frame is not None:
            self._btn_run_ocr.setEnabled(True)
        self._status.showMessage("취득 정지됨")

    def _on_frame(self, frame):
        self._fps_frames += 1
        if not self._is_paused:
            self._current_frame = frame
            self._viewer.set_image(frame)
            # 연속 OCR 모드: OCR이 바쁘지 않으면 새 프레임으로 즉시 OCR 실행
            if self._continuous_ocr and not self._ocr_busy:
                self._launch_ocr_worker(frame)

    def _launch_ocr_worker(self, frame: np.ndarray):
        """연속 OCR 워커 실행."""
        if self._ocr_busy or not self._continuous_ocr:
            return
        self._ocr_busy = True
        worker = OcrWorker(frame)
        worker.finished.connect(self._on_ocr_done)
        worker.error.connect(self._on_ocr_error)
        worker.status.connect(self._status.showMessage)
        worker.timing.connect(self._on_ocr_timing)
        worker.start()
        self._ocr_worker = worker

    def _update_fps(self):
        t = time.time()
        dt = t - self._fps_time
        if dt >= 1.0:
            fps = self._fps_frames / dt
            self._lbl_fps.setText(f"FPS: {fps:.1f}" if self._is_acquiring else "FPS: --")
            self._fps_frames = 0
            self._fps_time = t

    # ── ROI Control ──────────────────────────────────────────────────────────
    def _apply_roi(self, ox, oy, w, h):
        if not self._camera:
            return
        self._camera.MV_CC_SetIntValue("OffsetX", 0)
        self._camera.MV_CC_SetIntValue("OffsetY", 0)

        errors = []
        for name, val in (("Width", w), ("Height", h), ("OffsetX", ox), ("OffsetY", oy)):
            ret = self._camera.MV_CC_SetIntValue(name, val)
            if ret != 0:
                errors.append(f"{name}={val} (0x{ret:08X})")

        constraints = read_roi_constraints(self._camera)
        self._roi_panel.update_constraints(constraints)

        if errors:
            QMessageBox.warning(self, "ROI 설정 오류", "\n".join(errors))
        else:
            self._status.showMessage(f"ROI 적용: Offset({ox},{oy}) Size({w}x{h})")

    def _restore_max_roi(self):
        if not self._camera:
            return
        self._camera.MV_CC_SetIntValue("OffsetX", 0)
        self._camera.MV_CC_SetIntValue("OffsetY", 0)

        pw = MVCC_INTVALUE()
        ph = MVCC_INTVALUE()
        self._camera.MV_CC_GetIntValue("Width", pw)
        self._camera.MV_CC_GetIntValue("Height", ph)
        self._camera.MV_CC_SetIntValue("Width", pw.nMax)
        self._camera.MV_CC_SetIntValue("Height", ph.nMax)

        constraints = read_roi_constraints(self._camera)
        self._roi_panel.update_constraints(constraints)
        self._status.showMessage(f"최대 ROI 복원: {pw.nMax} × {ph.nMax}")

    # ── OCR Operations ───────────────────────────────────────────────────────
    def _start_ocr(self):
        """단발 OCR 실행 (파일 이미지 또는 취득 정지 상태에서 수동 실행)."""
        if self._current_frame is None:
            QMessageBox.warning(self, "경고", "분석할 이미지가 없습니다.")
            return
        # 연속 OCR 모드 중이면 개별 실행 무시 (버튼이 비활성화되어 있어야 하지만 안전 체크)
        if self._continuous_ocr:
            return

        self._btn_run_ocr.setEnabled(False)
        self._result_list.clear()
        self._text_edit.clear()

        # Display raw target image
        self._annotated_frame = self._current_frame.copy()
        self._viewer.set_image(self._annotated_frame)
        self._status.showMessage("OCR 실행 준비 중...")

        self._ocr_busy = True
        self._ocr_worker = OcrWorker(self._current_frame)
        self._ocr_worker.finished.connect(self._on_ocr_done)
        self._ocr_worker.error.connect(self._on_ocr_error)
        self._ocr_worker.status.connect(self._status.showMessage)
        self._ocr_worker.timing.connect(self._on_ocr_timing)
        self._ocr_worker.start()

    def _on_ocr_timing(self, total_ms: float, recog_ms: float):
        """OCR 처리 속도 표시 업데이트.

        ▸ total_ms  : 전처리 + 순수 추론 + 결과 조립 전체 (load_engine 제외)
        ▸ recog_ms  : 순수 AI 추론만 (전처리·파일I/O 제외)
        ▸ total_ms >= recog_ms 가 코드상 항상 보장됨
        """
        # 혹시라도 부동소수점 오차로 역전 시 recog 를 total 로 제한
        recog_ms = min(recog_ms, total_ms)
        pre_ms   = total_ms - recog_ms          # 전처리·오버헤드 시간

        # 전체 처리시간 (색상: 빠름=초록, 보통=노랑, 느림=빨강)
        if total_ms < 500:
            total_color = "#a5d6a7"
        elif total_ms < 1500:
            total_color = "#fff176"
        else:
            total_color = "#ef9a9a"
        self._lbl_time_total.setText(f"전체  {total_ms:.1f} ms")
        self._lbl_time_total.setStyleSheet(
            f"color:{total_color};font-size:12px;font-weight:bold;"
            "background:#132030;border:1px solid #1e3a50;"
            "border-radius:4px;padding:4px 6px;"
        )

        # 순수 인식 시간 (색상: 빠름=틸, 보통=노랑, 느림=빨강)
        if recog_ms < 300:
            recog_color = "#80cbc4"
        elif recog_ms < 1000:
            recog_color = "#fff176"
        else:
            recog_color = "#ef9a9a"
        self._lbl_time_recog.setText(f"인식  {recog_ms:.1f} ms")
        self._lbl_time_recog.setStyleSheet(
            f"color:{recog_color};font-size:12px;font-weight:bold;"
            "background:#132030;border:1px solid #1e3a50;"
            "border-radius:4px;padding:4px 6px;"
        )

        # 전처리·오버헤드 = 전체 - 인식
        self._lbl_time_pre.setText(f"전처리·오버헤드  {pre_ms:.1f} ms")

    def _on_ocr_done(self, results):
        self._ocr_busy = False
        # 단발 OCR 모드에서만 버튼 재활성화
        if not self._continuous_ocr:
            self._btn_run_ocr.setEnabled(True)

        self._ocr_results = results
        n = len(results)
        self._status.showMessage(f"OCR 완료: {n}개 항목 감지됨")

        self._draw_ocr_results(results)
        self._viewer.set_image(self._annotated_frame)

        self._result_list.clear()
        lines = []
        for bbox, text, conf in results:
            self._result_list.addItem(f" [{conf * 100:.0f}%] {text}")
            lines.append(text)
        self._text_edit.setPlainText("\n".join(lines))

        if n == 0:
            self._text_edit.setPlaceholderText("텍스트가 감지되지 않았습니다.")

    def _on_ocr_error(self, msg):
        self._ocr_busy = False
        if self._continuous_ocr:
            # 연속 OCR 모드: 팝업 없이 상태바에만 오류 표시 후 계속 진행
            self._status.showMessage(f"OCR 오류 (연속 모드): {msg[:80]}")
        else:
            # 단발 OCR 모드: 버튼 복구 + 오류 팝업
            self._btn_run_ocr.setEnabled(True)
            self._status.showMessage("OCR 오류 발생")
            QMessageBox.critical(self, "OCR 오류", msg)

    def _draw_ocr_results(self, results):
        img = self._current_frame.copy()
        for bbox, text, conf in results:
            pts = np.array([[int(x), int(y)] for x, y in bbox], dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 230, 60), 2)
            tx = int(pts[:, 0].min())
            ty = int(pts[:, 1].min()) - 6
            ty = max(ty, 14)
            label = f"{text} {conf * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 2), (0, 70, 0), -1)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 255, 100), 1, cv2.LINE_AA)
        self._annotated_frame = img

    def _on_result_selection_changed(self):
        row = self._result_list.currentRow()
        if 0 <= row < len(self._ocr_results):
            bbox, text, conf = self._ocr_results[row]
            self._viewer.set_highlighted_bbox(bbox)
        else:
            self._viewer.set_highlighted_bbox(None)

    def _copy_text(self):
        text = self._text_edit.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self._status.showMessage("클립보드에 복사되었습니다.")
        else:
            self._status.showMessage("복사할 텍스트가 없습니다.")

    def _save_result_image(self):
        if self._annotated_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 이미지 저장", "ocr_result.png",
            "PNG 이미지 (*.png);;JPEG 이미지 (*.jpg)"
        )
        if not path:
            return
        try:
            cv2.imwrite(path, self._annotated_frame)
            self._status.showMessage(f"이미지 저장 완료: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", f"이미지 저장 중 오류가 발생했습니다:\n{e}")

    # ── GPU/CPU 상태 뱃지 업데이트 ───────────────────────────────────────────
    def _on_cuda_ready(self, use_cuda: bool, gpu_name: str):
        if use_cuda:
            label = gpu_name if gpu_name else "NVIDIA GPU"
            self._lbl_gpu_badge.setText(f"GPU: {label}")
            self._lbl_gpu_badge.setStyleSheet(
                "color:#a5d6a7;font-size:11px;font-weight:bold;"
                "background:#0a2a0a;border:1px solid #2e7d32;"
                "border-radius:4px;padding:3px 8px;margin-right:6px;"
            )
        else:
            self._lbl_gpu_badge.setText("OCR: CPU 모드")
            self._lbl_gpu_badge.setStyleSheet(
                "color:#fff176;font-size:11px;font-weight:bold;"
                "background:#2a2800;border:1px solid #827717;"
                "border-radius:4px;padding:3px 8px;margin-right:6px;"
            )

    # ── Clean up ─────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._stop_acq()
        self._close_camera()
        # 상주 OCR 서버 프로세스 정리
        if OcrWorker._server is not None:
            OcrWorker._server.stop()
            OcrWorker._server = None
        event.accept()

# ── Main Entry ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    
    # Elegant dark palette style
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(18, 18, 18))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(18, 18, 18))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(30, 136, 229))
    dark_palette.setColor(QPalette.Highlight, QColor(30, 136, 229))
    dark_palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(dark_palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
