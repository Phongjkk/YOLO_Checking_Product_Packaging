import cv2
import subprocess
import threading
import numpy as np
import time
import json
from utils.visual import FPSCalculator

class CamThread:
    def __init__(
        self,
        cam_name,
        source,
        mode="rtsp",
        target_fps=20,
        gpu_type="nvidia" # Chọn: "nvidia" hoặc "intel" hoặc "none"
    ):
        self.cam_name = cam_name
        self.source = source
        self.mode = mode
        self.target_fps = target_fps
        self.gpu_type = gpu_type

        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.fps_calc = FPSCalculator()
        self.last_capture_time = 0
        self.frame_interval = 1.0 / target_fps

        if self.mode == "webcam":
            self._init_webcam()
        elif self.mode == "rtsp":
            self._init_rtsp_windows_gpu()
        else:
            raise ValueError(f"Unknown camera mode: {self.mode}")

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _init_webcam(self):
        # Trên Windows dùng CAP_DSHOW để truy cập webcam nhanh hơn
        print(f"[{self.cam_name}] Windows Webcam - DSHOW")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def _probe_rtsp_resolution(self):
        # Dùng ffprobe để lấy kích thước khung hình
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            self.source
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            info = json.loads(result.stdout)
            stream = info["streams"][0]
            return stream["width"], stream["height"]
        except Exception as e:
            print(f"[{self.cam_name}] Error probing resolution: {e}")
            return 1280, 720 # Giá trị mặc định nếu probe lỗi

    def _init_rtsp_windows_gpu(self):
        self.width, self.height = self._probe_rtsp_resolution()
        self.frame_size = self.width * self.height * 3
        
        # Thiết lập Hardware Acceleration dựa trên GPU
        hwaccel = "d3d11va" # Mặc định cho Windows (Intel/AMD/Nvidia)
        if self.gpu_type == "nvidia":
            hwaccel = "cuda"
            print(f"[{self.cam_name}] RTSP + NVIDIA CUDA Acceleration")
        else:
            print(f"[{self.cam_name}] RTSP + Windows D3D11VA Acceleration")

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-hwaccel", hwaccel,
            "-rtsp_transport", "tcp",
            "-i", self.source,
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-"
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 3
        )

    def update(self):
        while self.running:
            current_time = time.time()
            if current_time - self.last_capture_time < self.frame_interval:
                time.sleep(0.001)
                continue

            if self.mode == "webcam":
                ret, frame = self.cap.read()
                if not ret: continue
            
            elif self.mode == "rtsp":
                raw = self.proc.stdout.read(self.frame_size)
                if not raw or len(raw) < self.frame_size:
                    continue
                
                # Chuyển raw bytes thành mảng numpy
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))

            with self.lock:
                self.frame = frame
                self.last_capture_time = current_time
                self.fps_calc.update()

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.fps_calc.get_fps()

    def release(self):
        self.running = False
        if hasattr(self, "cap"): self.cap.release()
        if hasattr(self, "proc"):
            try:
                self.proc.terminate() # Thử tắt nhẹ nhàng trước khi kill
                self.proc.wait(timeout=1)
            except:
                self.proc.kill()
        print(f"[{self.cam_name}] Released.")