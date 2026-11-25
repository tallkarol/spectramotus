"""Local-only CSI snapshots. Frames stay in RAM; no recording or network exposure."""
import io
import threading
import time

class Frames(io.BufferedIOBase):
    def __init__(self):
        self.condition=threading.Condition();self.frame=None;self.updated=0
    def write(self,buf):
        with self.condition:
            self.frame=bytes(buf);self.updated=time.monotonic();self.condition.notify_all()
        return len(buf)

class CameraBridge:
    def __init__(self,size=(640,480)):
        self.last_read=0;self.size=size;self.camera=None;self.frames=Frames();self.lock=threading.RLock()
    def start(self):
        with self.lock:
            if self.camera:return
            try:
                from picamera2 import Picamera2
                from picamera2.encoders import JpegEncoder
                from picamera2.outputs import FileOutput
            except ImportError as error:
                raise RuntimeError('Install python3-picamera2 using setup.sh, then use system python3.') from error
            camera=Picamera2()
            try:
                camera.configure(camera.create_video_configuration(main={'size':self.size},controls={'FrameRate':15},buffer_count=4))
                self.frames=Frames()
                camera.start_recording(JpegEncoder(),FileOutput(self.frames))
                self.camera=camera;self.last_read=time.monotonic()+45 # Allow the first model load on a cold Pi.
                threading.Thread(target=self.watch,daemon=True).start()
            except Exception:
                camera.close();raise
    def watch(self):
        while True:
            time.sleep(2)
            with self.lock:
                if self.camera is None:return
                if time.monotonic()-self.last_read>15:self.stop();return
    def snapshot(self):
        with self.lock:
            if self.camera is None:raise RuntimeError('Camera is off. Press Start camera to reconnect.')
            self.last_read=time.monotonic()
            with self.frames.condition:
                self.frames.condition.wait_for(lambda:self.frames.frame is not None,timeout=3)
                if self.frames.frame is None or time.monotonic()-self.frames.updated>3:
                    raise RuntimeError('No fresh camera frames. Check the cable and restart the camera.')
                return self.frames.frame
    def stop(self):
        with self.lock:
            if self.camera:
                try:self.camera.stop_recording()
                finally:self.camera.close();self.camera=None;self.frames=Frames()
