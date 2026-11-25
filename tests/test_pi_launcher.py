import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'pi'))

spec=importlib.util.spec_from_file_location('pi_launcher',Path(__file__).parents[1]/'pi/launch.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)

@pytest.fixture
def bundle(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    (tmp_path/'player/runtime').mkdir(parents=True)
    (tmp_path/'player/runtime/test.js').write_text('trusted runtime')
    (tmp_path/'player/preview.html').write_text('trusted viewer: __MANIFEST__')
    manifest={'schema_version':1,'name':'Test','clips':[{'id':'wave','file':'clips/wave.mp4'}]}
    files={'manifest.json':json.dumps(manifest).encode(),'rest.png':b'photo','preview.html':b'untrusted viewer','README.txt':b'readme','clips/wave.mp4':b'video','runtime/test.js':b'imported runtime'}
    files['checksums.sha256']=''.join(hashlib.sha256(v).hexdigest()+'  '+k+'\n' for k,v in files.items()).encode()
    return tmp_path,files

def archive(root,files):
    p=root/'portrait.zip'
    with zipfile.ZipFile(p,'w') as z:
        for k,v in files.items():z.writestr(k,v)
    return p

def test_offline_kit_uses_trusted_player(bundle):
    root,files=bundle;launcher.prepare(archive(root,files),root/'output')
    assert (root/'output/preview.html').read_text().startswith('trusted viewer:')
    assert (root/'output/runtime/test.js').read_text()=='trusted runtime'

def test_tampered_kit_is_rejected_before_extracting(bundle):
    root,files=bundle;files['clips/wave.mp4']=b'tampered'
    with pytest.raises(ValueError,match='Checksum'):launcher.prepare(archive(root,files),root/'output')
    assert not (root/'output').exists()

def test_pack_traversal_is_rejected(bundle):
    root,files=bundle;files['../escape']=b'bad'
    with pytest.raises(ValueError,match='Unsafe'):launcher.prepare(archive(root,files),root/'output')
    assert not (root/'output').exists()


def test_csi_routes_are_loopback_only_and_require_custom_header(tmp_path):
    from functools import partial
    from http.server import ThreadingHTTPServer
    import threading
    import http.client
    class FakeCamera:
        running=False
        def start(self):self.running=True
        def stop(self):self.running=False
        def snapshot(self):return b'jpeg-test'
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(launcher.PlayerHandler,directory=str(tmp_path)))
    server.camera=FakeCamera()
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def request(method,path,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',server.server_port)
        connection.request(method,path,headers=headers or {})
        response=connection.getresponse();result=(response.status,response.read());connection.close();return result
    try:
        assert request('POST','/camera/start')[0]==403
        assert request('GET','/camera/frame')[0]==403
        headers={'X-Spectramotus-Camera':'1','Origin':'https://unrelated.example'}
        assert request('POST','/camera/start',headers)[0]==403
        headers['Origin']='http://127.0.0.1:'+str(server.server_port)
        assert request('POST','/camera/start',headers)[0]==200
        assert server.camera.running
        assert request('GET','/camera/frame',headers)==(200,b'jpeg-test')
        assert request('POST','/camera/stop',headers)[0]==200
        assert not server.camera.running
    finally:server.shutdown();server.server_close();thread.join()


def test_camera_frames_are_memory_only_and_stale_frames_fail():
    import time
    from camera_bridge import CameraBridge
    bridge=CameraBridge();bridge.camera=object()
    bridge.frames.write(b'latest-frame')
    assert bridge.snapshot()==b'latest-frame'
    bridge.frames.updated=time.monotonic()-5
    with pytest.raises(RuntimeError,match='fresh'):bridge.snapshot()
