#!/usr/bin/env python3
"""Offline Spectramotus launcher. Python standard library only."""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
import zipfile
from urllib.parse import urlencode
from camera_bridge import CameraBridge

ROOT=Path(__file__).resolve().parent

def prepare(archive, destination):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)>100 or len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>200*1024*1024:
            raise ValueError('Invalid or oversized portrait pack.')
        for name in names:
            if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name:
                raise ValueError('Unsafe pack path.')
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('schema_version')!=1:raise ValueError('Unsupported portrait pack version.')
        expected={'manifest.json','rest.png','preview.html','README.txt','checksums.sha256'}
        for clip in manifest['clips']:
            path=clip['file']
            if not path.startswith('clips/') or not path.endswith('.mp4'):raise ValueError('Invalid clip path.')
            expected.add(path)
        trusted={str(p.relative_to(ROOT/'player')) for p in (ROOT/'player/runtime').rglob('*') if p.is_file()}
        if set(names)!=expected|trusted:raise ValueError('Missing or unexpected pack files.')
        checks=dict((name,sha) for sha,name in (line.split('  ',1) for line in z.read('checksums.sha256').decode().splitlines()))
        if set(checks)!=(expected|trusted)-{'checksums.sha256'}:raise ValueError('Incomplete checksum list.')
        for name,sha in checks.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=sha:raise ValueError('Checksum mismatch: '+name)
        destination.mkdir()
        z.extractall(destination)
        shutil.copytree(ROOT/'player/runtime',destination/'runtime',dirs_exist_ok=True)
        html=(ROOT/'player/preview.html').read_text().replace('__MANIFEST__',json.dumps(manifest).replace('<','\\u003c'))
        (destination/'preview.html').write_text(html)
    return manifest


def install_autostart(camera,profile):
    if sys.platform!='linux':raise SystemExit('Autostart setup is for Raspberry Pi OS Desktop only.')
    folder=Path.home()/'.config/labwc';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'autostart';text=path.read_text() if path.exists() else ''
    marker='# Spectramotus player'
    if marker in text:
        print('Spectramotus autostart already exists. Edit ~/.config/labwc/autostart if you move this kit.');return
    command=' '.join(shlex.quote(s) for s in [sys.executable,str(ROOT/'launch.py'),'--kiosk','--camera',camera,'--camera-profile',profile])
    if path.exists():shutil.copyfile(path,path.with_name('autostart.before-spectramotus'))
    path.write_text(text+'\n'+marker+'\n'+command+' &\n')
    print('Added desktop autostart. Enable desktop auto-login in Raspberry Pi OS settings, then reboot after first-run camera setup.')


class PlayerHandler(SimpleHTTPRequestHandler):
    def camera_request(self):
        expected='127.0.0.1:'+str(self.server.server_port)
        return (self.headers.get('Host')==expected and
                self.headers.get('X-Spectramotus-Camera')=='1' and
                self.headers.get('Origin') in (None,'http://'+expected))
    def reply(self,status,data,kind='application/json'):
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def camera_error(self,error):
        self.reply(503,json.dumps({'error':str(error)}).encode())
    def do_POST(self):
        if self.path not in {'/camera/start','/camera/stop'}:self.send_error(404);return
        if not self.camera_request():self.send_error(403);return
        if not self.server.camera:self.send_error(404);return
        try:
            if self.path=='/camera/start':self.server.camera.start()
            else:self.server.camera.stop()
            self.reply(200,b'{"ok":true}')
        except Exception as error:self.camera_error(error)
    def do_GET(self):
        if self.path=='/camera/frame':
            if not self.camera_request():self.send_error(403);return
            if not self.server.camera:self.send_error(404);return
            try:self.reply(200,self.server.camera.snapshot(),'image/jpeg')
            except Exception as error:self.camera_error(error)
        else:super().do_GET()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pack',nargs='?',type=Path,default=ROOT/'portrait.zip')
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--kiosk',action='store_true')
    parser.add_argument('--camera',choices=['usb','csi'],default='usb')
    parser.add_argument('--camera-profile',choices=['balanced','light'],default='balanced')
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--install-autostart',action='store_true')
    args=parser.parse_args()
    if args.install_autostart:install_autostart(args.camera,args.camera_profile);return
    with tempfile.TemporaryDirectory(prefix='spectramotus-') as temp:
        target=Path(temp)/'portrait';manifest=prepare(args.pack,target)
        print('Verified portrait:',manifest['name'],flush=True)
        if args.verify_only:return
        handler=partial(PlayerHandler,directory=str(target))
        with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
            server.camera=CameraBridge((640,480) if args.camera_profile=='light' else (960,720)) if args.camera=='csi' else None
            options={'camera':args.camera,'cameraProfile':args.camera_profile}
            if args.kiosk:options['kiosk']='1'
            url=f'http://127.0.0.1:{server.server_port}/preview.html?'+urlencode(options)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            print('Offline player:',url,flush=True)
            process=None
            try:
                chromium=shutil.which('chromium') or shutil.which('chromium-browser')
                if not args.no_browser:
                    if sys.platform=='linux' and chromium:
                        profile=Path.home()/'.local/share/spectramotus/chromium'
                        profile.mkdir(parents=True,exist_ok=True)
                        flags=['--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile)]
                        if args.kiosk:flags+=['--kiosk']
                        process=subprocess.Popen([chromium,*flags,url])
                    elif args.kiosk:raise SystemExit('Kiosk requires Chromium on Raspberry Pi OS Desktop. Use normal mode for a computer preview.')
                    else:webbrowser.open(url)
                if process:process.wait()
                else:
                    while True:thread.join(1)
            except KeyboardInterrupt:pass
            finally:
                if process and process.poll() is None:process.terminate()
                server.shutdown();thread.join()
                if server.camera:server.camera.stop()

if __name__=='__main__':main()
