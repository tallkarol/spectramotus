"""Preview a downloaded pack locally without uploading it anywhere."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import webbrowser
from .packs import extract_pack


def main():
    parser=argparse.ArgumentParser(description='Preview a Spectramotus pack offline')
    parser.add_argument('pack',type=Path)
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='spectramotus-preview-') as temp:
        target=Path(temp)/'pack'
        extract_pack(args.pack,target)
        handler=partial(SimpleHTTPRequestHandler,directory=str(target))
        with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
            url=f'http://127.0.0.1:{server.server_port}/preview.html'
            print(f'Offline portrait preview: {url}',flush=True)
            if not args.no_browser:webbrowser.open(url)
            try:server.serve_forever()
            except KeyboardInterrupt:pass


if __name__=='__main__':main()
