"""Portable, self-contained portrait packs and safe local preview."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from PIL import Image, ImageOps

STATIC = Path(__file__).parent / 'static'
MAX_UPLOAD = 15 * 1024 * 1024
MAX_PACK = 200 * 1024 * 1024


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_photo(path):
    with Image.open(path) as im:
        if im.width * im.height > 40_000_000:
            raise ValueError('Please use a photo smaller than 40 megapixels.')
        im = ImageOps.exif_transpose(im).convert('RGB')
        return ImageOps.fit(im, (832, 1104), method=Image.Resampling.LANCZOS)


def photo_signature(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        # Exact decoded content, not filename or a loose visual similarity match.
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def runtime_files():
    return sorted(str(p.relative_to(STATIC)) for p in (STATIC/'runtime').rglob('*') if p.is_file())


def install_runtime(folder):
    shutil.copytree(STATIC/'runtime',folder/'runtime',dirs_exist_ok=True)


def make_pack(folder, manifest):
    install_runtime(folder)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    payload = json.dumps(manifest).replace('<', '\\u003c')
    html = (STATIC / 'preview.html').read_text().replace('__MANIFEST__', payload)
    (folder / 'preview.html').write_text(html)
    (folder / 'README.txt').write_text(
        'Spectramotus placeholder pack\n\nUnzip, then open preview.html in your browser. '
        'Buttons work offline without a camera. For webcam gestures, serve this folder on localhost (on desktop or Pi), or use python -m spectramotus.studio.preview portrait.zip.\n'
        'Enable camera, then wave side to side or hold a thumbs-up. Switch between gestures during playback; lower your hand briefly to repeat the same gesture. Camera frames stay on this device. The model and runtime are bundled.\n'
        'This is a content pack, NOT a bootable Raspberry Pi image.\n'
        'The manifest marks each clip as existing footage or a still-image placeholder.\n'
    )
    files = ['manifest.json', 'rest.png', 'preview.html', 'README.txt'] + [x['file'] for x in manifest['clips']] + runtime_files()
    (folder / 'checksums.sha256').write_text(''.join(f'{digest(folder / p)}  {p}\n' for p in files))
    dest = folder.parent / 'portrait.zip'
    with zipfile.ZipFile(dest.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
        for name in files + ['checksums.sha256']:
            z.write(folder / name, name)
    dest.with_suffix('.tmp').replace(dest)
    return dest


def extract_pack(archive, destination):
    """Validate before exposing a downloaded pack to a local HTTP server."""
    destination = Path(destination)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or len(names) > 100:
            raise ValueError('Duplicate entries or too many files in pack.')
        if sum(x.file_size for x in z.infolist()) > MAX_PACK:
            raise ValueError('Pack is too large.')
        for n in names:
            p = Path(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n:
                raise ValueError('Unsafe pack path.')
        manifest = json.loads(z.read('manifest.json'))
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported pack version.')
        required = {'manifest.json', 'rest.png', 'preview.html', 'README.txt', 'checksums.sha256'}
        for clip in manifest['clips']:
            name = clip['file']
            if not name.startswith('clips/') or Path(name).suffix != '.mp4':
                raise ValueError('Invalid clip path.')
            required.add(name)
        bundled={n for n in names if n.startswith('runtime/')}
        if bundled and bundled!=set(runtime_files()):
            raise ValueError('Incomplete or unexpected camera runtime.')
        required.update(bundled)
        if set(names) != required:
            raise ValueError('Unexpected or missing pack files.')
        checks = {}
        for line in z.read('checksums.sha256').decode().splitlines():
            sha, name = line.split('  ', 1)
            checks[name] = sha
        if set(checks) != required - {'checksums.sha256'}:
            raise ValueError('Incomplete checksums.')
        for name, sha in checks.items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError('Pack checksum mismatch.')
        destination.mkdir(parents=True, exist_ok=False)
        try:
            z.extractall(destination)
            # Replace imported executable runtime with our trusted bundled version.
            install_runtime(destination)
            # Use our trusted viewer, never execute HTML from an imported archive.
            payload = json.dumps(manifest).replace('<', '\\u003c')
            (destination / 'preview.html').write_text((STATIC / 'preview.html').read_text().replace('__MANIFEST__', payload))
        except Exception:
            shutil.rmtree(destination)
            raise
    return manifest
