import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest

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
