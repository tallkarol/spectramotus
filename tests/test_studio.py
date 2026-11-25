import io
import json
import time
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from spectramotus.studio.app import create_app
from spectramotus.studio.packs import extract_pack


def video(path):
    import subprocess
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=red:s=32x32:r=24','-t','0.5','-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)


def photo(color='navy'):
    b=io.BytesIO();Image.new('RGB',(120,160),color).save(b,format='PNG');return b.getvalue()


def wait(client,pid):
    for _ in range(150):
        r=client.get('/api/projects/'+pid).json()
        if r['status']!='preparing':return r
        time.sleep(.1)
    raise AssertionError('Preparation timed out')


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path/'data',password='test-secret',demo_dir=tmp_path/'demo')) as c:
        yield c


def login(c):
    assert c.post('/login',data={'password':'test-secret'}).status_code==200


def test_private_and_invalid_upload(client):
    assert client.get('/api/projects').status_code==401
    assert client.get('/media/nope/rest.png').status_code==401
    assert client.post('/login',data={'password':'wrong'}).status_code==401
    login(client)
    assert client.post('/api/projects',files={'photo':('bad.png',b'not an image')}).status_code==400
    assert client.post('/api/demo',headers={'Origin':'https://evil.example'}).status_code==403


def test_upload_review_export_offline_and_restart(client,tmp_path):
    login(client)
    r=client.post('/api/projects',data={'name':'<script>alert(1)</script>'},files={'photo':('portrait.png',photo())})
    assert r.status_code==202
    pid=r.json()['id']
    assert client.get(f'/api/projects/{pid}/download').status_code==409
    result=wait(client,pid)
    assert result['status']=='needs_review'
    assert result['existing_wave'] is False
    assert all(x['kind']=='placeholder' for x in result['manifest']['clips'])
    assert client.post(f'/api/projects/{pid}/approve').status_code==200
    download=client.get(f'/api/projects/{pid}/download');assert download.status_code==200
    archive=tmp_path/'portrait.zip';archive.write_bytes(download.content)
    manifest=extract_pack(archive,tmp_path/'preview')
    assert manifest['credits_spent']==0
    viewer=(tmp_path/'preview/preview.html').read_text()
    assert '__MANIFEST__' not in viewer
    assert '<script>alert(1)</script>' not in viewer
    assert 'https://' not in viewer
    with TestClient(create_app(client.app.state.root,password='test-secret',demo_dir=tmp_path/'demo')) as restarted:
        login(restarted)
        assert restarted.get(f'/api/projects/{pid}').json()['status']=='ready'
        assert restarted.get(f'/api/projects/{pid}/download').content==download.content


def test_existing_wave_only_exact_photo(tmp_path):
    demo=tmp_path/'demo';demo.mkdir()
    Image.new('RGB',(120,160),'red').save(demo/'photo.jpg')
    # Pack creation copies existing media, never generates/replaces it.
    video(demo/'wave.mp4')
    with TestClient(create_app(tmp_path/'data',password='test-secret',demo_dir=demo)) as c:
        login(c)
        pid=c.post('/api/demo').json()['id'];result=wait(c,pid)
        assert result['existing_wave'] is True
        assert c.get(f'/media/{pid}/clips/wave.mp4').content==(demo/'wave.mp4').read_bytes()
        pid=c.post('/api/projects',files={'photo':('photo.jpg',photo('green'))}).json()['id']
        assert wait(c,pid)['existing_wave'] is False


def test_reject_traversal_and_tampered_pack(tmp_path,client):
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../escape','bad')
    with pytest.raises(ValueError):extract_pack(bad,tmp_path/'unsafe')
    login(client)
    pid=client.post('/api/projects',files={'photo':('p.png',photo())}).json()['id']
    wait(client,pid);client.post(f'/api/projects/{pid}/approve')
    original=io.BytesIO(client.get(f'/api/projects/{pid}/download').content)
    with zipfile.ZipFile(original) as src,zipfile.ZipFile(bad,'w') as dst:
        for n in src.namelist():dst.writestr(n,b'changed' if n=='rest.png' else src.read(n))
    with pytest.raises(ValueError,match='checksum'):extract_pack(bad,tmp_path/'tampered')


def test_added_reaction_requires_review_once(tmp_path):
    import shutil
    demo=tmp_path/'demo';demo.mkdir()
    Image.new('RGB',(120,160),'red').save(demo/'photo.jpg')
    video(demo/'wave.mp4')
    root=tmp_path/'data'
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);pid=c.post('/api/demo').json()['id'];wait(c,pid)
        c.post(f'/api/projects/{pid}/approve')
        # A valid MP4 candidate arrives after the old pack was approved.
        shutil.copyfile(root/'projects'/pid/'pack/clips/thumbs_up.mp4',demo/'thumbs_up.mp4')
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);r=c.get(f'/api/projects/{pid}').json()
        assert r['status']=='needs_review'
        assert r['manifest']['version']==2
        assert r['manifest']['clips'][1]['kind']=='existing'
        assert c.get(f'/api/projects/{pid}/download').status_code==409
        c.post(f'/api/projects/{pid}/approve')
        z=zipfile.ZipFile(io.BytesIO(c.get(f'/api/projects/{pid}/download').content))
        assert z.read('clips/thumbs_up.mp4')==(demo/'thumbs_up.mp4').read_bytes()
        unrelated=c.post('/api/projects',files={'photo':('green.png',photo('green'))}).json()['id']
        assert wait(c,unrelated)['manifest']['clips'][1]['kind']=='placeholder'
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);r=c.get(f'/api/projects/{pid}').json()
        assert r['status']=='ready' and r['manifest']['version']==2


def test_idle_migration_export_and_photo_isolation(tmp_path):
    import shutil
    demo=tmp_path/'demo';demo.mkdir()
    Image.new('RGB',(120,160),'red').save(demo/'photo.jpg')
    video(demo/'wave.mp4')
    root=tmp_path/'data'
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);pid=c.post('/api/demo').json()['id'];wait(c,pid)
        c.post(f'/api/projects/{pid}/approve')
        shutil.copyfile(root/'projects'/pid/'pack/clips/thumbs_up.mp4',demo/'idle_blink.mp4')
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);r=c.get(f'/api/projects/{pid}').json()
        assert r['status']=='needs_review'
        assert r['manifest']['policy']['idle']['clip_id']=='idle_blink'
        assert r['manifest']['version']==2
        assert c.get(f'/media/{pid}/clips/idle_blink.mp4').content==(demo/'idle_blink.mp4').read_bytes()
        assert c.get(f'/api/projects/{pid}/download').status_code==409
        c.post(f'/api/projects/{pid}/approve')
        archive=tmp_path/'pack.zip';archive.write_bytes(c.get(f'/api/projects/{pid}/download').content)
        m=extract_pack(archive,tmp_path/'offline')
        assert m['clips'][-1]['role']=='idle'
        unrelated=c.post('/api/projects',files={'photo':('green.png',photo('green'))}).json()['id']
        assert next(clip for clip in wait(c,unrelated)['manifest']['clips'] if clip['id']=='idle_blink')['kind']=='placeholder'
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);r=c.get(f'/api/projects/{pid}').json()
        assert r['status']=='ready' and r['manifest']['version']==2


def test_camera_runtime_bundled_and_trusted_on_import(client,tmp_path):
    from spectramotus.studio.packs import runtime_files, STATIC
    login(client)
    pid=client.post('/api/projects',files={'photo':('p.png',photo())}).json()['id']
    wait(client,pid);client.post(f'/api/projects/{pid}/approve')
    archive=tmp_path/'camera.zip';archive.write_bytes(client.get(f'/api/projects/{pid}/download').content)
    with zipfile.ZipFile(archive) as z:
        assert set(runtime_files()).issubset(z.namelist())
        assert 'runtime/camera.js' in z.read('preview.html').decode()
        assert len(z.read('runtime/vendor/gesture_recognizer.task'))>1000000
    worker=client.get(f'/media/{pid}/runtime/gesture-worker.js')
    assert worker.status_code==200
    assert "'wasm-unsafe-eval'" in worker.headers['Content-Security-Policy']
    assert "'unsafe-eval'" not in worker.headers['Content-Security-Policy']
    assert "'wasm-unsafe-eval'" not in client.get(f'/media/{pid}/preview.html').headers['Content-Security-Policy']
    assert client.get(f'/media/{pid}/runtime/not-allowed.js').status_code==404
    extract_pack(archive,tmp_path/'camera')
    for name in runtime_files():assert (tmp_path/'camera'/name).read_bytes()==(STATIC/name).read_bytes()


def test_trimmed_wave_migrates_with_timing_and_requires_review(tmp_path):
    demo=tmp_path/'demo';demo.mkdir()
    Image.new('RGB',(120,160),'red').save(demo/'photo.jpg')
    video(demo/'wave.mp4');root=tmp_path/'data'
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);pid=c.post('/api/demo').json()['id'];wait(c,pid)
        c.post(f'/api/projects/{pid}/approve')
    # Replace the source with another valid existing clip and its measured timing.
    import shutil
    shutil.copyfile(root/'projects'/pid/'pack/clips/thumbs_up.mp4',demo/'wave.mp4')
    (demo/'clip-timing.json').write_text(json.dumps({'wave':{'motion_onset_seconds':.125}}))
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);r=c.get(f'/api/projects/{pid}').json()
        assert r['status']=='needs_review'
        wave=r['manifest']['clips'][0]
        assert wave['duration_seconds']==2
        assert wave['motion_onset_seconds']==.125
        assert c.get(f'/media/{pid}/clips/wave.mp4').content==(demo/'wave.mp4').read_bytes()
        c.post(f'/api/projects/{pid}/approve')
        z=zipfile.ZipFile(io.BytesIO(c.get(f'/api/projects/{pid}/download').content))
        assert json.loads(z.read('manifest.json'))['clips'][0]['motion_onset_seconds']==.125


def test_motus_admin_auth_upload_revision_and_selection(client,tmp_path):
    assert client.get('/api/admin/motus').status_code==401
    assert client.get('/admin',follow_redirects=False).status_code==303
    login(client)
    clip=tmp_path/'drive.mp4';video(clip)
    data={'name':'Small nod','description':'A little nod.','trigger':'manual','enabled':'true','token_price':'12','prompt':'Nod once. Return to rest.'}
    r=client.post('/api/admin/motus',data=data,files={'markdown':('notes.md',b'# Notes\nKeep still.','text/markdown'),'driving_video':('drive.mp4',clip.read_bytes(),'video/mp4')})
    assert r.status_code==200,r.text
    recipe=r.json();ident=recipe['id'];assert recipe['instructions']=='# Notes\nKeep still.'
    assert client.get('/api/admin/motus/'+ident+'/video').content==clip.read_bytes()
    catalog=client.get('/api/motus').json();assert catalog['generation_enabled'] is False and catalog['billing_enabled'] is False
    public=next(m for m in catalog['items'] if m['id']==ident);assert 'prompt' not in public
    pid=client.post('/api/projects',data={'motus_ids':json.dumps([ident])},files={'photo':('p.png',photo())}).json()['id']
    project=wait(client,pid);assert project['status']=='needs_review',project
    assert project['planned_tokens']==12 and project['tokens_charged']==0 and project['credits_spent']==0
    assert project['motus_plan'][0]['revision']==1
    assert [c['id'] for c in project['manifest']['clips']]==[ident]
    assert project['manifest']['clips'][0]['kind']=='placeholder'
    assert client.get('/media/'+pid+'/clips/'+ident+'.mp4').status_code==200
    update={**data,'ident':ident,'revision':1,'prompt':'Updated prompt.','enabled':'false'}
    assert client.post('/api/admin/motus',data=update).status_code==200
    assert client.post('/api/admin/motus',data=update).status_code==409
    assert client.get('/api/projects/'+pid).json()['motus_plan'][0]['prompt']=='Nod once. Return to rest.'
    assert all(m['id']!=ident for m in client.get('/api/motus').json()['items'])
    assert client.post('/api/projects',data={'motus_ids':json.dumps([ident])},files={'photo':('p.png',photo())}).status_code==409
    assert client.post('/api/projects',data={'motus_ids':'[]'},files={'photo':('p.png',photo())}).status_code==400
    assert client.post('/api/admin/motus',data={**data,'token_price':'-1'}).status_code==400
    assert client.post('/api/admin/motus',data=data,files={'driving_video':('bad.mp4',b'not video','video/mp4')}).status_code==400


def test_selected_motus_persist_and_do_not_reappear_on_restart(tmp_path):
    demo=tmp_path/'demo';demo.mkdir();Image.new('RGB',(120,160),'red').save(demo/'photo.jpg')
    video(demo/'wave.mp4');video(demo/'thumbs_up.mp4');video(demo/'idle_blink.mp4')
    root=tmp_path/'data'
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c)
        pid=c.post('/api/projects',data={'motus_ids':'["wave"]'},files={'photo':('p.jpg',(demo/'photo.jpg').read_bytes())}).json()['id']
        r=wait(c,pid);assert r['status']=='needs_review'
        assert [x['id'] for x in r['manifest']['clips']]==['wave']
        c.post('/api/projects/'+pid+'/approve')
    with TestClient(create_app(root,password='test-secret',demo_dir=demo)) as c:
        login(c);r=c.get('/api/projects/'+pid).json()
        assert r['status']=='ready'
        assert [x['id'] for x in r['manifest']['clips']]==['wave']
        assert len(c.get('/api/admin/motus').json()['items'])==3
