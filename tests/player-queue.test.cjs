const {test}=require('node:test');const assert=require('node:assert/strict');const vm=require('node:vm');const fs=require('node:fs');
function player(){
 const elements=new Map(),timers=new Map();let next=0;
 const element=()=>({style:{},dataset:{},classList:{add(){},remove(){},toggle(){}},append(){},setAttribute(){},events:{},addEventListener(name,fn){this.events[name]=fn},pause(){},play(){return Promise.resolve()},checked:false});
 const get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
 get('#idle').checked=true;
 const document={hidden:false,body:element(),querySelector:get,querySelectorAll:()=>[],createElement:element,addEventListener(){}};
 const manifest={name:'Test',canvas:{width:832,height:1104},clips:['wave','thumbs_up','idle_blink'].map(id=>({id,file:id+'.mp4',label:id,kind:'existing',duration_seconds:5,role:id==='idle_blink'?'idle':'reaction'})),policy:{idle:{clip_id:'idle_blink',delay_ms:4000}}};
 let source=fs.readFileSync('spectramotus/studio/static/preview.html','utf8').split('<script>')[1].split('</script>')[0].replace('__MANIFEST__',JSON.stringify(manifest));
 const context=vm.createContext({document,window:{},location:{search:''},URLSearchParams,setTimeout(fn,ms){timers.set(++next,{fn,ms});return next},clearTimeout(id){timers.delete(id)}});
 vm.runInContext(source,context);
 const run=code=>vm.runInContext(code,context);
 const tick=ms=>{for(const [id,t] of [...timers])if(t.ms===ms){timers.delete(id);t.fn()}};
 return {context,run,tick,document};
}
test('webcam wave queues one thumbs-up, then returns to idle',async()=>{
 const p=player();assert.equal(p.run("window.portraitPlayer.gesture('wave')"),true);await Promise.resolve();
 assert.equal(p.run("active.id"),'wave');assert.equal(p.run("window.portraitPlayer.gesture('thumbs_up')"),true);
 assert.equal(p.run("window.portraitPlayer.gesture('wave')"),false);
 p.run("players.get('wave').video.events.ended()");p.tick(0);await Promise.resolve();assert.equal(p.run('active.id'),'thumbs_up');
 p.run("players.get('thumbs_up').video.events.ended()");p.tick(0);assert.equal(p.run('phase'),'rest');
 p.tick(4000);await Promise.resolve();assert.equal(p.run('active.id'),'idle_blink');
});
test('gesture interrupts idle through a fade rather than waiting for the full clip',async()=>{
 const p=player();p.tick(4000);await Promise.resolve();assert.equal(p.run('active.id'),'idle_blink');
 assert.equal(p.run("window.portraitPlayer.gesture('wave')"),true);assert.equal(p.run('phase'),'settling');
 p.tick(230);p.tick(0);await Promise.resolve();assert.equal(p.run('active.id'),'wave');
});


test('isolated test rejects out-of-window gestures and manual reactions',async()=>{
 const p=player();p.run("window.portraitTelemetry={testing:()=>true,acceptingGesture:()=>false,emit(){}};window.portraitPlayer.prepareTest()");
 assert.equal(p.run("window.portraitPlayer.gesture('wave')"),false);p.run("trigger('wave')");await Promise.resolve();assert.equal(p.run('active'),null);
 p.run("window.portraitTelemetry.acceptingGesture=()=>true");p.run("trigger('wave')");assert.equal(p.run('active'),null);
 assert.equal(p.run("window.portraitPlayer.gesture('wave')"),true);await Promise.resolve();assert.equal(p.run('active.id'),'wave');
});

test('next reaction starts without a rest cooldown and accepts another queued gesture',async()=>{
 const p=player();p.run("window.portraitPlayer.gesture('wave')");await Promise.resolve();
 p.run("window.portraitPlayer.gesture('thumbs_up')");p.run("players.get('wave').video.events.ended()");p.tick(0);await Promise.resolve();
 assert.equal(p.run('active.id'),'thumbs_up');assert.equal(p.run("window.portraitPlayer.gesture('wave')"),true);
 p.run("players.get('thumbs_up').video.events.ended()");p.tick(0);await Promise.resolve();assert.equal(p.run('active.id'),'wave');
});

test('explicit event mappings honor custom Motus and unassigned gestures',async()=>{
 const p=player();p.run("manifest.events={wave:'thumbs_up'}");assert.equal(p.run("window.portraitPlayer.gesture('thumbs_up')"),false);assert.equal(p.run("window.portraitPlayer.gesture('wave')"),true);await Promise.resolve();assert.equal(p.run('active.id'),'thumbs_up');
});
