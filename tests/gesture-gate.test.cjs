const {test}=require('node:test');const assert=require('node:assert/strict');
const {GestureGate}=require('../spectramotus/studio/static/runtime/gesture-gate.js');
const sample=(label='Thumb_Up',x=.5,score=.95)=>({label,x,score,side:'Left'});
test('stable thumbs-up triggers once, requires neutral release',()=>{
 const g=new GestureGate(),events=[];
 for(let t=0;t<5000;t+=100){const e=g.update(sample(),t);if(e)events.push(e)}
 assert.deepEqual(events,['thumbs_up']);
 for(let t=5000;t<=5800;t+=100)g.update(null,t);
 for(let t=5900;t<=6600;t+=100){const e=g.update(sample(),t);if(e)events.push(e)}
 assert.deepEqual(events,['thumbs_up','thumbs_up']);
});
test('stationary palm, jitter, one-way motion and low confidence do not wave',()=>{
 for(const xs of [Array(15).fill(.5),[.50,.51,.49,.51,.49,.50],[.2,.3,.4,.5,.6,.7]]){
  const g=new GestureGate();xs.forEach((x,i)=>assert.equal(g.update(sample('Open_Palm',x),i*100),null));
 }
 const g=new GestureGate();for(let t=0;t<2000;t+=100)assert.equal(g.update(sample('Thumb_Up',.5,.5),t),null);
});
test('side-to-side open palm triggers a wave with two reversals',()=>{
 const g=new GestureGate(),events=[];
 [.3,.4,.5,.4,.3,.4,.5].forEach((x,i)=>{const e=g.update(sample('Open_Palm',x),i*100);if(e)events.push(e)});
 assert.deepEqual(events,['wave']);
});
test('interrupted thumbs-up does not accumulate hold time',()=>{
 const g=new GestureGate();g.update(sample(),0);g.update(sample(),100);g.update(null,200);
 assert.equal(g.update(sample(),500),null);assert.equal(g.update(sample(),600),null);
 assert.equal(g.update(sample(),1500),null);
});

test('wrist wave works without moving the palm center',()=>{
 const g=new GestureGate(),events=[];
 [0,.2,.45,.2,0].forEach((tilt,i)=>{const e=g.update({...sample('None',.5,.3),open:true,tilt,scale:.12},i*100);if(e)events.push(e)});
 assert.deepEqual(events,['wave']);
});
test('moving wave survives a brief classification dropout',()=>{
 const g=new GestureGate(),events=[];
 [.3,.34,null,.39,.34,.3].forEach((x,i)=>{const e=g.update(x===null?null:{...sample('None',x,.4),open:true,scale:.12},i*100);if(e)events.push(e)});
 assert.deepEqual(events,['wave']);
});
test('small distant wave uses hand-relative movement',()=>{
 const g=new GestureGate(),events=[];
 [.4,.42,.44,.42,.4].forEach((x,i)=>{const e=g.update({...sample('Open_Palm',x),scale:.06},i*100);if(e)events.push(e)});
 assert.deepEqual(events,['wave']);
});
test('long hand loss resets motion; tilted static hand does not trigger',()=>{
 const g=new GestureGate();g.update({...sample('Open_Palm',.3),open:true},0);g.update({...sample('Open_Palm',.4),open:true},100);g.update(null,800);
 for(let t=900;t<1800;t+=100)assert.equal(g.update({...sample('Open_Palm',.3),tilt:.8,open:true},t),null);
});


test('high-confidence thumbs-up triggers at 250ms, not a one-frame flash',()=>{
 const g=new GestureGate();assert.equal(g.update(sample(),0),null);assert.equal(g.update(sample(),200),null);assert.equal(g.update(sample(),250),'thumbs_up');
});
test('moderate-confidence thumb holds for 350ms',()=>{
 const g=new GestureGate();assert.equal(g.update(sample('Thumb_Up',.5,.75),0),null);assert.equal(g.update(sample('Thumb_Up',.5,.75),250),null);assert.equal(g.update(sample('Thumb_Up',.5,.75),350),'thumbs_up');
});
test('clear short wave triggers at 200ms',()=>{
 const g=new GestureGate();assert.equal(g.update(sample('Open_Palm',.3),0),null);assert.equal(g.update(sample('Open_Palm',.4),100),null);assert.equal(g.update(sample('Open_Palm',.3),200),'wave');
});


test('thumb diagnostics explain low confidence, broken holds and release latch',()=>{
 const g=new GestureGate();g.update(sample('Thumb_Up',.5,.52),0);assert.equal(g.debug.reason,'thumb_low_confidence');
 g.update(sample(),100);g.update(null,230);assert.equal(g.debug.resetReason,'no_hand');
 g.update(sample(),300);assert.equal(g.update(sample(),550),'thumbs_up');g.update(sample(),650);assert.equal(g.debug.reason,'waiting_for_release');
 g.reset();g.update(sample(),700);assert.equal(g.debug.holdMs,0);assert.equal(g.debug.reason,'thumb_hold');
});

test('consistent borderline thumbs-up survives the old confidence cliff',()=>{
 const g=new GestureGate();let event=null;[.66,.68,.71,.64,.69,.65,.67].forEach((score,i)=>{event=g.update(sample('Thumb_Up',.5,score),i*67)||event});assert.equal(event,'thumbs_up');
});
test('brief loss preserves thumb evidence, long loss and different pose reset it',()=>{
 const g=new GestureGate();g.update(sample('Thumb_Up',.5,.65),0);g.update(sample('Thumb_Up',.5,.65),67);g.update(null,100);assert.equal(g.debug.reason,'thumb_grace');g.update(sample('Thumb_Up',.5,.65),134);g.update(sample('Thumb_Up',.5,.65),234);g.update(sample('Thumb_Up',.5,.65),334);assert.equal(g.update(sample('Thumb_Up',.5,.65),400),'thumbs_up');
});
test('wave diagnostics explain stationary and reversing motion',()=>{
 const g=new GestureGate();g.update(sample('Open_Palm',.3),0);g.update(sample('Open_Palm',.4),100);assert.equal(g.debug.wave.translation.passed,false);g.update(sample('Open_Palm',.3),200);assert.equal(g.debug.wave.translation.passed,true);assert.ok(g.debug.wave.translation.reversal>.09);
});

test('switching gestures needs no neutral release but holding one does not repeat',()=>{
 const g=new GestureGate();g.update(sample(),0);g.update(sample(),100);g.update(sample(),200);assert.equal(g.update(sample(),300),'thumbs_up');
 for(let t=400;t<900;t+=100)assert.equal(g.update(sample(),t),null);
 g.update(sample('Open_Palm',.3),900);g.update(sample('Open_Palm',.4),1000);assert.equal(g.update(sample('Open_Palm',.3),1100),'wave');
 for(let t=1400;t<1700;t+=100)assert.equal(g.update(sample(),t),null);assert.equal(g.update(sample(),1700),'thumbs_up');
});
test('same gesture rearms after a brief 250ms neutral release',()=>{
 const g=new GestureGate();g.update(sample(),0);g.update(sample(),100);g.update(sample(),200);assert.equal(g.update(sample(),300),'thumbs_up');
 g.update(null,400);g.update(null,650);g.update(sample(),700);g.update(sample(),800);g.update(sample(),900);assert.equal(g.update(sample(),1000),'thumbs_up');
});
