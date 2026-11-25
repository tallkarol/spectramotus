const {test}=require('node:test');const assert=require('node:assert/strict');const vm=require('node:vm');const fs=require('node:fs');
function setup(){
 const elements=new Map(),timers=new Map();let seq=0,time=0;
 const get=id=>{if(!elements.has(id))elements.set(id,{textContent:'',checked:true});return elements.get(id)};
 const context=vm.createContext({document:{getElementById:get},window:{portraitPlayer:{prepareTest(){},endTest(){},isAtRest:()=>true}},performance:{now:()=>time},Date,JSON,setTimeout(fn,ms){timers.set(++seq,{fn,ms});return seq},clearTimeout(id){timers.delete(id)}});
 vm.runInContext(fs.readFileSync('spectramotus/studio/static/runtime/diagnostics.js','utf8'),context);
 return {get,telemetry:context.window.portraitTelemetry,emit:context.window.portraitTelemetry.emit,tick(ms){time+=ms;for(const [id,t] of [...timers])if(t.ms===ms){timers.delete(id);t.fn()}},result(){return JSON.parse(get('test-results').textContent)}};
}
test('requires camera, records five of each with timings and final idle',()=>{
 const p=setup();p.get('test-start').onclick();assert.match(p.get('test-prompt').textContent,/Start the camera/);
 p.emit('camera_ready');p.get('test-start').onclick();
 for(let i=0;i<10;i++){
  p.telemetry.sample(null,20,{});p.tick(700);p.telemetry.sample(null,20,{});p.tick(2000);const clip=i<5?'wave':'thumbs_up',requestId=i+1;
  p.emit('recognized',{gesture:clip,candidateMs:260});p.emit('gesture_requested',{clip,requestId});p.emit('queued',{clip,requestId,behind:'idle_blink'});
  p.tick(230);p.emit('first_frame',{clip,requestId,measurement:'video-frame-callback'});p.emit('estimated_motion_onset',{clip,requestId,estimatedOnset:.667});p.emit('playback_ended',{clip,requestId});
  if(i<9)p.tick(1700);
 }
 p.emit('first_frame',{clip:'idle_blink',requestId:null});const r=p.result();assert.equal(r.trials.length,10);assert.equal(r.trials.filter(t=>t.expected==='wave').length,5);assert.ok(r.trials.every(t=>t.status==='matched'&&t.firstFrameMs===230));assert.equal(r.returnedToIdle,true);
});
test('misses, wrong gestures and camera interruption remain explicit',()=>{
 const p=setup();p.emit('camera_ready');p.get('test-start').onclick();p.telemetry.sample(null,20,{});p.tick(700);p.telemetry.sample(null,20,{});p.tick(2000);p.tick(6000);assert.equal(p.result().trials[0].status,'not_detected');
 p.telemetry.sample(null,20,{});p.tick(700);p.telemetry.sample(null,20,{});p.tick(2000);p.emit('recognized',{gesture:'thumbs_up',candidateMs:300});p.emit('gesture_requested',{clip:'thumbs_up',requestId:1});p.emit('playback_ended',{clip:'thumbs_up',requestId:1});assert.equal(p.result().trials[1].status,'wrong_gesture');
 p.telemetry.sample(null,20,{});p.tick(700);p.telemetry.sample(null,20,{});p.tick(2000);p.emit('camera_stopped');assert.equal(p.result().trials[2].status,'cancelled');assert.equal(p.result().returnedToIdle,false);
});


test('gesture windows stay closed between isolated trials and while responding',()=>{
 const p=setup();p.emit('camera_ready');p.get('test-start').onclick();assert.equal(p.telemetry.acceptingGesture(),false);
 p.telemetry.sample(null,20,{});p.tick(700);p.telemetry.sample(null,20,{});assert.equal(p.telemetry.acceptingGesture(),false);
 p.tick(2000);assert.equal(p.telemetry.acceptingGesture(),true);
 p.telemetry.sample({label:'Thumb_Up',score:.62},20,{reason:'thumb_low_confidence',resetReason:'thumb_low_confidence'});
 p.emit('recognized',{gesture:'wave',candidateMs:300});p.emit('gesture_requested',{clip:'wave',requestId:1});assert.equal(p.telemetry.acceptingGesture(),false);
 p.emit('playback_ended',{clip:'wave',requestId:1});assert.equal(p.telemetry.acceptingGesture(),false);
 const t=p.result().trials[0];assert.equal(t.samples,1);assert.equal(t.reasons.thumb_low_confidence,1);
});
