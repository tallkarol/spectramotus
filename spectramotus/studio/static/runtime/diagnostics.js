/* Local timing/classification diagnostics. No images or landmark coordinates saved. */
(()=>{
 const events=[],trials=[];let cameraReady=false,running=false,current=null,timer=null,index=0,started=null,stage='off',neutralSince=null,returnedToIdle=false;
 const $=id=>document.getElementById(id),prompt=$('test-prompt'),summary=$('test-summary');
 const emit=(kind,detail={})=>{
  const event={kind,t:Math.round(performance.now()),...detail};events.push(event);if(events.length>1500)events.shift();
  if(kind==='camera_ready')cameraReady=true;
  if(kind==='camera_stopped'){cameraReady=false;if(running)stop('Test stopped: camera unavailable.')}
  if(current?.status==='listening'){
   if(kind==='recognized'){current.recognized=detail.gesture;current.candidateMs=detail.candidateMs;current.promptToRecognitionMs=event.t-current.promptAt;current.recognitionAt=event.t}
   if(kind==='gesture_rejected'){current.rejectReason=detail.reason;finishTrial('rejected')}
   if(kind==='gesture_requested'){current.requestId=detail.requestId;current.requestAt=event.t;current.recognized=detail.gesture??detail.clip;current.status='responding';stage='responding';clearTimeout(timer);timer=setTimeout(()=>finishTrial('playback_timeout'),18000)}
  }
  if(current?.status==='responding'&&event.requestId===current.requestId){
   if(kind==='queued')current.queuedBehind=detail.behind;
   if(kind==='first_frame'){current.firstFrameMs=event.t-current.requestAt;current.firstFrameMeasurement=detail.measurement}
   if(kind==='estimated_motion_onset'){current.estimatedMotionMs=event.t-current.requestAt;current.clipMotionEstimateSeconds=detail.estimatedOnset}
   if(kind==='playback_ended')finishTrial(current.recognized===current.expected?'matched':'wrong_gesture');
   if(kind==='playback_error')finishTrial('playback_error');
  }
  if(stage==='awaiting_idle'&&kind==='first_frame'&&(detail.role==='idle'||detail.clip==='idle_blink')){
   clearTimeout(timer);running=false;stage='complete';returnedToIdle=true;prompt.textContent='Finished. Ten isolated attempts tested; portrait returned to idle.';window.portraitPlayer?.endTest();controls();
  }
  render();return event;
 };
 function render(){
  const done=trials.filter(t=>!['listening','responding'].includes(t.status));
  const counts=id=>{const set=done.filter(t=>t.expected===id);return set.filter(t=>t.status==='matched').length+'/'+set.length};
  summary.textContent=trials.length?'Matched responses · Wave '+counts('wave')+' · Thumbs-up '+counts('thumbs_up'):'';
  // Keep the page readable; full frame-by-frame decisions are in the JSON download.
  $('test-results').textContent=JSON.stringify({schema:3,stage,trials:trials.map(({filterSamples,...trial})=>trial),returnedToIdle},null,2);
 }
 function controls(){$('test-start').disabled=running;$('test-stop').hidden=!running}
 function waitForNeutral(){stage='waiting_neutral';neutralSince=null;prompt.textContent='Lower your hand. Waiting for the portrait to finish and your hand to reset…'}
 function finishTrial(status){
  if(!current)return;clearTimeout(timer);current.status=status;current.finishedAt=Math.round(performance.now());
  if(current.samples)current.meanInferenceMs=Math.round(current.totalInferenceMs/current.samples);
  current=null;index++;render();
  if(status==='playback_timeout'||status==='playback_error'){stop('Playback failed. Test stopped to preserve isolated results.');return}
  if(index===10){stage='awaiting_idle';prompt.textContent='All ten attempts finished. Waiting for idle…';window.portraitPlayer?.endTest();timer=setTimeout(()=>{running=false;stage='complete';prompt.textContent='Ten attempts finished; return to idle was not observed.';window.portraitPlayer?.endTest();render();controls()},12000)}
  else waitForNeutral();
 }
 function countdown(){
  stage='countdown';const expected=index<5?'wave':'thumbs_up',number=index%5+1;
  prompt.textContent=(expected==='wave'?'Wave ':'Thumbs-up ')+number+'/5 · Get ready…';
  timer=setTimeout(()=>{
   stage='listening';current={attempt:index+1,expected,status:'listening',promptAt:Math.round(performance.now()),samples:0,handSamples:0,labels:{},reasons:{},filterSamples:[],totalInferenceMs:0};trials.push(current);
   prompt.textContent=(expected==='wave'?'Wave now':'Give a thumbs-up now')+' · '+number+'/5';
   emit('trial_prompt',{attempt:index+1,expected,isolated:true});timer=setTimeout(()=>finishTrial('not_detected'),6000);
  },2000);
 }
 function stop(reason){clearTimeout(timer);if(current){current.status='cancelled';current=null}running=false;stage='stopped';prompt.textContent=reason;window.portraitPlayer?.endTest();render();controls()}
 $('test-start').onclick=()=>{
  prompt.hidden=false;if(!cameraReady){prompt.textContent='Start the camera first, then start the test.';return}
  trials.length=0;index=0;started=new Date().toISOString();running=true;returnedToIdle=false;
  window.portraitPlayer?.prepareTest();controls();emit('test_started',{schema:3,isolated:true});waitForNeutral();
 };
 $('test-stop').onclick=()=>stop('Test stopped. You can start again.');
 $('report-download').onclick=()=>{
  const blob=new Blob([JSON.stringify({schema:3,started,stage,returnedToIdle,trials,events,notes:'Isolated attempts. Filter samples cover the listening window only. No video or landmark coordinates recorded. Prompt timing includes human reaction time; clip motion onset is estimated.'},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='spectramotus-gesture-test.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 };
 window.portraitTelemetry={emit,testing:()=>running,acceptingGesture:()=>!running||stage==='listening',sample(sample,inferenceMs,filter={},frameTime){
  const now=performance.now();
  if(stage==='waiting_neutral'){
   const intentional=sample&&(sample.open||(sample.label==='Thumb_Up'&&sample.score>=.7)||(sample.label==='Open_Palm'&&sample.score>=.6));
   if(intentional)neutralSince=null;else neutralSince??=now;
   if(neutralSince!==null&&now-neutralSince>=700&&window.portraitPlayer?.isAtRest())countdown();
  }
  if(current?.status==='listening'){
   current.samples++;if(sample)current.handSamples++;const label=sample?.label??'no_hand';current.labels[label]=(current.labels[label]||0)+1;current.totalInferenceMs+=inferenceMs||0;
   const reason=filter.reason??'unknown';current.reasons[reason]=(current.reasons[reason]||0)+1;
   if(current.filterSamples.length<200)current.filterSamples.push({t:Math.round(now-current.promptAt),frameAgeMs:Number.isFinite(frameTime)?Math.round(now-frameTime):null,label,confidence:sample?Math.round(sample.score*1000)/1000:null,open:sample?.open??false,handScale:sample?.scale??null,nearEdge:sample?.nearEdge??false,inferenceMs:Math.round(inferenceMs||0),...filter});
  }
 }};
})();
