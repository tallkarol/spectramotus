/* Same camera adapter for Railway, localhost desktop, and Pi kiosk. */
(()=>{
 const button=document.querySelector('#camera-toggle'),message=document.querySelector('#camera-status'),video=document.querySelector('#webcam');
 const overlay=document.querySelector('#hand-overlay'),overlayToggle=document.querySelector('#overlay-toggle'),ctx=overlay.getContext('2d');
 const bones=[[0,1,2,3,4],[0,5,6,7,8],[5,9,10,11,12],[9,13,14,15,16],[13,17,18,19,20],[0,17]];
 function draw(points){
  const w=video.videoWidth||640,h=video.videoHeight||480;if(overlay.width!==w||overlay.height!==h){overlay.width=w;overlay.height=h}
  ctx.clearRect(0,0,w,h);if(!overlayToggle.checked||!points?.length)return;
  ctx.strokeStyle='#9dffb0';ctx.lineWidth=3;
  for(const chain of bones){ctx.beginPath();chain.forEach((i,j)=>{const p=points[i];if(j)ctx.lineTo(p.x*w,p.y*h);else ctx.moveTo(p.x*w,p.y*h)});ctx.stroke()}
  ctx.fillStyle='#ffd18c';for(const p of points){ctx.beginPath();ctx.arc(p.x*w,p.y*h,4,0,Math.PI*2);ctx.fill()}
 }
 overlayToggle.onchange=()=>{if(!overlayToggle.checked)draw([])};
 const gate=new GestureGate();let wasListening=false;let worker=null,stream=null,timer=null,timeout=null,run=0,busy=false,starting=false,lastVideoTime=-1;
 let feedbackUntil=0,lastDiagnostic=0,frames=0,rateStarted=performance.now();
 const diagnostic=document.querySelector('#camera-diagnostic'),log=document.querySelector('#gesture-log');
 const say=text=>{if(message.textContent!==text)message.textContent=text;const kioskStatus=document.querySelector('#kiosk-camera-status');if(kioskStatus&&new URLSearchParams(location.search).has('kiosk')){kioskStatus.textContent=text;kioskStatus.hidden=!!stream&&!starting}};
 function record(text){const item=document.createElement('li');item.textContent=new Date().toLocaleTimeString()+' · '+text;log.prepend(item);while(log.children.length>12)log.lastChild.remove()}

 function stop(text='Camera off. Your video stays on this device.'){
  window.portraitTelemetry?.emit('camera_stopped',{reason:text});draw([]);
  run++;clearTimeout(timer);clearTimeout(timeout);worker?.terminate();worker=null;
  stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.hidden=true;
  gate.reset();feedbackUntil=0;busy=false;starting=false;lastVideoTime=-1;diagnostic.textContent='Camera off';button.textContent='Start camera';button.setAttribute('aria-pressed','false');say(text);
 }
 async function frame(token){
  if(token!==run||!stream)return;
  if(!busy&&video.readyState>=2&&video.currentTime!==lastVideoTime){
   busy=true;lastVideoTime=video.currentTime;
   try{const bitmap=await createImageBitmap(video);if(token!==run){bitmap.close();return}worker.postMessage({type:'frame',frame:bitmap,time:performance.now()},[bitmap]);timeout=setTimeout(()=>stop('Camera detection stalled. Enable camera to retry.'),10000)}
   catch(error){stop('Could not read the camera. Try enabling it again.');return}
  }
  timer=setTimeout(()=>frame(token),66);
 }
 button.onclick=async()=>{
  if(starting||stream){stop();return}
  if(!isSecureContext||!navigator.mediaDevices?.getUserMedia||location.protocol==='file:'){
   say('Camera needs HTTPS or the local preview launcher. Buttons still work.');return;
  }
  starting=true;const token=++run;button.textContent='Cancel camera';say('Allow camera access to try your gestures.');
  try{
   const acquired=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:960},height:{ideal:720},frameRate:{ideal:15,max:20},facingMode:'user'}});
   if(token!==run){acquired.getTracks().forEach(t=>t.stop());return}
   stream=acquired;video.srcObject=stream;video.hidden=false;await video.play();video.parentElement.style.aspectRatio=video.videoWidth+'/'+video.videoHeight;
   if(token!==run)return;
   stream.getVideoTracks()[0].onended=()=>stop('Camera disconnected. Reconnect it and enable camera again.');
   say('Preparing local gesture recognition…');
   worker=new Worker('runtime/gesture-worker.js');
   timeout=setTimeout(()=>stop('Could not start recognition. Reload the player and try again.'),45000);
   worker.onerror=()=>stop('Recognition could not load. Reload the player and try again.');
   worker.onmessage=({data})=>{
    if(token!==run)return;
    clearTimeout(timeout);
    if(data.type==='ready'){starting=false;button.textContent='Stop camera';button.setAttribute('aria-pressed','true');say('Ready. Wave hello or give a thumbs-up.');record('Camera ready');window.portraitTelemetry?.emit('camera_ready',{width:video.videoWidth,height:video.videoHeight,trackingProfile:'acquisition-v2'});rateStarted=performance.now();frames=0;frame(token)}
    else if(data.type==='error'){console.error('Portrait gesture worker:',data.message);stop('Gesture recognition stopped. Enable camera to retry.')}
    else if(data.type==='result'){
     busy=false;const now=performance.now(),sample=data.sample;frames++;draw(data.landmarks);
     const listening=window.portraitTelemetry?.acceptingGesture()??true;
     if(!listening||!wasListening)gate.reset();
     const event=listening?gate.update(sample,now):null;wasListening=listening;
     window.portraitTelemetry?.sample(sample,data.inferenceMs,listening?gate.debug:{reason:'test_window_closed'},data.frameTime);

     if(now-lastDiagnostic>300){
       diagnostic.textContent=(sample?'Hand tracked · '+sample.label+' '+Math.round(sample.score*100)+'% · '+(sample.open?'fingers open':'fingers folded'):'No hand tracked')+' · '+(frames*1000/(now-rateStarted)).toFixed(1)+' fps · '+Math.round(data.inferenceMs||0)+' ms';
       lastDiagnostic=now;
     }
     if(event){
       window.portraitTelemetry?.emit('recognized',{gesture:event,candidateMs:Math.round(gate.lastDetectionMs),inferenceMs:Math.round(data.inferenceMs||0),frameAgeMs:Math.round(now-data.frameTime)});
       const accepted=window.portraitPlayer.gesture(event);
       record((event==='wave'?'Wave':'Thumbs-up')+' · detected in '+Math.round(gate.lastDetectionMs)+' ms'+(accepted?' → accepted':' → portrait busy'));
       feedbackUntil=now+2000;
       say(accepted?(window.portraitPlayer.waiting()?'Got it — you’re next.':(event==='wave'?'Wave received.':'Thumbs-up received.'))+' Switch gestures, or lower your hand briefly to repeat.':'A reaction is already queued. Try again when it starts.');
     }else if(now>feedbackUntil){
       if(gate.latched)say('Switch gestures, or lower your hand briefly to repeat.');
       else if(!sample)say('Hand not found. Bring your whole hand into view, with light on your fingers.');
       else if(sample.nearEdge)say('Move your hand toward the middle of the camera preview.');
       else if(sample.scale<.07)say('Bring your hand a little closer to the camera.');
       else if(sample.label==='Thumb_Up'&&sample.score>=.6)say('Hold your thumbs-up for a moment…');
       else if(sample.open||sample.label==='Open_Palm')say('Hand seen. Give a little wave.');
       else say('Hand seen. Open your fingers to wave.');
     }
    }
   };
   worker.postMessage({type:'init'});
  }catch(error){if(token===run)stop(error.name==='NotAllowedError'?'Camera permission was not granted. Allow it in your browser, then retry.':error.name==='NotFoundError'?'No camera found. Connect one and try again.':'Camera unavailable. Close other camera apps and retry.')}
 };
 document.addEventListener('visibilitychange',()=>{if(document.hidden&&(stream||starting))stop('Camera paused while away. Enable it to resume.')});
 window.addEventListener('pagehide',()=>stop());
 if(new URLSearchParams(location.search).has('kiosk'))button.click();
})();
