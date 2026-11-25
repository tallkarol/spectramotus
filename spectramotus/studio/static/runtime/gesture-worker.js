// Classic worker: MediaPipe's WASM loader uses importScripts.
self.exports={};
importScripts('vendor/vision_bundle.js');
let recognizer;
self.onmessage=async({data})=>{
  try{
    if(data.type==='init'){
      const files=await exports.FilesetResolver.forVisionTasks(new URL('vendor/wasm',self.location.href).href);
      recognizer=await exports.GestureRecognizer.createFromOptions(files,{
        baseOptions:{modelAssetPath:new URL('vendor/gesture_recognizer.task',self.location.href).href,delegate:'CPU'},
        runningMode:'VIDEO',numHands:1,minHandDetectionConfidence:.45,minHandPresenceConfidence:.5,minTrackingConfidence:.5
      });
      self.postMessage({type:'ready'});
    }else if(data.type==='frame'){
      try{
        const started=performance.now();
        const result=recognizer.recognizeForVideo(data.frame,data.time);
        const g=result.gestures?.[0]?.[0],hand=result.landmarks?.[0];
        let sample=null;
        if(hand){
          const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
          const scale=distance(hand[0],hand[9]);
          const extended=[[8,6],[12,10],[16,14],[20,18]].filter(([tip,pip])=>distance(hand[tip],hand[0])>distance(hand[pip],hand[0])*1.18).length;
          sample={label:g?.categoryName??'None',score:g?.score??0,x:(hand[0].x+hand[9].x)/2,scale,
            nearEdge:hand.some(p=>p.x<.035||p.x>.965||p.y<.035||p.y>.965),
            tilt:Math.atan2(hand[9].x-hand[0].x,hand[0].y-hand[9].y),open:extended>=3,side:result.handedness?.[0]?.[0]?.categoryName};
        }
        self.postMessage({type:'result',sample,landmarks:hand?.map(p=>({x:p.x,y:p.y}))??[],frameTime:data.time,inferenceMs:performance.now()-started});
      }finally{data.frame.close()}
    }
  }catch(error){self.postMessage({type:'error',message:String(error)})}
};
