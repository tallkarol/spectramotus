/* Temporal filtering shared by desktop and Pi. No camera images are retained. */
class GestureGate {
  constructor(){this.reset()}
  reset(){this.latched=false;this.lastGesture=null;this.neutralSince=null;this.lastFire=-Infinity;this.lastIntent=-Infinity;this.lastTime=null;this.debug={reason:"reset"};this.clearMotion()}
  clearMotion(){this.thumbSince=null;this.thumbLast=null;this.wave=[]}
  update(sample,now){
    const priorHold=this.thumbSince;
    this.debug={reason:'evaluating',score:sample?.score??null,latched:this.latched,holdMs:priorHold===null?0:Math.round(now-priorHold),requiredHoldMs:sample?.score>=.85?250:350,cooldownRemainingMs:Math.max(0,Math.round(250-(now-this.lastFire))),resetReason:null};
    if(this.lastTime!==null&&now-this.lastTime>=700){this.clearMotion();this.debug.resetReason='frame_gap'}
    this.lastTime=now;
    const thumb=sample?.label==='Thumb_Up'&&sample.score>=.6;
    const palm=sample&&(sample.open===true||(sample.label==='Open_Palm'&&sample.score>=.6));
    if(!thumb&&!palm){
      this.debug.reason=!sample?'no_hand':sample.label==='Thumb_Up'?'thumb_low_confidence':'other_pose';
      if(priorHold!==null)this.debug.resetReason=this.debug.reason;
      const briefDrop=this.thumbLast!==null&&now-this.thumbLast<=120&&(!sample||sample.label==='Thumb_Up');
      if(briefDrop){this.debug.reason='thumb_grace';this.debug.resetReason=null;return null}
      this.thumbSince=null;this.thumbLast=null;
      this.neutralSince??=now;
      // Brief classifier/tracking dropouts are normal during a moving wave.
      if(now-this.lastIntent>350)this.clearMotion();
      if(now-this.neutralSince>=250)this.latched=false;
      return null;
    }
    this.neutralSince=null;this.lastIntent=now;
    const intent=thumb?'thumbs_up':'wave';
    // A different deliberate gesture can follow without lowering the hand.
    if((this.latched&&intent===this.lastGesture)||now-this.lastFire<250){this.debug.reason=this.latched&&intent===this.lastGesture?'waiting_for_release':'cooldown';this.clearMotion();return null}
    let event=null;
    if(thumb){
      this.wave=[];if(this.thumbLast!==null&&now-this.thumbLast>350)this.thumbSince=null;this.thumbSince??=now;this.thumbLast=now;this.debug.reason='thumb_hold';this.debug.holdMs=Math.round(now-this.thumbSince);
      if(now-this.thumbSince>=(sample.score>=.85?250:350))event='thumbs_up';
    }else{
      this.debug.reason='wave_motion';if(priorHold!==null)this.debug.resetReason='changed_to_palm';
      this.thumbSince=null;
      this.wave.push({x:sample.x,tilt:sample.tilt??0,scale:sample.scale??.15,t:now});
      this.wave=this.wave.filter(p=>now-p.t<=2200);
      const threshold=Math.max(.018,Math.min(.06,(sample.scale??.15)*.3));
      // A clear out-and-back movement, either translation or wrist rotation.
      const measure=(key,step)=>{
        let anchor=this.wave[0][key],extreme=anchor,direction=0,reverse=0;
        for(const p of this.wave){const value=p[key];
          if(!direction){if(Math.abs(value-anchor)>=step){direction=Math.sign(value-anchor);extreme=value}}
          else if((value-extreme)*direction>0)extreme=value;
          else reverse=Math.max(reverse,(extreme-value)*direction);
        }
        return {range:Math.max(...this.wave.map(p=>p[key]))-Math.min(...this.wave.map(p=>p[key])),reversal:reverse,threshold:step,passed:reverse>=step};
      };
      const translation=measure('x',threshold),rotation=measure('tilt',.30);
      this.debug.wave={samples:this.wave.length,windowMs:Math.round(now-this.wave[0].t),translation,rotation};
      if(this.wave.length>=3&&now-this.wave[0].t>=180&&(translation.passed||rotation.passed))event='wave';
    }
    if(event){this.debug.reason='recognized';this.lastDetectionMs=now-(event==='thumbs_up'?this.thumbSince:this.wave[0].t);this.latched=true;this.lastGesture=event;this.lastFire=now;this.clearMotion()}
    return event;
  }
}
if(typeof module!=='undefined')module.exports={GestureGate};
