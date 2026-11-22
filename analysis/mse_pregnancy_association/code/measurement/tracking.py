"""Tracklets with last-reference re-localization and one-frame confirmation.

target_valid describes supported positions. step_valid additionally excludes
every re-anchor transition. Features must consume step_valid + segment_id.
"""
import numpy as np
import cv2
from ._lk import match, local_contrast

def track_segments(gray,initial,time_s,poses,*,recover=True,retry_frames=4,
                   window=21,max_fb_px=.75,contrast_sigma=5.,qc_downsample=1,qc_window=None,pose_epoch=None,pose_valid=None):
    t=np.asarray(time_s,float);ini=np.asarray(initial,np.float32);poses=np.asarray(poses,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all():raise ValueError('Expected at least two finite timestamps')
    if ini.ndim!=2 or ini.shape[1]!=2 or len(ini)==0 or not np.isfinite(ini).all():raise ValueError('Expected finite nonempty Nx2 seeds')
    if not np.isfinite(poses).all():raise ValueError('Poses must be finite')
    if poses.shape!=(len(t),3,3) or not np.allclose(poses[:,2],[0,0,1]) or np.any(np.abs(np.linalg.det(poses))<1e-12):raise ValueError('Expected invertible affine image-to-reference poses')
    if len(gray)!=len(t) or poses.shape!=(len(t),3,3) or np.any(np.diff(t)<=0):raise ValueError('Frame/time/pose mismatch')
    if gray[0].ndim!=2 or gray[0].dtype!=np.uint8:raise ValueError('Expected uint8 grayscale frames')
    epoch=np.zeros(len(t),int) if pose_epoch is None else np.asarray(pose_epoch)
    if epoch.shape!=t.shape:raise ValueError('pose_epoch must match timestamps')
    pose_valid=None if pose_valid is None else np.asarray(pose_valid,bool)
    if pose_valid is not None and pose_valid.shape!=t.shape:raise ValueError('pose_valid must match timestamps')
    if retry_frames<1:raise ValueError('retry_frames must be positive')
    if window<7 or window%2!=1 or max_fb_px<=0 or contrast_sigma<=0:raise ValueError('Invalid tracking scale')
    if not isinstance(qc_downsample,int) or qc_downsample<1:raise ValueError('QC downsample must be a positive integer')
    if qc_window is not None and (qc_window<3 or qc_window%2!=1):raise ValueError('QC window must be odd and at least 3')
    qc=gray if qc_downsample==1 else [cv2.resize(frame,None,fx=1/qc_downsample,fy=1/qc_downsample,interpolation=cv2.INTER_AREA) for frame in gray]
    n,k=len(t),len(ini)
    p=np.full((n,k,2),np.nan,np.float32);v=np.zeros((n,k),bool);step=np.zeros((n,k),bool)
    segment=np.full((n,k),-1,int);reanchor=np.zeros((n,k),bool);attempt=np.zeros((n,k),bool)
    score=np.full((n,k),np.nan);fb=np.full_like(score,np.nan);reason=np.full((n,k),'waiting',dtype='U28')
    p[0]=ini;v[0]=True;segment[0]=0;reason[0]='seed'
    last_index=np.zeros(k,int);last_point=ini.copy();next_retry=np.zeros(k,int);sid=np.zeros(k,int)
    pending_index=np.full(k,-1,int);pending_point=np.full((k,2),np.nan,np.float32)
    enhanced=[local_contrast(gray[0],sigma=contrast_sigma)]
    for i in range(1,n):
        if gray[i].shape!=gray[0].shape or gray[i].dtype!=np.uint8:raise ValueError('Frame geometry or dtype changed')
        enhanced.append(local_contrast(gray[i],sigma=contrast_sigma))
        # A pending seed is committed only after a successful native-frame step.
        pending=np.flatnonzero(pending_index==i-1)
        if len(pending):
            q,g,s,f,r=match(enhanced[i-1],enhanced[i],qc[i-1],qc[i],pending_point[pending],window,max_fb_px=max_fb_px,qc_scale=qc_downsample,qc_window=qc_window)
            for j,col in enumerate(pending):
                score[i,col]=s[j];fb[i,col]=f[j]
                if g[j]:
                    seed=int(pending_index[col]);sid[col]+=1
                    p[seed,col]=pending_point[col];v[seed,col]=True;segment[seed,col]=sid[col];reanchor[seed,col]=True
                    reason[seed,col]='reanchor_confirmed';step[seed,col]=False
                    p[i,col]=q[j];v[i,col]=True;segment[i,col]=sid[col];step[i,col]=True;reason[i,col]='accepted_after_reanchor'
                    last_index[col]=i;last_point[col]=q[j]
                else:reason[i,col]='confirmation_failed'
                pending_index[col]=-1
        # Existing tracks keep the exact original localization/QC behavior.
        active=np.flatnonzero(v[i-1]&~reanchor[i-1])
        # Confirmed seeds were handled above, so do not process them twice.
        active=np.setdiff1d(active,pending,assume_unique=True)
        if len(active):
            q,g,s,f,r=match(enhanced[i-1],enhanced[i],qc[i-1],qc[i],p[i-1,active],window,max_fb_px=max_fb_px,qc_scale=qc_downsample,qc_window=qc_window)
            score[i,active]=s;fb[i,active]=f;reason[i,active]=r
            keep=active[g];p[i,keep]=q[g];v[i,keep]=True;segment[i,keep]=sid[keep];step[i,keep]=True
            last_index[keep]=i;last_point[keep]=q[g]
            next_retry[active[~g]]=i+retry_frames
        if recover:
            waiting=np.flatnonzero(~v[i]&(pending_index<0)&(i>=next_retry))
            for ref in np.unique(last_index[waiting]):
                cols=waiting[last_index[waiting]==ref]
                m=np.linalg.inv(poses[i])@poses[int(ref)]
                if not np.isfinite(m).all():continue
                guess=last_point[cols]@m[:2,:2].T+m[:2,2]
                q,g,s,f,r=match(enhanced[int(ref)],enhanced[i],qc[int(ref)],qc[i],last_point[cols],window,guess,max_fb_px=max_fb_px,qc_scale=qc_downsample,qc_window=qc_window)
                attempt[i,cols]=True;next_retry[cols]=i+retry_frames
                for j,col in enumerate(cols):
                    if g[j]:
                        pending_index[col]=i;pending_point[col]=q[j]
                        reason[i,col]='awaiting_confirmation';score[i,col]=s[j];fb[i,col]=f[j]
    corrected=np.einsum('tni,tji->tnj',p.astype(float),poses[:,:2,:2])+poses[:,None,:2,2]
    result=dict(positions=p,corrected=corrected,target_valid=v,step_valid=step,segment_id=segment,reanchor=reanchor,
        recovery_attempted=attempt,time_s=t,frame_indices=np.arange(n),initial_points=ini,poses=poses,
        target_ncc=score,target_fb=fb,reject_reason=reason,pose_epoch=epoch)
    if pose_valid is not None:result['pose_valid']=pose_valid.copy()
    return result


