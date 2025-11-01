#!/usr/bin/env python3
import cv2, json, sys
p='docs/assets/agent_demo.mp4'
cap=cv2.VideoCapture(p)
if not cap.isOpened():
    print(json.dumps({'ok':False,'error':'cannot_open'}))
    sys.exit(0)
frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
if fps<=0: fps=30.0
duration = frame_count / fps if fps>0 else None
max_samples=300
step = max(1, frame_count//max_samples) if frame_count>0 else 1
sampled=0
valid=0
means=[]
stds=[]
for i in range(0, frame_count, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue
    import numpy as np
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m = float(gray.mean())
    s = float(gray.std())
    means.append(m); stds.append(s)
    sampled+=1
    if m>6 and s>2:
        valid+=1
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
ret, f0 = cap.read()
fm=None
if ret:
    fm = float(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY).mean())
cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count-1))
ret, fl = cap.read()
lm=None
if ret:
    lm = float(cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY).mean())
cap.release()
res={'ok':True,'path':p,'frame_count':frame_count,'fps':fps,'duration':duration,'sampled':sampled,'valid':valid,'valid_ratio': (valid/sampled if sampled else 0.0),'first_frame_mean':fm,'last_frame_mean':lm,'sample_means_avg': (sum(means)/len(means) if means else None),'sample_stds_avg': (sum(stds)/len(stds) if stds else None)}
print(json.dumps(res, indent=2))
