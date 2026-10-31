# PLUMBING ONLY: real photo (real MediaPipe detections), artificially shifted to make wrist motion.
import cv2, numpy as np, sys
img=cv2.imread("/tmp/claude-0/imgs/male_full_height_hands.jpg"); h,w=img.shape[:2]
img=cv2.resize(img,(int(w*480/h),480)); h,w=img.shape[:2]
out=sys.argv[1]; fps=15
vw=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
def shifted(dx):
    M=np.float32([[1,0,dx],[0,1,0]]); return cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REPLICATE)
for _ in range(20): vw.write(shifted(0))
for i in range(25): vw.write(shifted(i*4))      # ~ moving
for _ in range(30): vw.write(shifted(96))
vw.release(); print(out,w,h,75,"frames")
