import mediapipe as mp, numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
M="models/"; I="/tmp/claude-0/imgs/"
h=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=M+"hand_landmarker.task"),num_hands=2))
p=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=M+"pose_landmarker_lite.task")))
for n in ["woman_hands","male_full_height_hands"]:
    im=mp.Image.create_from_file(I+n+".jpg")
    hr=h.detect(im); pr=p.detect(im)[0] if False else p.detect(im)
    pl=pr.pose_landmarks[0]
    print(n,"pose L-wrist(15) x=%.2f  R-wrist(16) x=%.2f"%(pl[15].x,pl[16].x))
    for l,c in zip(hr.hand_landmarks,hr.handedness):
        wx=l[0].x
        nearest = "pose15(L)" if abs(wx-pl[15].x)<abs(wx-pl[16].x) else "pose16(R)"
        print("   hand label=%s wrist x=%.2f nearest-pose-wrist=%s"%(c[0].category_name,wx,nearest))
h.close(); p.close()
