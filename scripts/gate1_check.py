import sys, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
M="models/"
I="/tmp/claude-0/imgs/"
h=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=M+"hand_landmarker.task"),num_hands=2))
p=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=M+"pose_landmarker_lite.task")))
f=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=M+"face_landmarker.task")))
print("mediapipe",mp.__version__)
for n in ["thumb_up","pointing_up","woman_hands","pose","portrait","male_full_height_hands"]:
    im=mp.Image.create_from_file(I+n+".jpg")
    hr=h.detect(im); pr=p.detect(im); fr=f.detect(im)
    print(n, "hands:",[(len(l),c[0].category_name) for l,c in zip(hr.hand_landmarks,hr.handedness)], "pose:",[len(x) for x in pr.pose_landmarks], "face:",[len(x) for x in fr.face_landmarks])
