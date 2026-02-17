download wireframe data:
smpl-x model (from internet; Max Flanks)

project/
  body_fit.py
  smpl_models/
    simplx/
      SMPLX_NEUTRAL.pkl   (or SMPLX model files etc.)


cd "/Users/youngwooksohn/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Dtop/codes/AI/Wireframe_3d_body"
source venv-body/bin/activate

image: (ex. ww2_soldier2.png)

ipython body_fit.py \
  --mode fit_photo \
  --smpl_model_folder ./smpl_models \
  --pose_task ./pose_landmarker_heavy.task \
  --image ww2_soldier2.png \
  --export_obj fit.obj \
  --view

or: (fewer interations)
python body_fit.py \
  --mode image \
  --smpl_model_folder ./smpl_models \
  --pose_task ./pose_landmarker_heavy.task \
  --image wwII_soldier1.jpg \
  --export_obj fit.obj \
  --view \
  --iters 150

python body_fit.py --mode fit_photo --smpl_model_folder ./smpl_models --image ww2_soldier2.png --pose_task ./pose_landmarker_heavy.task --view --export_obj fit.obj



video:


python body_fit.py \
  --mode animate \
  --smpl_model_folder ./smpl_models \
  --view

Stop with Ctrl+C.




