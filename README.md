# Learning-to-Reconstruct-3D-Faces-by-Watching-TV
3D Vision course project from ETHZ, generate person-specific high-quality 3D face models from TV videos 
# Face segmentation
Use own repo
```Shell
git clone git@github.com:xiyichen/face-parsing.PyTorch.git
```
then
```Shell
from pred_mask import evaluate
evaluate(respth=RESULT_SAVE_PATH,
         dspth=INPUT_IMAGE_PATH,
         model_path=PRETRAINED_MODEL_PATH,
         save_masks=TRUE/FALSE,
         save_imgs=TRUE/FALSE)
```
