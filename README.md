# Qwen2.1 Align

ComfyUI/custom_nodes  
`git clone https://github.com/CoreyCorza/Qwen2.1-Align.git`  


Attempt to reduce Qwen2.1 (and other editing models) pixel shift.    
Uses OpenCV SIFT feature matching and RANSAC to estimate transform correction.     
Significant edits might cause it to fail as it requires before/after images to contain (mostly) matching areas needed for realignment.  









