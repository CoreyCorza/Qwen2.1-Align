# Qwen2.1 Align

ComfyUI/custom_nodes  
`git clone https://github.com/CoreyCorza/Qwen2.1-Align.git`  


Attempt to reduce Qwen2.1 (and other editing models) pixel shift.    
Uses OpenCV SIFT feature matching and RANSAC to estimate transform correction.     
Significant edits might cause it to fail as it requires before/after images to contain (some) matching areas needed for realignment.  



<img width="723" height="499" alt="image" src="https://github.com/user-attachments/assets/06dc4125-75c7-4828-9552-d90692e035e7" />






