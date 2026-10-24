# Qwen2.1 Align

A standalone ComfyUI node that aligns an edited image back to its original. Built for the small shifts and stretches introduced by Qwen image editing, it also works with other editors when enough original detail remains.

No mask, model download, or GPU inference is required. Uses OpenCV SIFT feature matching and RANSAC to estimate one global affine transform, correcting translation, rotation, independent width/height scaling, and shear.

## Installation

Requires a current ComfyUI installation with `comfy_api.latest` support and its existing PyTorch installation.

1. Open a terminal in `ComfyUI/custom_nodes` and run:

   ```sh
   git clone https://github.com/CoreyCorza/Qwen2.1-Align.git
   ```

2. Install dependencies using **ComfyUI's Python environment**:

   ```sh
   python -m pip install -r Qwen2.1-Align/requirements.txt
   ```

   For Windows portable, run this instead from the portable root (the folder containing `python_embeded`):

   ```powershell
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\Qwen2.1-Align\requirements.txt
   ```

3. Restart ComfyUI and search for **Qwen2.1 Align**, under `Qwen2.1/image`.

## Usage

Connect the image before editing to **original_image** and the editor's output to **edited_image**. Send **aligned_image** to a preview, save, or subsequent processing node. The **report** output describes matching quality and the correction applied.

**edge_fill** controls exposed borders:

- **Original image** (default): fills from the original with a 4-pixel feather into the aligned edit. This may reintroduce original content if the edit intentionally changed the edges.
- **Reflect**: mirrors edited pixels at the border.

Output is RGB at the original image's resolution. RGBA inputs have alpha removed. A single original can align a batch of edits; otherwise, batch sizes must match. Output retains the edited tensor's device and dtype, while matching runs on CPU.

## Limits

Designed for small global transformations, not local deformation or major perspective changes. Matching requires recognizable details distributed across the image. Extensive edits, repeated textures, or low-detail images can prevent reliable alignment. When the checks reject a transform, the node returns the edited image resized to the original resolution and explains the reason in the report. Accepted matches are an estimate, not a guarantee of correct alignment.

The node preserves the edit; it does not detect and composite an edit mask. Only the exposed border and its feather use the original when that fill mode is selected.

Standalone copy of the Corza alignment node with a separate node ID, so both can be installed together.
