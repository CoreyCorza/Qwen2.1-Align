import cv2
import numpy as np
import torch
from comfy_api.latest import io


def _features(gray):
    detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    points, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        return np.empty((0, 2), np.float32), None
    # Keep textured subjects from consuming the entire matching budget.
    height, width = gray.shape
    counts = {}
    selected = []
    for index in sorted(range(len(points)), key=lambda i: points[i].response, reverse=True):
        x, y = points[index].pt
        cell = (min(5, int(x * 6 / width)), min(5, int(y * 6 / height)))
        if counts.get(cell, 0) < 100:
            selected.append(index)
            counts[cell] = counts.get(cell, 0) + 1
    return np.float32([points[i].pt for i in selected]), descriptors[selected]


def _align(original, edited, edge_fill="Original image"):
    height, width = original.shape[:2]
    edit_height, edit_width = edited.shape[:2]
    scale = min(1.0, 1400 / max(height, width))
    work_width, work_height = max(1, round(width * scale)), max(1, round(height * scale))

    def gray(image):
        small = cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    reference_points, reference_desc = _features(gray(original))
    edit_points, edit_desc = _features(gray(edited))

    def unchanged(reason):
        result = edited if (edit_height, edit_width) == (height, width) else cv2.resize(
            edited, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return np.clip(result, 0, 1), f"Alignment skipped: {reason}. Edited image returned at original resolution."

    if reference_desc is None or edit_desc is None or min(len(reference_desc), len(edit_desc)) < 12:
        return unchanged("too few distinctive details")
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def good_matches(a, b):
        return {pair[0].queryIdx: pair[0].trainIdx
                for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance}

    forward = good_matches(edit_desc, reference_desc)
    backward = good_matches(reference_desc, edit_desc)
    pairs = [(a, b) for a, b in forward.items() if backward.get(b) == a]
    if len(pairs) < 12:
        return unchanged(f"only {len(pairs)} reliable matches")
    source = np.float32([edit_points[a] for a, b in pairs])
    target = np.float32([reference_points[b] for a, b in pairs])
    matrix, inliers = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.5,
        maxIters=5000, confidence=0.999, refineIters=20)
    if matrix is None:
        return unchanged("no consistent global transform")
    keep = inliers.ravel().astype(bool)
    count = int(keep.sum())
    spread = np.ptp(target[keep], axis=0) / [work_width, work_height]
    coverage = cv2.contourArea(cv2.convexHull(target[keep])) / (work_width * work_height)
    singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
    if count < 12 or count / len(pairs) < 0.3 or min(spread) < 0.3 or coverage < 0.08:
        return unchanged(f"matches lack sufficient agreement or coverage ({count}/{len(pairs)} inliers)")
    if np.linalg.det(matrix[:, :2]) <= 0 or singular.min() < 0.75 or singular.max() > 1.33:
        return unchanged("estimated stretch is too large for a small edit correction")

    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    residual = float(np.median(np.linalg.norm(predicted[keep] - target[keep], axis=1)))
    # Convert the estimated transform to full-resolution pixel-center coordinates.
    source_scale = np.array([work_width / edit_width, work_height / edit_height])
    target_scale = np.array([width / work_width, height / work_height])
    full = matrix.copy()
    full[:, :2] = target_scale[:, None] * matrix[:, :2] * source_scale[None, :]
    full[:, 2] = target_scale * (matrix[:, :2] @ (0.5 * source_scale - 0.5) + matrix[:, 2] + 0.5) - 0.5
    aligned = cv2.warpAffine(edited, full, (width, height), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT_101)
    if edge_fill == "Original image":
        coverage_mask = cv2.warpAffine(
            np.ones((edit_height, edit_width), np.float32), full, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = (coverage_mask >= 0.999).astype(np.uint8)
        if not valid.all():
            distance = cv2.distanceTransform(valid, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            weight = np.clip(distance / 4.0, 0, 1)
            weight = (weight * weight * (3.0 - 2.0 * weight))[..., None]
            aligned = aligned * weight + original * (1.0 - weight)
    sx = np.linalg.norm(matrix[:, 0])
    sy = np.linalg.norm(matrix[:, 1])
    angle = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    report = (f"Aligned: {count}/{len(pairs)} matches agree; coverage {coverage:.0%}; "
              f"median error {residual:.2f} working pixels.\n"
              f"Correction relative to resized edit: X scale {sx:.5f}, Y scale {sy:.5f}, "
              f"rotation {angle:.3f} degrees; translation "
              f"{matrix[0, 2] * target_scale[0]:.2f}, {matrix[1, 2] * target_scale[1]:.2f} px.\n"
              + ("Exposed borders use original pixels with a 4 px feather."
                 if edge_fill == "Original image" else "Exposed borders use reflected edited pixels."))
    return np.clip(aligned, 0, 1), report


class Qwen21Align(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Qwen21Align", display_name="Qwen2.1 Align",
            category="Qwen2.1/image",
            description="Automatically align an edited image to its original using matched details and a global affine transform. Corrects shift, rotation, independent width/height stretch, and shear. No mask needed. Outputs RGB at the original resolution. Choose reflected or original pixels for exposed borders. If matching is unreliable, returns the resized edit and explains why in the report.",
            inputs=[io.Image.Input("original_image"), io.Image.Input("edited_image"),
                    io.Combo.Input("edge_fill", options=["Original image", "Reflect"], default="Original image",
                                   optional=True, tooltip="Fill exposed borders by reflection, or use the original image with a 4 pixel feather into the aligned edit.")],
            outputs=[io.Image.Output(display_name="aligned_image"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, original_image: torch.Tensor, edited_image: torch.Tensor, edge_fill="Original image") -> io.NodeOutput:
        for image in (original_image, edited_image):
            if image.ndim != 4 or image.shape[-1] not in (3, 4) or image.shape[0] == 0:
                raise ValueError("Qwen2.1 Align expects nonempty RGB or RGBA IMAGE batches.")
        if original_image.shape[0] not in (1, edited_image.shape[0]):
            raise ValueError("Use one original image for all edits, or matching original and edited batch sizes.")
        results, reports = [], []
        for index in range(edited_image.shape[0]):
            original = original_image[0 if original_image.shape[0] == 1 else index, ..., :3].detach().cpu().float().numpy()
            edited = edited_image[index, ..., :3].detach().cpu().float().numpy()
            aligned, report = _align(original, edited, edge_fill)
            results.append(torch.from_numpy(np.ascontiguousarray(aligned)))
            reports.append(f"Image {index + 1}: {report}")
        return io.NodeOutput(torch.stack(results).to(edited_image), "\n\n".join(reports))


NODE_CLASS_MAPPINGS = {"Qwen21Align": Qwen21Align}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen21Align": "Qwen2.1 Align"}

