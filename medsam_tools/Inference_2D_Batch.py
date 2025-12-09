import os
import glob
import numpy as np
import torch
from skimage import io, transform, feature
import torch.nn.functional as F
from segment_anything import sam_model_registry
import argparse

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W, threshold=0.5):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]
    se, de = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de,
        multimask_output=False,
    )
    pred = torch.sigmoid(logits)
    pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
    return (pred.squeeze().cpu().numpy() > float(threshold)).astype(np.uint8)

def auto_bbox(img):
    if img.ndim == 3:
        gray = img[..., 0].astype(np.float32)
    else:
        gray = img.astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    blobs = feature.blob_log(gray, max_sigma=15, num_sigma=10, threshold=0.1)
    if len(blobs) == 0:
        h, w = gray.shape
        return np.array([[w//4, h//4, 3*w//4, 3*h//4]])
    blobs = blobs[np.argsort(-blobs[:, 2])]
    y, x, r = blobs[0]
    r = int(max(10, min(64, r * 3)))
    x_min = max(0, int(x - r))
    x_max = min(gray.shape[1], int(x + r))
    y_min = max(0, int(y - r))
    y_max = min(gray.shape[0], int(y + r))
    return np.array([[x_min, y_min, x_max, y_max]])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default="assets\\2D")
    p.add_argument("--output_dir", type=str, default="outputs_png")
    p.add_argument("--checkpoint", type=str, default="medsam_tools\\MedSAM\\medsam_vit_b.pth")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bbox", type=str, default="")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--save_overlay", action="store_true")
    args = p.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device).eval()

    for path in files:
        img_np = io.imread(path)
        if img_np.ndim == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
        img_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        if args.bbox:
            box_np = np.array([[int(x) for x in args.bbox.split(",")]])
        else:
            box_np = auto_bbox(img_3c)
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        with torch.no_grad():
            emb = model.image_encoder(img_t)
        seg = medsam_inference(model, emb, box_1024, H, W, threshold=args.threshold)
        out_name = os.path.join(args.output_dir, "seg_" + os.path.basename(path))
        io.imsave(out_name, (seg * 255).astype(np.uint8), check_contrast=False)
        if args.save_overlay:
            base = img_np if img_np.ndim == 3 else np.repeat(img_np[:, :, None], 3, axis=-1)
            mask3 = np.stack([seg]*3, axis=-1)
            overlay = (0.7 * base + 0.3 * (mask3 * np.array([255, 0, 0]))).astype(np.uint8)
            io.imsave(os.path.join(args.output_dir, "overlay_" + os.path.basename(path)), overlay, check_contrast=False)

if __name__ == "__main__":
    main()
