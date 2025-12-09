import os
import numpy as np
import torch
from skimage import transform, feature
from segment_anything import sam_model_registry
import argparse

def window_image(vol, wl, ww):
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / max(ww, 1e-6) * 255.0).astype(np.uint8)

@torch.no_grad()
def infer_slice(model, img_3c, box_1024, device, H, W):
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    emb = model.image_encoder(img_t)
    box_t = torch.as_tensor(box_1024, dtype=torch.float, device=emb.device)
    if len(box_t.shape) == 2:
        box_t = box_t[:, None, :]
    se, de = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=emb,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=se,
        dense_prompt_embeddings=de,
        multimask_output=False
    )
    pred = torch.sigmoid(logits)
    pred = torch.nn.functional.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

def bbox_from_mask(mask, shift=5):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([[0, 0, mask.shape[1], mask.shape[0]]])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    H, W = mask.shape
    x_min = max(0, x_min - shift)
    x_max = min(W, x_max + shift)
    y_min = max(0, y_min - shift)
    y_max = min(H, y_max + shift)
    return np.array([[x_min, y_min, x_max, y_max]])

def auto_bbox(slice_img):
    img = slice_img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    blobs = feature.blob_log(img, max_sigma=15, num_sigma=10, threshold=0.1)
    if len(blobs) == 0:
        h, w = img.shape
        return np.array([[w//4, h//4, 3*w//4, 3*h//4]])
    blobs = blobs[np.argsort(-blobs[:, 2])]
    y, x, r = blobs[0]
    r = int(max(10, min(64, r * 3)))
    x_min = max(0, int(x - r))
    x_max = min(slice_img.shape[1], int(x + r))
    y_min = max(0, int(y - r))
    y_max = min(slice_img.shape[0], int(y + r))
    return np.array([[x_min, y_min, x_max, y_max]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_path", type=str, default="assets\\LNDb_nii\\LNDb-0001.nii.gz")
    ap.add_argument("--checkpoint", type=str, default="work_dir\\MedSAM\\medsam_vit_b.pth")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--save_dir", type=str, default="outputs")
    ap.add_argument("--bbox", type=str, default="")
    ap.add_argument("--slice_z", type=int, default=-1)
    ap.add_argument("--wl", type=float, default=-600.0)
    ap.add_argument("--ww", type=float, default=1500.0)
    args = ap.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    from SimpleITK import ReadImage, GetArrayFromImage, GetImageFromArray, WriteImage

    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device).eval()

    img_sitk = ReadImage(args.ct_path)
    vol = GetArrayFromImage(img_sitk).astype(np.float32)
    vol_w = window_image(vol, args.wl, args.ww)
    Z, H, W = vol_w.shape
    seg = np.zeros_like(vol_w, dtype=np.uint8)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.slice_z >= 0:
        z0 = args.slice_z
    else:
        z0 = int(np.argmax(vol_w.std(axis=(1,2))))

    if args.bbox:
        box_np = np.array([[int(x) for x in args.bbox.split(",")]])
    else:
        box_np = auto_bbox(vol_w[z0])

    bbox_dict = {}
    for z in range(z0, Z):
        img_2d = vol_w[z]
        img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
        if z == z0:
            box_1024 = box_np / np.array([W, H, W, H]) * 1024
        elif z in bbox_dict:
            box_1024 = bbox_dict[z] / np.array([W, H, W, H]) * 1024
        else:
            prev = seg[z - 1]
            if np.max(prev) > 0:
                prev_1024 = transform.resize(prev, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                box_1024 = bbox_from_mask(prev_1024)
            else:
                box_1024 = box_np / np.array([W, H, W, H]) * 1024
        seg2d = infer_slice(model, img_3c, box_1024, device, H, W)
        seg[z, seg2d > 0] = 1
        bbox_dict[z] = box_1024 / 1024 * np.array([W, H, W, H])

    for z in range(z0 - 1, -1, -1):
        img_2d = vol_w[z]
        img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
        if z in bbox_dict:
            box_1024 = bbox_dict[z] / np.array([W, H, W, H]) * 1024
        else:
            nxt = seg[z + 1]
            if np.max(nxt) > 0:
                nxt_1024 = transform.resize(nxt, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                box_1024 = bbox_from_mask(nxt_1024)
            else:
                box_1024 = box_np / np.array([W, H, W, H]) * 1024
        seg2d = infer_slice(model, img_3c, box_1024, device, H, W)
        seg[z, seg2d > 0] = 1
        bbox_dict[z] = box_1024 / 1024 * np.array([W, H, W, H])

    out = GetImageFromArray(seg)
    out.CopyInformation(img_sitk)
    WriteImage(out, os.path.join(args.save_dir, "seg_" + os.path.basename(args.ct_path)))

if __name__ == "__main__":
    main()
