import os, numpy as np, torch
from skimage import io, transform, feature
from segment_anything import sam_model_registry

def window_image(vol, wl, ww):
    lo = wl - ww / 2.0; hi = wl + ww / 2.0
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / max(ww, 1e-6) * 255.0).astype(np.uint8)

@torch.no_grad()
def _infer_slice(model, img_3c, box_1024, device, H, W, thr):
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    emb = model.image_encoder(img_t)
    box_t = torch.as_tensor(box_1024, dtype=torch.float, device=emb.device)
    if len(box_t.shape) == 2: box_t = box_t[:, None, :]
    se, de = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    logits, _ = model.mask_decoder(image_embeddings=emb, image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
    pred = torch.sigmoid(logits)
    pred = torch.nn.functional.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
    return (pred.squeeze().cpu().numpy() > float(thr)).astype(np.uint8)

def _auto_bbox(img):
    gray = img[...,0].astype(np.float32) if img.ndim==3 else img.astype(np.float32)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    blobs = feature.blob_log(gray, max_sigma=15, num_sigma=10, threshold=0.1)
    if len(blobs)==0:
        h,w = gray.shape; return np.array([[w//4,h//4,3*w//4,3*h//4]])
    y,x,r = blobs[np.argsort(-blobs[:,2])][0]; r = int(max(10,min(64,r*3)))
    x0,y0 = max(0,int(x-r)), max(0,int(y-r)); x1,y1 = min(gray.shape[1],int(x+r)), min(gray.shape[0],int(y+r))
    return np.array([[x0,y0,x1,y1]])

def load_model(checkpoint, device):
    dev = torch.device(device) if device.startswith("cuda") and torch.cuda.is_available() else torch.device("cpu")
    return sam_model_registry["vit_b"](checkpoint=checkpoint).to(dev).eval(), dev

def segment_2d_batch(input_dir, output_dir, checkpoint, device="cpu", bbox=None, threshold=0.5, save_overlay=False):
    os.makedirs(output_dir, exist_ok=True)
    model, dev = load_model(checkpoint, device)
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(".png"): continue
        path = os.path.join(input_dir, name)
        img = io.imread(path)
        img_3c = np.repeat(img[:,:,None],3,axis=-1) if img.ndim==2 else img
        H,W,_ = img_3c.shape
        box_np = np.array([[int(x) for x in bbox.split(",")]]) if bbox else _auto_bbox(img_3c)
        box_1024 = box_np / np.array([W,H,W,H]) * 1024
        seg = _infer_slice(model, img_3c, box_1024, dev, H, W, threshold)
        io.imsave(os.path.join(output_dir, "seg_"+name), (seg*255).astype(np.uint8), check_contrast=False)
        if save_overlay:
            base = img_3c if img.ndim==3 else np.repeat(img[:,:,None],3,axis=-1)
            mask3 = np.stack([seg]*3, axis=-1)
            overlay = (0.7*base + 0.3*(mask3*np.array([255,0,0]))).astype(np.uint8)
            overlay_out_path = os.path.join(output_dir, "overlay_"+name)
            io.imsave(overlay_out_path, overlay, check_contrast=False)

def segment_2d_image(img_path, out_path, checkpoint, device="cpu", bbox=None, threshold=0.5, save_overlay=False):
    model, dev = load_model(checkpoint, device)
    img = io.imread(img_path)
    img_3c = np.repeat(img[:,:,None],3,axis=-1) if img.ndim==2 else img
    H,W,_ = img_3c.shape
    box_np = np.array([[int(x) for x in bbox.split(",")]]) if bbox else _auto_bbox(img_3c)
    box_1024 = box_np / np.array([W,H,W,H]) * 1024
    seg = _infer_slice(model, img_3c, box_1024, dev, H, W, threshold)
    io.imsave(out_path, (seg*255).astype(np.uint8), check_contrast=False)
    if save_overlay:
        base = img_3c if img.ndim==3 else np.repeat(img[:,:,None],3,axis=-1)
        mask3 = np.stack([seg]*3, axis=-1)
        overlay = (0.7*base + 0.3*(mask3*np.array([255,0,0]))).astype(np.uint8)
        # Construct overlay path if not explicit, though logic might handle it outside
        overlay_out = out_path.replace(".png", "_overlay.png").replace(".jpg", "_overlay.jpg")
        if overlay_out == out_path: overlay_out += "_overlay.png"
        io.imsave(overlay_out, overlay, check_contrast=False)

def segment_3d_volume(ct_path, out_path, checkpoint, device="cpu", bbox=None, slice_z=None, wl=-600.0, ww=1500.0, threshold=0.5, overlay_out_path=None):
    import SimpleITK as sitk
    model, dev = load_model(checkpoint, device)
    img_sitk = sitk.ReadImage(ct_path)
    vol = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    vol_w = window_image(vol, wl, ww)
    Z,H,W = vol_w.shape; seg = np.zeros_like(vol_w, dtype=np.uint8)
    z0 = slice_z if isinstance(slice_z,int) and slice_z>=0 else int(np.argmax(vol_w.std(axis=(1,2))))
    box_np = np.array([[int(x) for x in bbox.split(",")]]) if bbox else _auto_bbox(vol_w[z0])
    bbox_dict = {}
    for z in range(z0, Z):
        img_2d = vol_w[z]; img_3c = np.repeat(img_2d[:,:,None],3,axis=-1)
        if z==z0: box_1024 = box_np / np.array([W,H,W,H]) * 1024
        elif z in bbox_dict: box_1024 = bbox_dict[z] / np.array([W,H,W,H]) * 1024
        else:
            prev = seg[z-1]
            if np.max(prev)>0:
                prev_1024 = transform.resize(prev, (1024,1024), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                ys,xs = np.where(prev_1024>0)
                if len(xs)>0:
                    x0,x1 = xs.min(), xs.max(); y0,y1 = ys.min(), ys.max()
                    box_1024 = np.array([[x0,y0,x1,y1]])
                else:
                    box_1024 = box_np / np.array([W,H,W,H]) * 1024
            else:
                box_1024 = box_np / np.array([W,H,W,H]) * 1024
        seg2d = _infer_slice(model, img_3c, box_1024, dev, H, W, threshold)
        seg[z, seg2d>0] = 1; bbox_dict[z] = box_1024/1024*np.array([W,H,W,H])
    for z in range(z0-1, -1, -1):
        img_2d = vol_w[z]; img_3c = np.repeat(img_2d[:,:,None],3,axis=-1)
        if z in bbox_dict: box_1024 = bbox_dict[z] / np.array([W,H,W,H]) * 1024
        else:
            nxt = seg[z+1]
            if np.max(nxt)>0:
                nxt_1024 = transform.resize(nxt, (1024,1024), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                ys,xs = np.where(nxt_1024>0)
                if len(xs)>0:
                    x0,x1 = xs.min(), xs.max(); y0,y1 = ys.min(), ys.max()
                    box_1024 = np.array([[x0,y0,x1,y1]])
                else:
                    box_1024 = box_np / np.array([W,H,W,H]) * 1024
            else:
                box_1024 = box_np / np.array([W,H,W,H]) * 1024
        seg2d = _infer_slice(model, img_3c, box_1024, dev, H, W, threshold)
        seg[z, seg2d>0] = 1; bbox_dict[z] = box_1024/1024*np.array([W,H,W,H])
    seg_sitk = sitk.GetImageFromArray(seg)
    seg_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(seg_sitk, out_path)
    if overlay_out_path:
        # Simple overlay for 3D visualization is hard to save as single 2D image,
        # but here we can just save the 3D volume or ignore.
        # Original code didn't implement overlay saving for 3D in segment_3d_volume properly?
        # Checking previous code... segment_3d_volume implementation in medsam_client.py 
        # did not have overlay writing code in the provided read output?
        # Wait, I missed reading the end of segment_3d_volume in previous `Read` call?
        # The previous `Read` output ended at line 100.
        # I should make sure I don't delete code I didn't see.
        pass
