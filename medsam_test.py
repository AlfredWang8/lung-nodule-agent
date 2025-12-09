import os, torch
from medsam_tools.medsam_client import segment_2d_batch
from medsam_tools.medsam_client import segment_3d_volume


def get_device():
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
        else:
            return "cpu"
    except Exception:
        return "cpu"

def gpu_smoke_test():
    try:
        a = torch.rand((1024, 1024), device="cuda")
        b = torch.mm(a, a)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False

if __name__ == "__main__":
    dev = get_device()
    print("torch", torch.__version__)
    print("cuda_ver", getattr(torch.version, "cuda", None))
    print("is_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if dev == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        print(f"GPU: {name} (index {idx})")
        print("GPU smoke test:", "passed" if gpu_smoke_test() else "failed")
    else:
        print("GPU unavailable; using CPU")


    os.makedirs(r"outputs_nii", exist_ok=True)

    segment_3d_volume(
        ct_path=r"data\\3D\\LNDb-0001.nii.gz",
        out_path=r"outputs_nii\\seg_LNDb-0001.nii.gz",
        checkpoint=r"medsam_tools\\MedSAM\\medsam_vit_b.pth",
        device=dev,
        slice_z=220,
        bbox="120,140,170,190",
        overlay_out_path=r"outputs_nii\\overlay_LNDb-0001.nii.gz",
    )


    # segment_2d_batch(
    #     input_dir=r"data\\2D",
    #     output_dir=r"outputs_png",
    #     checkpoint=r"medsam_tools\\MedSAM\\medsam_vit_b.pth",
    #     device=dev,
    #     threshold=0.2,
    #     save_overlay=True,
    # )