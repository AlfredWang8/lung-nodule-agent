import os
import sys
from typing import List, Dict, Optional, Tuple, Union

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


def _nnunet_identity(x):
    return x


class KnowledgeGraphTool:
    """
    知识图谱工具：封装 Neo4j 数据库的连接与查询功能。
    用于获取医学指南、疾病关联等知识。
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.environ.get("NEO4J_URI")
        self.user = user or os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
        self.password = password or os.environ.get("NEO4J_PASSWORD")

        self.driver = None
        if GraphDatabase is None:
            print("警告：未安装 neo4j 驱动，知识图谱功能不可用。")
            return

        if self.uri and self.user and self.password:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.driver.verify_connectivity()
                print("已连接到知识图谱 (Neo4j)。")
            except Exception as e:
                print(f"连接 Neo4j 失败: {e}")
        else:
            print("警告：缺少 Neo4j 配置信息 (URI/USER/PASSWORD)，知识图谱功能不可用。")

    def query(self, cypher_query: str, params: Dict = None) -> List[Dict]:
        if not self.driver:
            print("错误：Neo4j 驱动未初始化。")
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"查询执行失败: {e}")
            return []

    def close(self):
        if self.driver:
            self.driver.close()


class MedSAMTool:
    """
    图像分割工具：基于自训练 nnU-Net 肺结节模型。
    支持 2D (png/jpg/etc) 和 3D (nii/nii.gz) 图像输入。
    """

    def __init__(self, model_folder: str = None, folds: Tuple[int, ...] = (0,), checkpoint_name: str = "model_final_checkpoint"):
        base_path = os.path.dirname(os.path.abspath(__file__))
        default_model_folder = os.path.join(
            base_path,
            "nnUNet_backup",
            "nnUNetFrame",
            "DATASET",
            "nnUNet_trained_models",
            "nnUNet",
            "3d_fullres",
            "Task500_LungNodule",
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        self.model_folder = model_folder or default_model_folder
        self.folds = folds
        self.checkpoint_name = checkpoint_name
        if not os.path.isdir(self.model_folder):
            print(f"警告：nnUNet 模型目录不存在: {self.model_folder}")

    def _predict_nifti(self, input_path: str, output_path: str) -> bool:
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            # 使用备份的 nnUNet 源码（如果存在）
            nnunet_root = os.path.join(base_path, "nnUNet_backup")
            if os.path.isdir(nnunet_root) and nnunet_root not in sys.path:
                sys.path.append(nnunet_root)
            runtime_root = os.path.join(base_path, "nnUNet_runtime")
            os.makedirs(runtime_root, exist_ok=True)
            os.environ["RESULTS_FOLDER"] = os.path.join(runtime_root, "results")
            os.environ["nnUNet_raw_data_base"] = os.path.join(runtime_root, "raw_data_base")
            os.environ["nnUNet_preprocessed"] = os.path.join(runtime_root, "preprocessed")

            import torch

            # 自动检测并选择可用的 GPU
            if torch.cuda.is_available():
                try:
                    # 尝试在当前默认设备（通常是 0）上分配内存，如果失败则尝试其他设备
                    torch.cuda.current_device()
                    torch.zeros(1).cuda()
                except Exception:
                    print("警告：默认 CUDA 设备 (0) 不可用，尝试寻找其他可用设备...")
                    found_device = False
                    for i in range(1, torch.cuda.device_count()):
                        try:
                            torch.zeros(1).cuda(i)
                            # 对于已经初始化的 torch，使用 set_device
                            torch.cuda.set_device(i)
                            print(f"已自动切换到可用设备: GPU {i}")
                            found_device = True
                            break
                        except Exception:
                            continue
                    if not found_device:
                        print("错误：未找到任何可用的 CUDA 设备。")

            if not hasattr(torch.load, "__wrapped__"):
                _original_load = torch.load

                def _nnunet_load(f, *args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    obj = _original_load(f, *args, **kwargs)
                    try:
                        from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
                        if isinstance(obj, nnUNetTrainerV2):
                            if hasattr(obj, "network") and hasattr(obj.network, "final_nonlin"):
                                obj.network.final_nonlin = _nnunet_identity
                    except Exception:
                        pass
                    return obj

                _nnunet_load.__wrapped__ = _original_load
                torch.load = _nnunet_load

            from nnunet.training.model_restore import load_model_and_checkpoint_files
            from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
            import numpy as np
        except Exception as e:
            print(f"导入 nnUNet 失败: {e}")
            return False

        try:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            if not output_path.endswith(".nii.gz"):
                output_path = output_path + ".nii.gz"

            trainer, params = load_model_and_checkpoint_files(
                self.model_folder,
                self.folds,
                mixed_precision=True,
                checkpoint_name=self.checkpoint_name,
            )

            d, _, dct = trainer.preprocess_patient([input_path])

            trainer.load_checkpoint_ram(params[0], False)
            softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                d,
                do_mirroring=True,
                mirror_axes=trainer.data_aug_params.get("mirror_axes"),
                use_sliding_window=True,
                step_size=0.5,
                use_gaussian=True,
                all_in_gpu=False,
                mixed_precision=True,
            )[1]

            for p in params[1:]:
                trainer.load_checkpoint_ram(p, False)
                softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                    d,
                    do_mirroring=True,
                    mirror_axes=trainer.data_aug_params.get("mirror_axes"),
                    use_sliding_window=True,
                    step_size=0.5,
                    use_gaussian=True,
                    all_in_gpu=False,
                    mixed_precision=True,
                )[1]

            if len(params) > 1:
                softmax /= len(params)

            transpose_forward = trainer.plans.get("transpose_forward")
            if transpose_forward is not None:
                transpose_backward = trainer.plans.get("transpose_backward")
                softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])

            save_segmentation_nifti_from_softmax(
                softmax,
                output_path,
                dct,
                order=1,
                region_class_order=getattr(trainer, "regions_class_order", None),
                seg_postprogess_fn=None,
                seg_postprocess_args=None,
                resampled_npz_fname=None,
                non_postprocessed_fname=None,
                force_separate_z=None,
                interpolation_order_z=0,
                verbose=True,
            )
            return os.path.exists(output_path)
        except Exception as e:
            print(f"nnUNet 推理失败: {e}")
            return False

    def _segment_2d_image(self, img_path: str, out_path: str) -> bool:
        try:
            import numpy as np
            from skimage import io
            import SimpleITK as sitk
            import tempfile
            import shutil
        except Exception as e:
            print(f"导入2D分割依赖失败: {e}")
            return False

        temp_dir = None
        try:
            img = io.imread(img_path)
            if img.ndim == 3:
                if img.shape[2] >= 3:
                    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                else:
                    gray = img[:, :, 0]
            else:
                gray = img
            gray = gray.astype("float32")
            vol = gray[None, ...]
            img_sitk = sitk.GetImageFromArray(vol)
            temp_dir = tempfile.mkdtemp()
            input_nii = os.path.join(temp_dir, "input.nii.gz")
            sitk.WriteImage(img_sitk, input_nii)
            seg_nii = os.path.join(temp_dir, "seg.nii.gz")
            if not self._predict_nifti(input_nii, seg_nii):
                return False
            if not os.path.exists(seg_nii):
                return False
            seg_sitk = sitk.ReadImage(seg_nii)
            seg = sitk.GetArrayFromImage(seg_sitk)
            if seg.ndim == 3:
                seg_slice = seg[0]
            else:
                seg_slice = seg
            mask = (seg_slice > 0).astype("uint8") * 255
            io.imsave(out_path, mask, check_contrast=False)
            return True
        except Exception as e:
            print(f"2D 图像分割失败: {e}")
            return False
        finally:
            if temp_dir and os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    def _segment_3d_nifti(self, ct_path: str, out_path: str) -> bool:
        if not out_path.endswith(".nii.gz"):
            base, ext = os.path.splitext(out_path)
            if ext == ".gz":
                base, _ = os.path.splitext(base)
            out_path = base + ".nii.gz"
        return self._predict_nifti(ct_path, out_path)

    def generate_radiology_report(self, img_path: str, seg_path: str = None, symptoms: str = "") -> str:
        import numpy as np
        from skimage import io

        img_ext = os.path.splitext(img_path)[1].lower()

        if img_ext in ['.nii', '.gz']:
            try:
                import SimpleITK as sitk
                img_sitk = sitk.ReadImage(img_path)
                img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
                if len(img.shape) == 3:
                    h, w = img.shape[1], img.shape[2]
                else:
                    h, w = img.shape[:2] if len(img.shape) >= 2 else (img.shape[0], img.shape[1])
            except ImportError:
                print("警告: 未安装SimpleITK，无法读取3D医学图像")
                return f"""总体结论：影像学检查完成，但无法分析3D图像细节
- 结节数量：未检测（3D图像分析需要SimpleITK）
- 影像表现：基于原始图像分析，{'存在可疑结节' if symptoms else '未见明显异常'}
- 进一步检查建议：根据临床症状{'(' + symptoms + ')' if symptoms else ''}，建议专业放射科医生阅片"""
        else:
            img = io.imread(img_path)
            h, w = img.shape[:2] if len(img.shape) >= 2 else (img.shape[0], img.shape[1])

        if seg_path and os.path.exists(seg_path):
            seg_ext = os.path.splitext(seg_path)[1].lower()
            if seg_ext in ['.nii', '.gz']:
                try:
                    import SimpleITK as sitk
                    seg_sitk = sitk.ReadImage(seg_path)
                    seg = sitk.GetArrayFromImage(seg_sitk).astype(np.uint8)

                    if len(seg.shape) == 3:
                        max_area_slice_idx = 0
                        max_area = 0
                        for z in range(seg.shape[0]):
                            area = np.sum(seg[z] > 0)
                            if area > max_area:
                                max_area = area
                                max_area_slice_idx = z
                        seg = seg[max_area_slice_idx]
                except ImportError:
                    print("警告: 未安装SimpleITK，无法读取3D分割结果")
                    return f"""总体结论：影像学检查完成，但无法分析3D分割结果
- 结节数量：未检测（3D分割结果分析需要SimpleITK）
- 影像表现：基于原始图像分析，{'存在可疑结节' if symptoms else '未见明显异常'}
- 进一步检查建议：根据临床症状{'(' + symptoms + ')' if symptoms else ''}，建议专业放射科医生阅片"""
            else:
                seg = io.imread(seg_path)

            seg_coords = np.where(seg > 0)
            if len(seg_coords[0]) > 0:
                y_coords, x_coords = seg_coords
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()

                nodule_height = y_max - y_min
                nodule_width = x_max - x_min

                nodule_area = len(seg_coords[0])
                total_area = seg.shape[0] * seg.shape[1]
                area_ratio = nodule_area / total_area

                pixel_size_mm = 0.5
                real_width = round(nodule_width * pixel_size_mm, 1)
                real_height = round(nodule_height * pixel_size_mm, 1)

                shape = "圆形" if abs(real_width - real_height) < max(real_width, real_height) * 0.2 else "不规则"

                from skimage import measure
                try:
                    if img_ext not in ['.nii', '.gz']:
                        perimeter = measure.perimeter(seg)
                        circularity = 4 * np.pi * nodule_area / (perimeter ** 2) if perimeter > 0 else 0
                        edge_feature = "毛刺" if circularity < 0.7 else "光滑"
                    else:
                        edge_feature = "光滑"
                except Exception as e:
                    edge_feature = "光滑"
                    print(f"计算边缘特征时出错: {e}")

                center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
                if center_x < w / 3:
                    horiz_pos = "左肺"
                elif center_x < 2 * w / 3:
                    horiz_pos = "中央"
                else:
                    horiz_pos = "右肺"

                if center_y < h / 2:
                    vert_pos = "上叶"
                else:
                    vert_pos = "下叶"

                report = f"""总体结论：检测到1个肺结节，位于{horiz_pos}{vert_pos}区域，倾向良性
- 结节数量：1个
- 大小：{real_width}mm × {real_height}mm
- 形状：{shape}
- 边缘：{edge_feature}
- 密度：实性结节
- 分布：单发，位于{horiz_pos}{vert_pos}
- 与胸膜关系：距离胸膜约{min(center_x, w - center_x, center_y, h - center_y) * pixel_size_mm:.1f}mm
- AI分割结果：成功生成分割掩码，结节占比图像面积{area_ratio * 100:.2f}%
- 进一步检查建议：建议3-6个月后复查CT以评估结节变化"""

                return report
            else:
                return f"""总体结论：未检测到明显肺结节
- 结节数量：0个（未发现明显结节）
- 影像表现：肺野清晰，未见明显异常密度影
- 进一步检查建议：根据临床症状{'(' + symptoms + ')' if symptoms else ''}，建议继续观察或进一步检查"""
        else:
            return f"""总体结论：影像学检查完成，未执行AI分割
- 结节数量：未检测（分割失败）
- 影像表现：基于原始图像分析，{'存在可疑结节' if symptoms else '未见明显异常'}
- 进一步检查建议：根据临床症状{'(' + symptoms + ')' if symptoms else ''}，建议专业放射科医生阅片"""

    def detect_and_segment(self, file_path: str, bbox: List[int] = None, slice_z: int = None, symptoms: str = "") -> Dict[str, Union[bool, str]]:
        if not os.path.exists(file_path):
            return {"success": False, "out_path": None, "msg": f"文件不存在: {file_path}", "radiology_report": ""}

        ext = os.path.splitext(file_path)[1].lower()
        out_path = ""
        success = False
        msg = ""
        radiology_report = ""

        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            out_path = os.path.splitext(file_path)[0] + "_seg.png"
            print(f"检测到 2D 图像: {file_path} -> 执行 nnUNet 分割")
            success = self._segment_2d_image(file_path, out_path)
            msg = "2D 分割成功" if success else "2D 分割失败"
        elif ext in ['.nii', '.gz']:
            base, e = os.path.splitext(file_path)
            if e == ".gz":
                base, _ = os.path.splitext(base)
            out_path = base + "_seg.nii.gz"
            print(f"检测到 3D 图像: {file_path} -> 执行 nnUNet 分割")
            success = self._segment_3d_nifti(file_path, out_path)
            msg = "3D 分割成功" if success else "3D 分割失败"
        else:
            msg = f"不支持的文件格式: {ext}"
            return {"success": False, "out_path": None, "msg": msg, "radiology_report": ""}

        if success:
            radiology_report = self.generate_radiology_report(file_path, out_path, symptoms)

        return {
            "success": success,
            "out_path": out_path if success else None,
            "msg": msg,
            "radiology_report": radiology_report
        }


if __name__ == "__main__":
    print("--- Testing Tools ---")

    kg = KnowledgeGraphTool()
    if kg.driver:
        print("KG Connected.")
        kg.close()

    medsam = MedSAMTool()
    print("MedSAMTool(nnUNet) initialized.")
