import os
import sys
import torch
from typing import List, Dict, Optional, Tuple, Union
from neo4j import GraphDatabase

# Try to import MedSAM tools
try:
    # Add current directory to path to find medsam_tools if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from medsam_tools.medsam_client import segment_3d_volume, segment_2d_image
    MEDSAM_AVAILABLE = True
except ImportError:
    MEDSAM_AVAILABLE = False
    print("警告：未找到 medsam_tools 模块。MedSAM 功能将被禁用。")

class KnowledgeGraphTool:
    """
    知识图谱工具：封装 Neo4j 数据库的连接与查询功能。
    用于获取医学指南、疾病关联等知识。
    """
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        # 优先使用传入参数，否则尝试读取环境变量
        self.uri = uri or os.environ.get("NEO4J_URI")
        self.user = user or os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
        self.password = password or os.environ.get("NEO4J_PASSWORD")
        
        self.driver = None
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
        """执行 Cypher 查询并返回字典列表"""
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
    图像分割工具：封装 MedSAM 模型。
    支持 2D (png/jpg/etc) 和 3D (nii/nii.gz) 医学图像的分割。
    """
    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path:
            # 默认尝试寻找路径
            base_path = os.path.dirname(os.path.abspath(__file__))
            default_ckpt = os.path.join(base_path, "medsam_tools", "MedSAM", "medsam_vit_b.pth")
            if os.path.exists(default_ckpt):
                self.checkpoint_path = default_ckpt
            else:
                print(f"警告：未指定 checkpoint 且默认路径 {default_ckpt} 不存在。")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if MEDSAM_AVAILABLE:
            print(f"MedSAM 工具已初始化 (设备: {self.device})")

    def segment_3d(self, ct_path: str, out_path: str, slice_z: int, bbox: List[int], overlay_path: str = None) -> bool:
        if not MEDSAM_AVAILABLE:
            print("未安装 MedSAM 工具库。")
            return False
        try:
            bbox_str = ",".join(map(str, bbox))
            segment_3d_volume(
                ct_path=ct_path,
                out_path=out_path,
                checkpoint=self.checkpoint_path,
                device=self.device,
                slice_z=slice_z,
                bbox=bbox_str,
                overlay_out_path=overlay_path
            )
            return True
        except Exception as e:
            print(f"3D 分割失败: {e}")
            return False

    def segment_2d(self, img_path: str, out_path: str, bbox: List[int], overlay_path: str = None) -> bool:
        if not MEDSAM_AVAILABLE:
            print("未安装 MedSAM 工具库。")
            return False
        try:
            bbox_str = ",".join(map(str, bbox))
            segment_2d_image(
                img_path=img_path,
                out_path=out_path,
                checkpoint=self.checkpoint_path,
                device=self.device,
                bbox=bbox_str,
                save_overlay=True # 默认开启叠加图保存
            )
            return True
        except Exception as e:
            print(f"2D 分割失败: {e}")
            return False

    def detect_and_segment(self, file_path: str, bbox: List[int], slice_z: int = None) -> Dict[str, Union[bool, str]]:
        """
        自动检测文件类型并执行分割。
        返回结果字典: {'success': bool, 'out_path': str, 'msg': str}
        """
        if not os.path.exists(file_path):
            return {"success": False, "out_path": None, "msg": f"文件不存在: {file_path}"}

        ext = os.path.splitext(file_path)[1].lower()
        out_path = ""
        success = False
        msg = ""

        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            out_path = os.path.splitext(file_path)[0] + "_seg" + ext
            print(f"检测到 2D 图像: {file_path} -> 执行 2D 分割")
            success = self.segment_2d(file_path, out_path, bbox)
            msg = "2D 分割成功" if success else "2D 分割失败"

        elif ext in ['.gz', '.nii']:
            if file_path.endswith(".nii.gz"):
                out_path = file_path.replace(".nii.gz", "_seg.nii.gz")
            else:
                out_path = os.path.splitext(file_path)[0] + "_seg.nii"
            
            print(f"检测到 3D 卷数据: {file_path} -> 执行 3D 分割")
            if slice_z is None:
                msg = "3D 分割需要提供 slice_z 参数"
                success = False
            else:
                success = self.segment_3d(file_path, out_path, slice_z, bbox)
                msg = "3D 分割成功" if success else "3D 分割失败"
        else:
            msg = f"不支持的文件格式: {ext}"
            return {"success": False, "out_path": None, "msg": msg}

        return {
            "success": success,
            "out_path": out_path if success else None,
            "msg": msg
        }

if __name__ == "__main__":
    # 简单的测试代码
    print("--- Testing Tools ---")
    
    # 1. Test KG Tool (Assuming env vars are set or will fail gracefully)
    kg = KnowledgeGraphTool()
    if kg.driver:
        print("KG Connected.")
        kg.close()
    
    # 2. Test MedSAM Tool init
    medsam = MedSAMTool()
    print(f"MedSAM Checkpoint: {medsam.checkpoint_path}")
