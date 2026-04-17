import os
import io
import shutil
from typing import Dict, Any, List
import builtins

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from skimage import io as skio

import streamlit as st
from lung_nodule_multi_agent import generate_patient_id, MedicalAgentSystem


load_dotenv()


def _apply_base_style():
    """设置整体页面的基础配色和样式"""
    st.set_page_config(
        page_title="肺结节多智能体诊断系统",
        layout="wide"
    )
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f8ff;
        }
        h1, h2, h3 {
            color: #003366;
        }
        .deepblue-card {
            border-radius: 8px;
            padding: 16px 20px;
            background-color: #ffffff;
            border: 1px solid #d0ddf5;
        }
        .treatment-card {
            border-left: 6px solid #003f8c;
            padding: 16px 20px;
            background-color: #ffffff;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        }
        .risk-high {
            color: #d0021b;
            font-weight: 700;
        }
        .risk-medium {
            color: #f5a623;
            font-weight: 700;
        }
        .risk-low {
            color: #2e7d32;
            font-weight: 700;
        }
        .stButton>button {
            background-color: #003f8c;
            color: #ffffff;
            border-radius: 4px;
            border: 1px solid #003f8c;
        }
        .stButton>button:hover {
            background-color: #0050b3;
            border-color: #0050b3;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def _save_upload_to_disk(upload, patient_id: str) -> Dict[str, str]:
    """将上传文件保存到本地并处理 DICOM -> PNG 转换"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(base_dir, "patient", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    filename = upload.name
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    raw_path = os.path.join(upload_dir, f"{patient_id}_raw{ext}")
    with open(raw_path, "wb") as f:
        f.write(upload.getbuffer())

    if ext in [".dcm", ".dicom"]:
        try:
            import SimpleITK as sitk

            img_sitk = sitk.ReadImage(raw_path)
            arr = sitk.GetArrayFromImage(img_sitk)

            if arr.ndim == 3:
                z_mid = arr.shape[0] // 2
                arr2d = arr[z_mid]
            else:
                arr2d = arr

            arr2d = arr2d.astype(np.float32)
            arr2d = (arr2d - arr2d.min()) / (arr2d.max() - arr2d.min() + 1e-8)
            arr2d = (arr2d * 255).astype(np.uint8)

            display_path = os.path.join(upload_dir, f"{patient_id}_input.png")
            skio.imsave(display_path, arr2d, check_contrast=False)

            return {
                "raw_path": raw_path,
                "display_path": display_path
            }
        except Exception as e:
            st.error(f"DICOM 图像读取或转换失败：{e}")
            raise
    else:
        fname_lower = filename.lower()
        if fname_lower.endswith(".nii.gz"):
            display_path = os.path.join(upload_dir, f"{patient_id}_input.nii.gz")
        elif ext == ".nii":
            display_path = os.path.join(upload_dir, f"{patient_id}_input.nii")
        else:
            display_path = os.path.join(upload_dir, f"{patient_id}_input{ext}")

        with open(display_path, "wb") as f:
            f.write(upload.getbuffer())

        return {
            "raw_path": raw_path,
            "display_path": display_path
        }


def _derive_overlay_path(seg_path: str) -> str:
    """根据 MedSAM 输出路径推断叠加图路径"""
    overlay_out = seg_path.replace(".png", "_overlay.png").replace(".jpg", "_overlay.jpg")
    if overlay_out == seg_path:
        overlay_out = overlay_out + "_overlay.png"
    return overlay_out


def _build_risk_text(patient_info: Dict[str, Any]) -> str:
    """根据表单信息构建危险因素描述文本"""
    smoking = "有吸烟史" if patient_info.get("smoking_history") else "无明确吸烟史"
    cancer = "有既往恶性肿瘤史" if patient_info.get("cancer_history") else "无明确既往恶性肿瘤史"
    age = patient_info.get("age")
    gender = patient_info.get("gender") or "未填写"
    name = patient_info.get("name") or "未填写"
    symptoms = patient_info.get("symptoms") or "未补充典型呼吸系统症状"

    return (
        f"患者：{name}，性别：{gender}，年龄：{age} 岁；"
        f"{smoking}，{cancer}；主要症状/备注：{symptoms}"
    )


def _compute_risk_level(radiology_report: str, pathology_text: str) -> str:
    """基于影像与病理文本粗略估计风险分层"""
    text = (radiology_report or "") + "\n" + (pathology_text or "")
    if any(k in text for k in ["浸润性", "腺癌", "鳞癌", "高度怀疑恶性", "恶性肿瘤"]):
        return "high"
    if any(k in text for k in ["可疑恶性", "不除外恶性"]):
        return "medium"
    return "low"


def diagnose(
    image_path: str,
    patient_info: Dict[str, Any],
    respiratory_placeholder=None,
    radiology_placeholder=None,
    pathology_placeholder=None,
    surgery_placeholder=None,
    oncology_placeholder=None,
    rehab_placeholder=None,
) -> Dict[str, Any]:
    name = patient_info.get("name") or "未提供"
    age = patient_info.get("age") or "未填写"
    gender = patient_info.get("gender") or "未填写"
    smoking_text = "是" if patient_info.get("smoking_history") else "否"
    cancer_text = "是" if patient_info.get("cancer_history") else "否"
    patient_id = patient_info.get("patient_id") or "UNKNOWN"
    symptoms_text = patient_info.get("symptoms") or ""

    history_text = f"吸烟史：{smoking_text}；既往肿瘤史：{cancer_text}"

    system = MedicalAgentSystem()

    init_state: Dict[str, Any] = {
        "patient_id": patient_id,
        "name": name,
        "age": str(age),
        "gender": gender,
        "symptoms": symptoms_text,
        "ct_image_path": image_path,
        "ct_date": "",
        "nodule_bbox": None,
        "slice_z": 50,
        "pathology_result": "",
        "respiratory_report": None,
        "radiology_report": None,
        "segmentation_path": None,
        "pathology_report": None,
        "surgical_plan": None,
        "oncology_plan": None,
        "rehab_plan": None,
        "history": [],
        "current_step": "start",
    }

    original_input = builtins.input

    def input_mock(prompt: str = "") -> str:
        if "患者 (请输入):" in prompt and "症状" in prompt:
            return symptoms_text
        if "患者 (请输入):" in prompt and "吸烟史或家族病史" in prompt:
            return history_text
        if "请输入路径" in prompt:
            return ""
        if "检查日期" in prompt:
            return ""
        return ""

    try:
        builtins.input = input_mock
        state: Dict[str, Any] = dict(init_state)

        step_result = system.role_respiratory_physician(state)
        state.update(step_result)
        if respiratory_placeholder is not None:
            respiratory_placeholder.markdown(
                "### 呼吸科初筛报告\n\n" + (state.get("respiratory_report") or "")
            )

        step_result = system.role_radiologist(state)
        state.update(step_result)
        if radiology_placeholder is not None:
            radiology_placeholder.markdown(
                "### 放射科影像报告\n\n" + (state.get("radiology_report") or "")
            )

        step_result = system.role_thoracic_surgeon(state)
        state.update(step_result)
        if surgery_placeholder is not None:
            surgery_placeholder.markdown(
                "### 胸外科治疗方案\n\n" + (state.get("surgical_plan") or "")
            )

        step_result = system.role_pathologist(state)
        state.update(step_result)
        if pathology_placeholder is not None:
            pathology_placeholder.markdown(
                "### 病理科报告\n\n" + (state.get("pathology_report") or "")
            )

        step_result = system.role_radiation_oncologist(state)
        state.update(step_result)
        if oncology_placeholder is not None:
            oncology_placeholder.markdown(
                "### 肿瘤科治疗规划\n\n" + (state.get("oncology_plan") or "")
            )

        step_result = system.role_rehabilitation_physician(state)
        state.update(step_result)
        if rehab_placeholder is not None:
            rehab_placeholder.markdown(
                "### 康复科康复指导\n\n" + (state.get("rehab_plan") or "")
            )

        final_state = state
    finally:
        builtins.input = original_input

    seg_path = final_state.get("segmentation_path")
    radiology_report = final_state.get("radiology_report", "")
    respiratory_report = final_state.get("respiratory_report", "")
    surgical_plan = final_state.get("surgical_plan", "")
    pathology_report = final_state.get("pathology_report", "")
    oncology_plan = final_state.get("oncology_plan", "")
    rehab_plan = final_state.get("rehab_plan", "")

    return {
        "patient_id": patient_id,
        "image_path": image_path,
        "segmentation_path": seg_path,
        "respiratory_report": respiratory_report,
        "radiology_report": radiology_report,
        "pathology_report": pathology_report,
        "surgical_plan": surgical_plan,
        "oncology_plan": oncology_plan,
        "rehab_plan": rehab_plan,
    }


def main():
    _apply_base_style()

    st.title("肺结节多智能体诊断系统")

    with st.form("patient_form"):
        uploaded = st.file_uploader(
            "上传胸部 CT 图像",
            type=["dcm", "dicom", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "nii", "gz"],
            accept_multiple_files=False
        )

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("姓名（可选）", "")
            age = st.number_input("年龄", min_value=0, max_value=120, step=1, value=60)
            gender = st.selectbox("性别", ["男", "女", "其他"])
        with col2:
            symptoms = st.text_area("主要症状 / 既往病史 / 备注", "")

        submitted = st.form_submit_button("开始诊断")

    if submitted:
        if not uploaded:
            st.warning("请先上传 CT 图像文件。")
            return

        try:
            patient_id = generate_patient_id()
            saved_paths = _save_upload_to_disk(uploaded, patient_id)
            image_path = saved_paths["display_path"]

            patient_info = {
                "patient_id": patient_id,
                "name": name.strip(),
                "age": int(age),
                "gender": gender,
                "smoking_history": False,
                "cancer_history": False,
                "symptoms": symptoms.strip(),
            }

            seg_section = st.empty()
            resp_section = st.empty()
            rad_section = st.empty()
            surg_section = st.empty()
            path_section = st.empty()
            onco_section = st.empty()
            rehab_section = st.empty()

            # seg_section.markdown(f"### 图像与分割结果\n\n- 原始图像路径: {image_path}")

            with st.spinner("正在进行图像分割和多学科会诊，请稍候..."):
                result = diagnose(
                    image_path,
                    patient_info,
                    respiratory_placeholder=resp_section,
                    radiology_placeholder=rad_section,
                    pathology_placeholder=path_section,
                    surgery_placeholder=surg_section,
                    oncology_placeholder=onco_section,
                    rehab_placeholder=rehab_section,
                )

            seg_path = result.get("segmentation_path")

            base_dir = os.path.dirname(os.path.abspath(__file__))
            report_dir = os.path.join(base_dir, "patient", "report")
            os.makedirs(report_dir, exist_ok=True)
            report_file = os.path.join(report_dir, f"{result['patient_id']}_report.md")

            with open(report_file, "w", encoding="utf-8") as f:
                f.write("# 肺结节多学科会诊(MDT)综合报告\n\n")
                f.write(f"- 患者ID: {result['patient_id']}\n")
                f.write(f"- 姓名: {patient_info.get('name') or '未记录'}\n")
                f.write(f"- 性别: {patient_info.get('gender') or '未记录'}\n")
                f.write(f"- 年龄: {patient_info.get('age')}\n")
                f.write(f"- 主诉症状: {patient_info.get('symptoms') or ''}\n\n")

                if seg_path:
                    ext = os.path.splitext(seg_path)[1].lower()
                    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                        rel_seg = os.path.relpath(seg_path, os.path.dirname(report_file))
                        f.write(f"![AI分割结果]({rel_seg})\n\n")
                    else:
                        f.write(f"> AI 分割结果路径: {seg_path}\n\n")
                else:
                    f.write("> AI 分割结果：未生成\n\n")

                f.write("## 1. 呼吸科初筛报告\n\n")
                f.write((result.get('respiratory_report') or "") + "\n\n")

                f.write("## 2. 放射科影像报告\n\n")
                f.write((result.get('radiology_report') or "") + "\n\n")

                f.write("## 3. 胸外科手术/治疗建议\n\n")
                f.write((result.get('surgical_plan') or "") + "\n\n")

                f.write("## 4. 病理科确诊报告\n\n")
                f.write((result.get('pathology_report') or "") + "\n\n")

                f.write("## 5. 肿瘤科治疗规划\n\n")
                f.write((result.get('oncology_plan') or "") + "\n\n")

                f.write("## 6. 康复科康复指导\n\n")
                f.write((result.get('rehab_plan') or "") + "\n")

            if seg_path:
                seg_section.markdown(
                    f"### 图像与分割结果\n\n- 分割结果文件: {seg_path}\n- 报告文件: {report_file}"
                )
            else:
                seg_section.markdown(
                    f"### 图像与分割结果\n\n- 分割结果文件: 未生成\n- 报告文件: {report_file}"
                )

            st.success("诊断完成")
        except Exception as e:
            st.error(f"诊断流程运行失败：{e}")


if __name__ == "__main__":
    main()

