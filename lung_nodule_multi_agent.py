import os
import csv
from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from tools import KnowledgeGraphTool, MedSAMTool

# --- 1. Agent Configuration ---
def get_llm():
    """配置 DeepSeek LLM"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 文件中配置 DEEPSEEK_API_KEY")
    
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.1
    )

def save_patient_info(patient_info: Dict):
    """保存患者信息到 CSV"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient", "info", "patients.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    headers = ["patient_id", "name", "age", "gender", "symptoms", "ct_date", "pathology_result", "created_at"]
    file_exists = os.path.exists(file_path)
    
    import datetime
    row = {
        "patient_id": patient_info.get("patient_id"),
        "name": patient_info.get("name"),
        "age": patient_info.get("age"),
        "gender": patient_info.get("gender"),
        "symptoms": patient_info.get("symptoms"),
        "ct_date": patient_info.get("ct_date", ""),
        "pathology_result": patient_info.get("pathology_result", ""),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[系统] 患者信息已存档至: {file_path}")

def generate_patient_id() -> str:
    """根据现有记录生成唯一的患者ID (P-YYYY-NNN)"""
    import datetime
    current_year = datetime.datetime.now().year
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "patient", "info", "patients.csv")
    
    if not os.path.exists(csv_path):
        return f"P-{current_year}-001"
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # 筛选当年的ID
            ids = [row["patient_id"] for row in reader if row["patient_id"] and row["patient_id"].startswith(f"P-{current_year}")]
            
        if not ids:
             return f"P-{current_year}-001"
        
        # 找到最大序列号
        max_seq = 0
        for pid in ids:
            try:
                parts = pid.split("-")
                if len(parts) >= 3:
                    seq = int(parts[-1])
                    if seq > max_seq:
                        max_seq = seq
            except ValueError:
                continue
        
        return f"P-{current_year}-{max_seq + 1:03d}"
    except Exception as e:
        print(f"生成ID时出错: {e}, 使用时间戳作为后备")
        return f"P-{current_year}-{datetime.datetime.now().strftime('%m%d%H%M')}"

# --- 2. State Definition ---
class MedicalRecord(TypedDict):
    """电子病历状态"""
    patient_id: str
    name: str
    age: str
    gender: str
    symptoms: str
    
    ct_image_path: str
    ct_date: str # CT检查日期
    nodule_bbox: Optional[List[int]] # [x1, y1, x2, y2]
    slice_z: Optional[int]
    
    pathology_result: str # 病理结果输入
    
    # 各科室报告
    respiratory_report: Optional[str]
    radiology_report: Optional[str]
    segmentation_path: Optional[str]
    pathology_report: Optional[str]
    surgical_plan: Optional[str]
    oncology_plan: Optional[str]
    rehab_plan: Optional[str]
    
    # 决策历史
    history: List[str]
    current_step: str

# --- 3. Agent Nodes ---

class MedicalAgentSystem:
    def __init__(self):
        self.llm = get_llm()
        self.kg_tool = KnowledgeGraphTool() # 自动连接 Neo4j
        self.medsam_tool = MedSAMTool()     # 自动加载 MedSAM
        
        # 构建图
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MedicalRecord)

        # 添加节点 (6类角色)
        workflow.add_node("respiratory_dept", self.role_respiratory_physician)
        workflow.add_node("radiology_dept", self.role_radiologist)
        workflow.add_node("thoracic_surgery", self.role_thoracic_surgeon)
        workflow.add_node("pathology_dept", self.role_pathologist)
        workflow.add_node("oncology_dept", self.role_radiation_oncologist)
        workflow.add_node("rehab_dept", self.role_rehabilitation_physician)

        # 定义流程 (临床路径)
        # 1. 呼吸科初筛 -> 放射科拍片
        workflow.set_entry_point("respiratory_dept")
        workflow.add_edge("respiratory_dept", "radiology_dept")
        
        # 2. 放射科出报告 -> 胸外科会诊
        workflow.add_edge("radiology_dept", "thoracic_surgery")
        
        # 3. 胸外科判断是否需要病理 (简化逻辑：默认需要确诊 -> 病理科)
        workflow.add_edge("thoracic_surgery", "pathology_dept")
        
        # 4. 病理科确诊 -> 肿瘤科/外科/康复科 (这里做一个条件分支示例)
        # 为简化演示，我们设计一个线性流：病理 -> 肿瘤科制定放疗计划 -> 康复科
        workflow.add_edge("pathology_dept", "oncology_dept")
        workflow.add_edge("oncology_dept", "rehab_dept")
        workflow.add_edge("rehab_dept", END)

        return workflow.compile()

    # --- 角色实现 ---

    def role_respiratory_physician(self, state: MedicalRecord) -> Dict:
        """呼吸科医生：初筛与问诊 (交互式)"""
        print(f"\n[呼吸科] 正在接诊...")
        
        # 交互式问诊逻辑 - 收集基本信息
        print("医生: 您好，我是呼吸科医生。在开始之前，请登记一下您的基本信息。")
        
        # 如果 state 中没有信息，则询问
        name = state.get('name') or input("医生: 请问您的姓名是？\n患者: ")
        age = state.get('age') or input("医生: 您的年龄是？\n患者: ")
        gender = state.get('gender') or input("医生: 您的性别是？\n患者: ")
        
        # 更新状态
        state['name'] = name
        state['age'] = age
        state['gender'] = gender
        
        print(f"医生: 好的，{name}。请详细描述您的症状（如咳嗽时长、是否有痰/血、胸痛情况等）：")
        user_symptoms = input("患者 (请输入): ")
        if not user_symptoms:
            user_symptoms = state['symptoms'] # Fallback
        else:
            state['symptoms'] = user_symptoms
            
        print("医生: 好的，我了解了。请问您有吸烟史或家族病史吗？")
        history_input = input("患者 (请输入): ")
        full_symptoms = f"{state['symptoms']}; 病史补充: {history_input}"
        
        # 保存患者信息到表格 (部分信息，后续更新)
        # 此时还没有 CT 和 病理信息
        pass
        
        # 查询指南
        guidelines = self.kg_tool.query(
            "MATCH (n:Disease)-[:HAS_SYMPTOM]->(s) WHERE s.name CONTAINS '咳嗽' OR s.name CONTAINS '咯血' RETURN n.name, s.name LIMIT 3"
        )
        guideline_text = str(guidelines) if guidelines else "未找到具体指南，按常规流程处理。"

        prompt = f"""
        你是呼吸科医生。请输出初筛结论，要求：
        - 第一行：一句话总结当前判断与建议（是否需影像学检查与紧急程度）
        - 后续5-8条要点：症状关键点、危险因素（吸烟/家族史）、初步风险分层、体检与实验室建议、影像学检查类型与时机、随访建议
        背景：患者信息 {name}, {gender}, {age}岁；症状 {full_symptoms}；指南参考 {guideline_text}
        直接输出内容，不要包含额外提示语。
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": full_symptoms,
            "respiratory_report": response.content,
            "history": state["history"] + ["呼吸科已完成初筛"],
            "current_step": "radiology"
        }

    def role_radiologist(self, state: MedicalRecord) -> Dict:
        """放射科医生：影像检查与分割 (交互式)"""
        print(f"\n[放射科] 正在接诊...")
        
        # 交互式询问 CT 信息
        print("放射科医生: 请提供胸部CT影像文件的路径：")
        ct_path_input = input("患者/助手 (请输入路径，回车使用默认测试图): ")
        if not ct_path_input:
            ct_path_input = state['ct_image_path'] # Default
        
        print("放射科医生: 请提供检查日期 (YYYY-MM-DD)：")
        ct_date_input = input("患者/助手 (请输入): ")
        if not ct_date_input:
            import datetime
            ct_date_input = datetime.datetime.now().strftime("%Y-%m-%d")
            
        state['ct_image_path'] = ct_path_input
        state['ct_date'] = ct_date_input
        
        print(f"[放射科] 正在分析影像 {ct_path_input}...")
        
        # 调用 MedSAM 工具进行分割
        seg_path = None
        print("  -> 执行 AI 辅助分割...")
        bbox_used = state.get("nodule_bbox")
        if not bbox_used:
            bbox_used = self._suggest_bbox_with_llm(ct_path_input, state.get("symptoms", ""))
            if bbox_used:
                print(f"  -> 使用 DeepSeek 建议的目标框: {bbox_used}")
            else:
                print("  -> 无有效建议，使用自动目标框")
        res = self.medsam_tool.detect_and_segment(
            state['ct_image_path'], 
            bbox_used, 
            state.get('slice_z')
        )
        if not res['success'] and bbox_used is not None:
            print("  -> 建议目标框分割失败，回退为自动目标框")
            res = self.medsam_tool.detect_and_segment(
                state['ct_image_path'],
                None,
                state.get('slice_z')
            )
        if res['success']:
            original_out = res['out_path']
            ext = os.path.splitext(original_out)[1].lower()
            if ext == ".gz" and original_out.endswith(".nii.gz"):
                new_filename = f"{state['patient_id']}_seg.nii.gz"
            else:
                new_filename = f"{state['patient_id']}_seg{ext}"
            target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient", "pic")
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, new_filename)
            import shutil
            try:
                shutil.move(original_out, new_path)
                seg_path = new_path
                print(f"  -> 分割结果已归档至: {seg_path}")
            except Exception as e:
                print(f"  -> 归档失败: {e}, 保留原路径")
                seg_path = original_out
        
        prompt = f"""
        你是放射科医生。请输出详细影像学诊断：
        - 第一行：一句话总结（总体结论：结节数量、最大结节位置与性质倾向）
        - 要点需覆盖：结节数量；逐个结节的大小（长径×短径，单位mm）；形状（圆形/卵圆/不规则）；边缘（毛刺/分叶/光滑）；密度/衰减（纯磨玻璃/部分实性/实性）；钙化与空洞情况；分布（肺叶/段、是否外周/中央）；与胸膜/血管/支气管关系；测量方法与层面索引；AI分割结果与路径；进一步检查与随访建议
        背景：检查日期 {ct_date_input}；症状 {state['symptoms']}；AI分割：{'成功: ' + str(seg_path) if seg_path else '未执行/失败'}
        直接输出内容。
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "ct_image_path": ct_path_input,
            "ct_date": ct_date_input,
            "radiology_report": response.content,
            "segmentation_path": seg_path,
            "history": state["history"] + ["放射科已出具报告"],
            "current_step": "surgery"
        }

    def _suggest_bbox_with_llm(self, img_path: str, context_text: str = "") -> Optional[List[int]]:
        try:
            from skimage import io
            img = io.imread(img_path)
            if img.ndim == 2:
                h, w = img.shape
            else:
                h, w = img.shape[0], img.shape[1]
            prompt = f"""
            你是放射科分割助手。根据以下临床上下文，选择结节大概率所在区域：
            {context_text}
            仅从以下选项中选择并输出一个词：center, right_upper, right_lower, left_upper, left_lower, perihilar
            只输出该词，不要输出其他内容。
            """
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            region = resp.content.strip().lower()
            if region == "center" or region == "perihilar":
                x1 = int(0.25 * w); y1 = int(0.25 * h); x2 = int(0.75 * w); y2 = int(0.75 * h)
            elif region == "right_upper":
                x1 = int(0.55 * w); y1 = int(0.05 * h); x2 = int(0.90 * w); y2 = int(0.45 * h)
            elif region == "right_lower":
                x1 = int(0.55 * w); y1 = int(0.55 * h); x2 = int(0.90 * w); y2 = int(0.95 * h)
            elif region == "left_upper":
                x1 = int(0.10 * w); y1 = int(0.05 * h); x2 = int(0.45 * w); y2 = int(0.45 * h)
            elif region == "left_lower":
                x1 = int(0.10 * w); y1 = int(0.55 * h); x2 = int(0.45 * w); y2 = int(0.95 * h)
            else:
                return None
            x1 = max(0, min(w - 2, x1))
            y1 = max(0, min(h - 2, y1))
            x2 = max(x1 + 1, min(w - 1, x2))
            y2 = max(y1 + 1, min(h - 1, y2))
            return [x1, y1, x2, y2]
        except Exception:
            return None

    def role_thoracic_surgeon(self, state: MedicalRecord) -> Dict:
        """胸外科医生：临床决策"""
        print(f"\n[胸外科] 正在评估手术指征...")
        
        prompt = f"""
        你是胸外科医生。请输出处置决策：
        - 第一行：一句话总结（是否手术/活检及主要依据与时机）
        - 后续6-8条要点：手术适应证与禁忌证；术式选择（楔形/肺段/肺叶切除，开胸/胸腔镜/机器人）；淋巴结处理策略；围手术期评估（心肺功能/麻醉风险）；术前准备与影像再次评估；风险与并发症防范；若不手术的替代方案（活检/随访/消融）；多学科协作计划
        背景：呼吸科报告：{state['respiratory_report']}；放射科报告：{state['radiology_report']}
        直接输出内容。
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "surgical_plan": response.content,
            "history": state["history"] + ["胸外科已完成评估"],
            "current_step": "pathology"
        }

    def role_pathologist(self, state: MedicalRecord) -> Dict:
        """病理科医生：确诊 (交互式录入)"""
        print(f"\n[病理科] 正在进行病理分析...")
        
        # --- KG 检索辅助 ---
        # 尝试检索与肺部疾病相关的知识作为辅助
        print(f"  -> [系统] 正在调用知识图谱辅助诊断...")
        kg_query = "MATCH (n:Disease) WHERE n.name CONTAINS '肺' OR n.name CONTAINS '癌' RETURN n.name, n.description, n.treatment LIMIT 3"
        kg_data = self.kg_tool.query(kg_query)
        kg_context = str(kg_data) if kg_data else "暂无直接相关的知识库条目。"
        
        prompt = f"""
        你是病理科医生。请输出确诊意见：
        - 第一行：一句话核心结论（例如：结论：浸润性腺癌，T1N0M0，风险分层）
        - 后续6-8条要点：形态学依据（结构/核异型/腺体形成/黏液）；分化程度；必要的免疫组化标志物及解释（如TTF-1、Napsin A、p40等）；分期建议与取材部位说明；切缘与侵袭特征（胸膜/血管/神经）；与影像的整合判断与鉴别诊断；进一步检查或分子检测建议；随访与转诊建议
        背景：影像学：{state['radiology_report']}；症状：{state['symptoms']}；知识图谱：{kg_context}
        直接输出内容。
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        pathology_conclusion = self._extract_pathology_conclusion(response.content)
        state['pathology_result'] = pathology_conclusion
        
        save_patient_info({
            "patient_id": state['patient_id'],
            "name": state['name'],
            "age": state['age'],
            "gender": state['gender'],
            "symptoms": state['symptoms'],
            "ct_date": state.get('ct_date', ''),
            "pathology_result": pathology_conclusion
        })
        
        return {
            "pathology_result": pathology_conclusion,
            "pathology_report": response.content,
            "history": state["history"] + ["病理科已出具报告"],
            "current_step": "oncology"
        }

    def role_radiation_oncologist(self, state: MedicalRecord) -> Dict:
        """放射肿瘤科医生：治疗规划"""
        print(f"\n[放射肿瘤科] 正在制定放疗计划...")
        
        prompt = f"""
        你是放射肿瘤科医生。请输出极简治疗规划：
        - 第一行：一句话总结（首选治疗方案与目标）
        - 后续3-5条要点（剂量/疗程、联合治疗、随访计划、注意事项）
        背景：病理：{state['pathology_report']}；外科：{state['surgical_plan']}
        直接输出内容。
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "oncology_plan": response.content,
            "history": state["history"] + ["肿瘤科已制定计划"],
            "current_step": "rehab"
        }

    def role_rehabilitation_physician(self, state: MedicalRecord) -> Dict:
        """康复科医生：术后康复"""
        print(f"\n[康复科] 正在制定康复方案...")
        
        prompt = f"""
        你是康复科医生。请输出极简康复指导：
        - 第一行：一句话总结（康复重点与总体目标）
        - 后续3-5条要点（呼吸训练、运动方案、随访与预警、生活方式）
        直接输出内容。
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "rehab_plan": response.content,
            "history": state["history"] + ["康复科已制定计划"],
            "current_step": "finished"
        }
    
    def _extract_pathology_conclusion(self, report_text: str) -> str:
        try:
            lines = [l.strip() for l in report_text.splitlines() if l.strip()]
            if not lines:
                return "自动推断病理结论"
            first = lines[0]
            if "：" in first:
                parts = first.split("：", 1)
                return parts[1].strip() or first.strip()
            if ":" in first:
                parts = first.split(":", 1)
                return parts[1].strip() or first.strip()
            return first[:100]
        except Exception:
            return "自动推断病理结论"

    def run(self, inputs: MedicalRecord):
        return self.workflow.invoke(inputs)

if __name__ == "__main__":
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    # 初始化系统
    system = MedicalAgentSystem()
    
    # 模拟患者数据
    # 注意：这里需要一个真实的或测试用的图片路径
    test_img = os.path.abspath("test_nodule.png") 
    # 如果没有图片，创建一个假的用于测试流程
    if not os.path.exists(test_img):
        import numpy as np
        from skimage import io
        io.imsave(test_img, np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    # 自动生成唯一 ID
    new_pid = generate_patient_id()
    print(f"[系统] 分配新患者ID: {new_pid}")

    patient_data = {
        "patient_id": new_pid,
        "symptoms": "持续性干咳2周，偶有胸痛，无发热。",
        "ct_image_path": test_img,
        "nodule_bbox": [50, 50, 100, 100], # 模拟 bbox
        "slice_z": 0,
        "history": [],
        "current_step": "start"
    }
    
    print("--- 开始多智能体临床路径模拟 ---")
    final_state = system.run(patient_data)
    
    # 将最终报告写入文件
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient", "report")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"{final_state['patient_id']}_report.md")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# 肺结节多学科会诊(MDT)综合报告\n\n")
        f.write(f"- 患者ID: {final_state['patient_id']}\n")
        f.write(f"- 姓名: {final_state.get('name', '未记录')}\n")
        f.write(f"- 性别: {final_state.get('gender', '未记录')}\n")
        f.write(f"- 年龄: {final_state.get('age', '未记录')}\n")
        f.write(f"- 主诉症状: {final_state['symptoms']}\n\n")
        
        seg_path_md = final_state.get('segmentation_path', None)
        if seg_path_md:
            ext = os.path.splitext(seg_path_md)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                rel_seg = os.path.relpath(seg_path_md, os.path.dirname(report_file))
                f.write(f"![AI分割结果]({rel_seg})\n\n")
            else:
                f.write(f"> AI 分割结果路径: {seg_path_md}\n\n")
        else:
            f.write(f"> AI 分割结果：未生成\n\n")
        
        f.write(f"## 1. 呼吸科初筛报告\n\n")
        f.write(final_state['respiratory_report'] + "\n\n")
        
        f.write(f"## 2. 放射科影像报告\n\n")
        f.write(final_state['radiology_report'] + "\n\n")
        
        f.write(f"## 3. 胸外科手术/治疗建议\n\n")
        f.write(final_state['surgical_plan'] + "\n\n")
        
        f.write(f"## 4. 病理科确诊报告\n\n")
        f.write(final_state['pathology_report'] + "\n\n")
        
        f.write(f"## 5. 肿瘤科治疗规划\n\n")
        f.write(final_state['oncology_plan'] + "\n\n")
        
        f.write(f"## 6. 康复科康复指导\n\n")
        f.write(final_state['rehab_plan'] + "\n")
    
    print(f"\n✅ 完整诊疗报告已生成: {os.path.abspath(report_file)}")
    
    print("\n=== 最终诊疗汇总 ===")
    print(f"1. 呼吸科报告: {final_state['respiratory_report'][:50]}...")
    print(f"2. 放射科报告: {final_state['radiology_report'][:50]}...")
    print(f"3. 分割结果: {final_state['segmentation_path']}")
    print(f"4. 外科方案: {final_state['surgical_plan'][:50]}...")
    print(f"5. 病理报告: {final_state['pathology_report'][:50]}...")
    print(f"6. 肿瘤科计划: {final_state['oncology_plan'][:50]}...")
    print(f"7. 康复计划: {final_state['rehab_plan'][:50]}...")
    
    # 清理测试文件
    # if "test_nodule" in test_img:
    #     try:
    #         os.remove(test_img)
    #         seg = final_state.get("segmentation_path")
    #         if seg and os.path.exists(seg):
    #             base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient", "pic")
    #             seg_abs = os.path.abspath(seg)
    #             base_abs = os.path.abspath(base_dir)
    #             try:
    #                 common = os.path.commonpath([seg_abs, base_abs])
    #             except ValueError:
    #                 common = ""
    #             if common != base_abs:
    #                 os.remove(seg)
    #     except:
    #         pass
