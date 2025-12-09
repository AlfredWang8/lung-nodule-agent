# Lung Nodule Multi-Agent System

这是一个基于 LangGraph 和 MedSAM 的肺结节多智能体诊疗系统。该系统模拟了呼吸科、放射科、胸外科、病理科、肿瘤科和康复科的多学科会诊（MDT）流程。

## 功能特性

- **多智能体协作**：模拟 6 个不同角色的医生进行协同工作。
- **医学影像分析**：集成 MedSAM 模型，支持 2D (PNG/JPG) 和 3D (NII.GZ) 肺结节自动分割。
- **知识图谱支持**：集成 Neo4j 知识图谱，用于辅助呼吸科初筛和指南查询。
- **全流程模拟**：覆盖从初筛、影像检查、手术决策、病理确诊到术后康复的全过程。
- **自动报告生成**：最终生成包含各科室意见的综合诊疗报告。

## 目录结构

```
d:\Code\agent\
├── lung_nodule_multi_agent.py  # 主程序入口
├── tools.py                    # 工具类 (Neo4j, MedSAM)
├── medsam_tools/               # MedSAM 相关工具和模型代码
├── KG_tools/                   # 知识图谱构建工具
├── patient/                    # 患者数据存储
│   ├── info/                   # 患者基本信息 (CSV)
│   ├── pic/                    # 影像分割结果
│   └── report/                 # 诊疗报告
├── guidelines/                 # 医学指南文件
└── requirements.txt            # 项目依赖
```

## 安装与配置

1.  **克隆项目**
    ```bash
    git clone https://github.com/AlfredWang8/lung-nodule-agent.git
    cd lung-nodule-agent
    ```

2.  **安装依赖**
    建议使用 Python 3.9+ 环境。
    ```bash
    pip install -r requirements.txt
    ```

3.  **配置环境变量**
    在项目根目录创建 `.env` 文件，并填写以下配置：
    ```ini
    # DeepSeek API
    DEEPSEEK_API_KEY=your_api_key_here

    # Neo4j Database
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    ```

4.  **准备模型文件**
    确保 `medsam_tools/MedSAM/medsam_vit_b.pth` 模型权重文件存在。如果不存在，请从 MedSAM 官方仓库下载并放置在该路径下。

## 运行系统

运行主程序启动多智能体模拟：

```bash
python lung_nodule_multi_agent.py
```

系统将自动：
1. 生成唯一的患者 ID。
2. 依次进入各科室节点。
3. 在终端与您进行交互（如呼吸科问诊、放射科影像确认等）。
4. 最终在 `patient/report` 目录下生成综合诊疗报告。

## 注意事项

- 确保 Neo4j 数据库已启动且配置正确。
- 确保已安装 PyTorch 且版本与您的硬件（CPU/GPU）匹配。
