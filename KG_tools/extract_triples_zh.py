import os
import json
import re
import csv
import fitz  # PyMuPDF
import requests
from typing import Dict, List, Tuple, Optional

def read_env(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                if "=" in s:
                    k, v = s.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), v)

def load_pdf_text(pdf_path: str, max_pages: int = 0) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    texts = []
    for i, page in enumerate(doc):
        if max_pages > 0 and i >= max_pages:
            break
        texts.append(page.get_text("text"))
    return "\n\n".join(texts)

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    chunks = []
    i, n = 0, len(text)
    stride = max(1, max_chars - overlap)
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        if j == n:
            break
        i += stride
    return chunks

# Chinese Prompt
PROMPT_TEMPLATE = """你是一个专业的医学信息提取系统。
任务：从提供的医学指南文本中提取结构化的“主体-谓语-客体”三元组。
目标：肺癌诊断、治疗指南、药物适应症和基因突变。

输出格式：
仅返回一个包含“triples”键的有效JSON对象，其中包含对象列表。
Schema:
{{
  "triples": [
    {{
      "subject": "确切的实体名称（中文）",
      "predicate": "关系类型（中文，例如：适用于、治疗、导致、相关于、属于、包含、剂量为、禁忌症）",
      "object": "确切的实体名称（中文）",
      "evidence": "支持此提取的文本简短引用",
      "confidence": 0.95,
      "section": "章节标题（如果已知）"
    }}
  ]
}}

约束：
- 尽可能将提取的实体和关系翻译为中文，或者保留通用的英文医学缩写（如EGFR, ALK）。
- 确保输出是合法的JSON。

待处理文本：
<<<
{chunk}
>>>
"""

def parse_json_response(txt: str) -> Dict:
    clean_txt = txt.strip()
    if clean_txt.startswith("```"):
        lines = clean_txt.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].strip() == "```": lines = lines[:-1]
            clean_txt = "\n".join(lines)
    
    try:
        return json.loads(clean_txt)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", clean_txt)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass
        return {"triples": []}

def normalize_triple(t: Dict) -> Tuple[str, str, str]:
    s = (t.get("subject") or "").strip().lower()
    p = (t.get("predicate") or "").strip().lower()
    o = (t.get("object") or "").strip().lower()
    return s, p, o

def extract_triples_from_chunk(chunk: str, api_key: str) -> List[Dict]:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    prompt = PROMPT_TEMPLATE.format(chunk=chunk)
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant. Please output in Chinese."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 4000
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        return parsed.get("triples", [])
    except Exception as e:
        print(f"API Error: {e}")
        return []

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root, ".env")
    read_env(env_path)
    
    # Config
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in .env")
        return

    # Input/Output
    pdf_path = os.path.join(root, "guidelines", "lung_cancer_guideline.pdf")
    out_dir = os.path.join(root, "outputs_triples")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "lung_cancer_triples_zh.json")
    out_csv = os.path.join(out_dir, "lung_cancer_triples_zh.csv")
    
    # Limits for testing
    max_pages = int(os.environ.get("MAX_PAGES", "5"))
    max_chunks = int(os.environ.get("MAX_CHUNKS", "0")) 
    
    print(f"Processing {pdf_path}")
    print(f"Max pages: {max_pages}, Max chunks: {max_chunks}")
    
    text = load_pdf_text(pdf_path, max_pages)
    print(f"Loaded {len(text)} chars.")
    
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")
    
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
        print(f"Limiting to {len(chunks)} chunks for testing.")
        
    all_triples = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        ts = extract_triples_from_chunk(chunk, api_key)
        for t in ts:
            t["chunk_id"] = i
            all_triples.append(t)
            
    print(f"Extracted {len(all_triples)} raw triples.")
    
    # Deduplicate
    unique_triples = []
    seen = set()
    for t in all_triples:
        key = normalize_triple(t)
        if all(key) and key not in seen:
            seen.add(key)
            unique_triples.append(t)
            
    print(f"Unique triples: {len(unique_triples)}")
    
    # Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"triples": unique_triples}, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_json}")
    
    # Save CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "predicate", "object", "evidence", "confidence", "section"])
        for t in unique_triples:
            w.writerow([
                t.get("subject"), t.get("predicate"), t.get("object"),
                t.get("evidence"), t.get("confidence"), t.get("section")
            ])
    print(f"Saved to {out_csv}")

if __name__ == "__main__":
    main()
