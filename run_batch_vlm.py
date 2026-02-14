import os
import re
import json
import base64
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ====== 你需要改的 2 个地方 ======
MODEL_NAME = "doubao-seed-1-8-251228"   # 换模型
PROMPT_PATH = "prompt.txt"
# =================================


# 输入输出文件
GOLDSET_XLSX = "gold set 2.0.xlsx"
IMAGES_DIR = "images"
OUT_XLSX = "pred_results.xlsx"   # 输出的新表（带 model_pred_json）
OUT_JSONL = "pred_results.jsonl" # 每条一行 json，方便调试/追溯

# 火山 OpenAI 兼容 SDK
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("ARK_API_KEY"),
)

def read_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def find_image_file(sample_id: str, images_dir: str) -> str:
    # 支持 png/jpg/jpeg/webp
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for ext in exts:
        p = Path(images_dir) / f"{sample_id}{ext}"
        if p.exists():
            return str(p)
    return ""

def image_to_data_url(image_path: str) -> str:
    # 把本地图片转成 data url（最省事，不用先上传图床）
    suffix = Path(image_path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(text: str):
    """
    有些模型可能会多输出一些字符/换行，这里做一个稳健提取：
    - 优先尝试整体 json.loads
    - 不行就用正则截取第一个 {...} 的大括号块
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试找最外层 JSON
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def call_vlm(image_path: str, prompt: str) -> dict:
    data_url = image_to_data_url(image_path)

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    )

    # 兼容取文本（不同 SDK 版本字段可能略有差别）
    # 1) 尝试 resp.output_text
    text = getattr(resp, "output_text", None)
    if text is None:
        # 2) 尝试拼 outputs
        try:
            parts = []
            for item in resp.output:
                for c in item.content:
                    if c.type in ("output_text", "text"):
                        parts.append(c.text)
            text = "\n".join(parts).strip()
        except Exception:
            text = str(resp)

    parsed = extract_json(text)
    return {
        "raw_text": text,
        "parsed_json": parsed,
    }

def main():
    # 0) 读取 goldset
    df = pd.read_excel(GOLDSET_XLSX)

    # 必须有 sample_id
    if "sample_id" not in df.columns:
        raise ValueError("gold set 2.0.xlsx 里找不到 sample_id 列")

    # 1) 读取 prompt
    prompt = read_prompt(PROMPT_PATH)

    results = []
    jsonl_lines = []

    # 2) 逐行跑
    for idx, row in df.iterrows():
        sample_id = str(row["sample_id"]).strip()

        img_path = find_image_file(sample_id, IMAGES_DIR)
        if not img_path:
            print(f"[WARN] 找不到图片: {sample_id}（跳过）")
            results.append({
                "sample_id": sample_id,
                "image_path": None,
                "model_pred_json": None,
                "schema_ok": 0,
                "error": "IMAGE_NOT_FOUND",
            })
            continue

        print(f"[RUN] {sample_id} -> {img_path}")

        try:
            out = call_vlm(img_path, prompt)
            parsed = out["parsed_json"]

            schema_ok = 1 if isinstance(parsed, dict) else 0

            # 这里把 parsed_json 原样 dumps 成字符串，回填到 excel
            model_pred_json = json.dumps(parsed, ensure_ascii=False) if schema_ok else None

            results.append({
                "sample_id": sample_id,
                "image_path": img_path,
                "model_pred_json": model_pred_json,
                "schema_ok": schema_ok,
                "error": "" if schema_ok else "JSON_PARSE_FAIL",
            })

            jsonl_lines.append(json.dumps({
                "sample_id": sample_id,
                "image_path": img_path,
                "raw_text": out["raw_text"],
                "parsed_json": parsed,
            }, ensure_ascii=False))

        except Exception as e:
            results.append({
                "sample_id": sample_id,
                "image_path": img_path,
                "model_pred_json": None,
                "schema_ok": 0,
                "error": f"EXCEPTION: {repr(e)}",
            })
            print(f"[ERR] {sample_id}: {e}")

    # 3) 合并回原表（按 sample_id 左连接）
    pred_df = pd.DataFrame(results)
    out_df = df.merge(pred_df[["sample_id", "model_pred_json", "schema_ok", "error"]], on="sample_id", how="left")

    # 4) 保存
    out_df.to_excel(OUT_XLSX, index=False)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines))

    print(f"✅ Done. Saved: {OUT_XLSX}")
    print(f"✅ Debug JSONL: {OUT_JSONL}")

if __name__ == "__main__":
    main()