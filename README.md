# E-commerce Order Screenshot VLM Eval Demo

一个轻量级的 VLM（视觉语言模型）评测 Demo：对电商订单详情页截图进行**结构化信息抽取 + 金额一致性校验（reasoning）**，并用 **Goldset 2.0真值表**，自动批量生成总体指标与分桶报告。

---

## 目标与产出

**目标**
- 从订单详情页截图中抽取关键字段（5 个）
- 对金额字段做一致性校验：`paid_amount = original_amount - discount_amount`
- 基于真值表（GT）自动评测并输出分桶报告（Domain / PageState / Info Completeness）

**产出**
- 批量预测结果（JSONL / 回填表）
- 自动评测报告（Excel，多 sheet：summary / error cases / bucket metrics）

---

## 字段与任务定义

**抽取字段（5）**
- `order_outcome`：订单结果（`SUCCESS` / `CLOSED`）
- `paid_amount`：实付金额
- `original_amount`：原始金额 / 商品总价 / 应付金额（以业务口径定义）
- `discount_amount`：优惠金额 / 立减 / 共减
- `order_time`：下单时间 / 创建时间

**校验任务（reasoning）**
- 当 `paid_amount`、`original_amount`、`discount_amount` 三者 `state=OK` 时：
  - 校验等式成立输出 `YES/NO`
- 否则输出 `SKIP` 并解释跳过原因

---

## Repo 文件说明

- `prompt.txt`：统一 Prompt（用于批量调用模型）
- `gold set 2.0.xlsx`：真值表（GT）
- `pred_results.xlsx`：模型输出回填表（含 `model_pred_json`）
- `pred_results.jsonl`：批量预测输出（每行一个 JSON）
- `run_batch_vlm.py`：脚本 1：批量跑模型 + 回填生成 `pred_results.xlsx/jsonl`
- `eval_goldset_v3.py`：脚本 2：自动评测 + 分桶报告生成
- `report_3.0.xlsx`：评测报告（自动生成）
- `标注指南.pdf`：标注规则与字段口径
- `项目复盘.pdf`：项目复盘与迭代思路

---

## 环境依赖

- Python 3.9+（建议 3.10/3.11）
- 依赖包：
  - `openai>=1.0`（用于火山/ARK OpenAI SDK 调用）
  - `pandas`
  - `openpyxl`

安装：
```bash
python3 -m pip install --upgrade "openai>=1.0" pandas openpyxl
```

---

## 评测报告（report_3.0.xlsx）结构

	•	summary：总体指标（schema、5字段准确率、reasoning 准确率、单字段准确率）
	•	error_cases：错例列表（含原始预测 JSON 便于溯源）
	•	bucket_by_domain：按 Domain 分桶指标
	•	bucket_by_pagestate：按 PageState（FULL/OCCLUDED/SHRINK）分桶指标
	•	bucket_by_infocompleteness：按 Info Completeness 分桶指标

---

## 评测口径概览（简要）

	•	schema_ok_rate：预测 JSON 是否满足结构与枚举约束
	•	all_5_fields_acc：5 个字段（value/state）是否全部与 GT 一致（严格匹配）
	•	reasoning_acc：reasoning.result 与 GT 是否一致（YES/NO/SKIP）
	•	*_acc：单字段准确率（严格：state 必须一致；若 OK 则 value 必须一致）

更完整的口径与示例见：标注指南.pdf / 项目复盘.pdf
