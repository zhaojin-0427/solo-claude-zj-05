# Foucault 刀口仪镜面分析 API

面向业余反射镜制作者的本机服务：把 Couder 遮罩分区读数还原为可比较的镜面/波前
误差。Python 完成光学计算，FastAPI + Pydantic 接收测试数据，SQLite 保存分析版本。

## 运行

```bash
pip install -r requirements.txt
python3 run.py                 # 或 uvicorn foucault.app:app
# 数据库路径可用环境变量 FOUCAULT_DB 指定（默认 ./foucault.db）
```

交互式文档：`http://127.0.0.1:8000/docs`

## 光学模型与公式

纵向像差（LA）一律换算为**固定光源等效**值：

| 光源模式 | 刀口位移与 LA 的关系 |
|---|---|
| `fixed`（固定光源） | 刀口位移 = LA |
| `moving`（移动光源） | 刀口位移 = LA / 2（读数差 ×2） |

- 目标圆锥（圆锥常数 K）的理想纵向像差：`LA_ideal(r) = -K · r² / R`
- 面形斜率误差：`α(r) = -LA_err(r) · r / (2R²)`，面形误差 `h(r) = ∫₀ʳ α dr`
  （实际面相对目标圆锥，正值为"凸起"）
- 波前误差：`W = 2h`（反射加倍）
- 最佳拟合：对 h 做 `{1, r², r⁴}` 最小二乘，`ΔK = 8R³·a₄`，`ΔR = -2R²·a₂`
- 横向像差（最佳焦点面）：`TA(r) = LA_resid(r) · r / R`
- **RMS 按圆形口径面积加权**（面元 dA = 2πr dr，权重 ∝ r）；均匀半径统计会
  低估边缘误差，使 RMS 偏低、Strehl 偏高
- Strehl 估计（Maréchal 近似）：`S ≈ exp(-(2π σ_W / λ)²)`
- 分区等效半径：`r_m = sqrt((r_in² + r_out²)/2)`

## 主要端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/tests` | 创建测试批次并自动生成分析版本 v1 |
| GET | `/api/tests` / `/api/tests/{id}` | 批次列表 / 详情（含读数状态） |
| POST | `/api/tests/{id}/readings/exclude` | 带原因剔除读数（可自动重分析） |
| POST | `/api/tests/{id}/readings/restore` | 恢复被剔除读数 |
| POST | `/api/tests/{id}/readings/freeze` | 冻结可信读数（冻结后不可剔除） |
| POST | `/api/tests/{id}/analyze` | 用当前有效读数（可换算法选项）生成新版本 |
| GET | `/api/tests/{id}/versions[/n]` | 版本列表 / 版本完整结果 |
| POST | `/api/compare` | 对比多个测试批次（遮罩一致时给出逐区差值） |
| POST | `/api/tests/{id}/corrections/search` | 约束下搜索分区修正量并排序 |

### 创建校验（不满足则 422 拒绝）

- 分区重叠、分区越出镜面（外半径 > 口径/2）
- 单位不明（`unit` 只接受 `mm` / `in`；波长固定 nm）
- 每区有效读数不足（默认 ≥ 2，可在 `options.min_readings_per_zone` 调整）
- 分区数 < 2、非正口径/曲率半径/波长、非有限数值

### 分析流程

零点校正（最内区或全区均值参考）→ 仪器偏移扣除 → 每区离散度汇总
（std/range/sem）→ LA 误差积分出面形轮廓 → 最佳拟合圆锥 → 横向像差、
面积加权 RMS 与峰谷波前误差、Strehl → 标出超出误差带的区段
（默认 ±λ/4，`options.error_band_waves` 可调）。

### 版本与哈希

每个版本冻结：全部原始读数（含被剔除者及其原因）、常量、算法选项与输入
哈希（SHA-256，只覆盖真正参与分析的数据）。输入未变时重复 `analyze` 幂等
复用最新版本；重复读取同一版本结果不变。

### 修正量搜索

在 `max_removal_nm`（每区最大磨除量）、`edge_zone_max_removal_nm` /
`preserve_edge`（边缘保留）、`max_mean_removal_nm`（面积加权平均深度上限）
约束下，对一组平滑权重解盒约束二次规划，候选方案按
**(剩余波前 RMS, 修正平滑度, 材料去除量)** 升序排列。

## 测试

```bash
python3 -m pytest tests/ -q
```
