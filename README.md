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
| POST | `/api/tests` | 创建测试批次并自动生成分析版本 v1（可直接给分区，或引用遮罩版本） |
| GET | `/api/tests` / `/api/tests/{id}` | 批次列表 / 详情（含读数状态） |
| POST | `/api/tests/{id}/readings/exclude` | 带原因剔除读数（可自动重分析） |
| POST | `/api/tests/{id}/readings/restore` | 恢复被剔除读数 |
| POST | `/api/tests/{id}/readings/freeze` | 冻结可信读数（冻结后不可剔除） |
| POST | `/api/tests/{id}/analyze` | 用当前有效读数（可换算法选项）生成新版本 |
| GET | `/api/tests/{id}/versions[/n]` | 版本列表 / 版本完整结果 |
| POST | `/api/compare` | 对比多个测试批次（遮罩一致时给出逐区差值） |
| POST | `/api/tests/{id}/corrections/search` | 约束下搜索分区修正量并排序 |
| POST | `/api/mask-schemes` | 创建遮罩方案并生成版本 v1（常量与制作限制冻结） |
| GET | `/api/mask-schemes` / `/api/mask-schemes/{id}` | 方案列表 / 详情（含版本列表） |
| POST | `/api/mask-schemes/{id}/versions` | 新参数生成新版本（参数相同则幂等复用） |
| GET | `/api/mask-schemes/{id}/versions/{v}` | 版本完整布局（环带/开窗/预测/指标） |
| GET | `/api/mask-schemes/{id}/versions/{v}/svg` | 1:1 遮罩 SVG（尺寸线 + 校准尺） |
| POST | `/api/mask-schemes/search` | 在分区数 × 桥宽 × 权重模式范围内搜索可行布局并排序 |

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

### Couder 遮罩设计

遮罩方案把**口径、曲率半径、目标圆锥常数、光源模式与制作限制**（计划分区数、
中心禁测半径、最小环宽、桥宽、刀口尺分辨率、打印缩放校准）冻结为独立版本，
每个版本保存计算出的环带边界、等效半径、左右开窗与指标，创建后不可变；
参数相同重复建版本时幂等复用。

- 分区权重：`equal_area`（等面积）、`equal_width`（等环宽）、`custom`
  （自定义权重，各环带面积 ∝ 权重）
- 开窗几何：每环带左右两个矩形窗，竖直方向以等效半径 r_m 为中心，窗高在
  "相邻开窗间距 ≥ 桥宽" 且 "窗高 ≤ 环宽 − 桥宽" 约束下取最大（投影法求解）；
  水平方向由环带边界弦长决定（内环带窗底边高于 r_in 时左右窗在中心线相连）
- **拒绝生成**（422，逐条指明相关环带）：分区越界（禁测半径 ≥ 镜面半径等）、
  环宽不足（< 最小环宽）、结构无法留桥（环宽 ≤ 桥宽、等效半径间距 < 桥宽、
  留桥约束下窗高为零）、开窗相交（开窗越出本环带边界侵入相邻环带）
- 刀口位移预测：各区理想刀口位移 `LA_ideal(r_m)`（移动光源减半），以最内
  环带为零点；相邻区对读数差低于刀口尺分辨率的区对被标出
  （`layout.prediction.unresolvable_pairs`）
- 边界搜索：`/api/mask-schemes/search` 在分区数 × 桥宽 × 权重模式范围内
  枚举，仅保留可行布局，按 **(最小可分辨位移, 面积均衡度, 制作余量)** 降序；
  候选 `params` 可直接用于创建遮罩版本
- SVG：`.../versions/{v}/svg` 输出 1:1 图纸（mm 单位，含直径/半径/窗高
  尺寸线、校准尺与参数说明），打印缩放校准 `print_scale` 已计入全部几何，
  100% 打印后用校准尺实测核对
- 批次冻结：`POST /api/tests` 传 `mask_scheme_id`（可指定 `mask_version_no`，
  缺省最新）+ `zone_readings`（按遮罩环带顺序）即可建批；常量与环带边界
  取自遮罩版本冻结值（重复提供且不一致则 422），批次记录遮罩版本 ID，
  此后遮罩另建版本不影响既有批次与分析

## 测试

```bash
python3 -m pytest tests/ -q
```
