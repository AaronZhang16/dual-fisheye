# 双鱼眼离线标定与全景拼接

面向两个背靠背、有效视场大于 180° 的鱼眼镜头。Python 函数与运行脚本分离，读取文件夹中的多帧静态图像，不调用相机，也不直接解码视频。

实现链路：**棋盘格 → DS 内参 → 重叠区 SIFT 匹配 → 单位射线 → 球面八点法 RANSAC → 外参优化 → 等距柱状全景映射 → 羽化融合**。

这是可运行的工程基线，不是已通过实机认证的量产标定工具。真实镜头精度需用独立实拍数据验证。合成数据与真实镜头存在差异，测试通过不代表真实镜头已经标定。

## 1. 项目结构

```text
dual_fisheye/
├── pyproject.toml             # src 包布局、命令行入口、Black 配置
├── requirements.txt           # 固定版本参考环境
├── configs/default.json       # 数据位置与算法参数
├── scripts/
│   ├── run.py                 # 轻量运行入口
│   └── synthetic_demo.py      # 生成可复现的合成演示数据
├── src/dual_fisheye/
│   ├── __main__.py            # python -m dual_fisheye 入口
│   ├── datasets.py            # 合成数据集生成流程
│   ├── models.py              # Double Sphere 投影、反投影、有效域
│   ├── intrinsics.py          # 棋盘格检测和内参估计
│   ├── features.py            # 重叠掩膜与原图 SIFT 匹配
│   ├── extrinsics.py          # 球面外参初始化和非线性优化
│   ├── stitching.py           # 全景映射与融合
│   ├── io_utils.py            # 文件读取、严格配对、JSON 保存
│   ├── visualization.py       # 已计算结果的可视化
│   ├── workflow.py            # 多阶段调度、调试文件输出
│   ├── cli.py                 # 参数解析、模式与错误处理
│   └── synthetic.py           # 合成场景渲染，仅用于验证
├── tests/test_geometry.py     # 几何精度、退化与输入检查
├── data/                      # 本地测试数据，Git 忽略内容
└── outputs/                   # 运行结果，Git 忽略
```

采用标准 src 布局：只有 `src/dual_fisheye/` 是可安装的 Python 包，`scripts/` 和 `tests/` 不打包。脚本及测试通过已安装的包导入，不修改 `sys.path`，也不要求配置 `PYTHONPATH`。修改源码后，可编辑安装会立即使用新代码。

算法模块不负责文件保存；诊断绘图在 visualization.py；运行编排在 workflow.py。避免把标定优化塞进运行脚本。

## 2. 环境准备

推荐 Python 3.10 或 3.11，在项目根目录使用 PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m dual_fisheye --help
```

运行脚本和测试前必须先完成上面的可编辑安装。不能只下载代码后直接运行脚本。安装后可以使用 `python -m dual_fisheye` 或 `dual-fisheye`；旧的 `python scripts/run.py` 入口仍保留。

后文的 `python` 指这个环境的 Python；也可以始终使用 `.venv\Scripts\python.exe`。开发格式化可额外安装 `black==24.10.0`。

参考依赖为 NumPy 1.26.4、SciPy 1.13.1、OpenCV 4.10.0.84；准确的本地验证版本记录在 VALIDATION.md。调试运行自动记录实际版本。当前无需 OpenGV、ROS 或 Ceres，外参及优化用 NumPy/SciPy 实现。

## 3. 准备文件夹和配置

```text
data/
├── intrinsics/cam1/   # 镜头 1 棋盘格，可独立拍摄
├── intrinsics/cam2/   # 镜头 2 棋盘格，可独立拍摄
├── extrinsics/cam1/   # 同步自然纹理图：0001.png、0002.png ...
├── extrinsics/cam2/   # 对应同名图：0001.png、0002.png ...
├── stitch/cam1/       # 要拼接的图像序列
└── stitch/cam2/       # 对应同名图像序列
```

支持 PNG/JPEG/BMP/TIFF。只读各指定目录的直接子文件，按文件名排序，建议使用补零序号。两路外参与拼接数据必须有完全相同的文件名主体，不允许重复主体或漏帧。文件名配对不能证明时间同步，采集时仍需保证同步或场景静止。视频请预先提取同步帧；本项目不含相机采集或视频提帧模块。

所有数据内容、视频、输出、缓存和虚拟环境都被 `.gitignore` 排除；空目录用 `.gitkeep` 保留。`.dockerignore` 同样排除数据。普通项目生成不会自动执行 git init、提交或推送。

修改 `configs/default.json`：路径相对于 `project_root`；该字段相对于配置文件所在目录。默认 `project_root: ".."` 表示项目根目录。`--config` 相对于当前工作目录，`--output`、`--cameras`、`--rig` 相对于项目根目录，也支持绝对路径。

| 参数 | 意义与注意事项 |
|---|---|
| `board_inner_corners` | 内角点列数、行数，不是方格数量 |
| `square_size_m` | 方格边长，米；与实物一致 |
| `fov_deg` | 每镜头可用圆形全视场角，不是对角线视场；这是外部提供的有效域，不会被自动估计 |
| `seed_params` | 可选 DS 初值 `[fx,fy,cx,cy,xi,alpha]`；null 假设圆形画面填满图像短边 |
| `max_frames` | 标定最多使用多少张/对图像；按排序取前 N 帧，要先均匀抽样 |
| `max_nfev` / `loss_px` | 内参优化预算、鲁棒损失像素尺度 |
| `max_rms_px` | 内参整体 RMS 拒绝阈值；2px 默认值只是基线，不代表验收标准 |
| `initial_rotation_xyz_deg` | 从 cam1 到 cam2 的粗旋转，SciPy 小写 xyz 外禀欧拉角，默认绕 y 轴 180° |
| `overlap_margin_deg` | 粗重叠搜索余量，不是匹配验收阈值 |
| `ratio` / `max_features` | SIFT 比率阈值、特征数量 |
| `per_bin_cap` | 每帧每个环向分区的匹配数上限，防止局部纹理主导 |
| `ransac_iterations` / `threshold_deg` | RANSAC 次数、近似对称球面极线角误差阈值 |
| `min_parallax_deg` | 平移可观测性粗筛阈值，过小视差直接失败 |
| `baseline_m` | 已知光心基线；null 时只保存平移方向，绝不虚构米制平移 |
| `width` | 偶数全景宽度，高度自动为宽度一半 |
| `seed` | 固定 NumPy/OpenCV/RANSAC 随机种子 |

## 4. 步骤一：估计内参

```powershell
python -m dual_fisheye intrinsics --mode debug --output outputs/intrinsics-01
```

输入：各镜头的全视场棋盘格图像、方格尺寸、DS 初值。

处理：在原图检测亚像素角点；使用初始 DS 反投影及局部虚拟透视相机初始化每张板位姿；联合优化 6 个 DS 参数与所有板位姿。

目标：最小化原图重投影误差，Soft-L1 降低异常定位的影响。

输出：`outputs/intrinsics-01/cameras.json`。调试文件还包含角点、检测失败列表、每帧板位姿、预测像素、整体及逐帧误差。

优势：保留超过 90° 入射角的射线，不需要将整张图拉成透视图。问题：该优化是局部求解，差初值、边缘角点检测失败或覆盖不足都会导致不可靠结果。

至少需要 6 张可检测图像，建议实际使用 40～80 张多距离、多倾角且覆盖边缘的图。不要仅拍图像中心。圆形被裁切、有效区域偏心时，应提供更好的 DS 初值。初始相机无法反投影棋盘格会明确报错，不能靠无限放大优化预算解决。

## 5. 步骤二：估计外参

```powershell
python -m dual_fisheye extrinsics --cameras outputs/intrinsics-01/cameras.json --mode debug --output outputs/extrinsics-01
```

输入：已标内参、多对同步自然场景图像、背靠背粗旋转。

处理：

1. 根据内参、粗旋转及余量构造重叠掩膜。
2. 原始鱼眼图中提取 SIFT，双向比率匹配，环向分区限额。
3. 通过 DS 将匹配像素反投影为单位射线 `b1,b2`。
4. 多帧共同约束一个外参；球面八点法 RANSAC 拟合 `b2.T @ E @ b1 = 0`。
5. 分解 E，使用沿射线的正深度选择 R,t，不用 `z>0` 排除超过 180° 的有效观测。
6. 固定内参，对旋转和单位平移方向进行 Soft-L1 球面极线误差优化。

目标：找到多帧一致的刚性关系 `X2 = R21 @ X1 + t21`。

输出：`rig.json`，包含 `R21`、`t21_direction`、`baseline_m`、`t21_m`。没有尺度时后两项为 null。注意 cam2 光心在 cam1 坐标系中的位置是 `-R21.T @ t21`，不是直接把 t21 当作该位置。

优势：不对整幅图重采样，支持负 z 射线，多帧降低场景偶然性。问题：原图边缘特征匹配可能困难，八点法在窄环带、低视差或平面主导场景下不稳定；当前没有跨时间特征跟踪或完整场景 BA。

采集应覆盖重叠环带各方向与不同深度；缓慢转动并停稳拍摄可降低同步风险。全是远景时旋转可能可估，但本项目的完整 R,t 求解会因平移弱约束拒绝数据。首次内参/尺度应离线建立，不必每次使用都重新标定。

## 6. 步骤三：拼接

```powershell
python -m dual_fisheye stitch --cameras outputs/intrinsics-01/cameras.json --rig outputs/extrinsics-01/rig.json --output outputs/stitch-01
```

输入：内参、外参、要拼接的同名图像对。

处理：为等距柱状图的每个像素生成 cam1 空间方向；用 R21 转换到 cam2；分别经过 DS 投影获得查找表；双线性采样并按距离可用 FOV 边界的角余量羽化融合。所有帧复用查找表。

目标：输出稳定的远景全景序列。输出为 `panoramas/原文件名.png`。

坐标：各相机 x 向右、y 向下、z 向前；全景中心朝 cam1 +z，北极朝 cam1 -y；没有 IMU，因此不是地理北方，也不自动保持水平。

此步骤采用无限深度假设，**不使用平移补偿近景**。固定映射无法消除光心分离的近景视差，融合也不能修复严重重影。当前没有深度估计、光流、动态接缝、曝光匹配或暗角校正。覆盖不足处填黑，可在调试 coverage.png 中检查。

## 7. 完整运行和两种模式

从棋盘格开始执行全部步骤：

```powershell
python -m dual_fisheye run --mode normal --output outputs/full-normal-01
python -m dual_fisheye run --mode debug --output outputs/full-debug-01
```

复用内参，但每次重新估计外参：

```powershell
python -m dual_fisheye run --cameras outputs/intrinsics-01/cameras.json --mode debug --output outputs/recalibrate-01
```

| 模式 | 行为 |
|---|---|
| normal | `run`/`stitch` 只写最终 panorama PNG，不保存中间参数、不打印进度；错误仍写 stderr 并返回非零码 |
| debug | 保存最终 PNG，以及每个实际执行步骤的结果和 run.log；读取已有参数的步骤不会假造角点等结果 |

`intrinsics` 和 `extrinsics` 是独立标定命令，即使普通模式也会写参数 JSON，因为这是这两个命令的必要结果。完整 `run --mode normal` 中间参数只保存在内存；需要复用时先分步标定，或使用 debug 导出的参数。

输出目录非空会拒绝运行，避免覆盖已有结果或混入上一次调试文件。请为每次运行选择新的 `--output`。

调试目录：

```text
debug/
├── 00_run.json                # 配置、命令、环境版本
├── 01_corners/                # 检测点、被拒图像、角点覆盖图
├── 02_intrinsics/             # DS 参数、板位姿、重投影、逐帧误差
├── 03_overlap/                # 两镜头重叠搜索掩膜
├── 04_matches/                # 匹配连线，展示最多150条
├── 05_bearings/               # 每帧所有保留像素及单位射线 NPZ
├── 06_extrinsics/             # 优化外参、内点、逐点误差、内点图
├── 07_mapping/                # 查找表、权重 NPZ、覆盖图及覆盖率
├── 08_projection/             # 融合前两路等距柱状投影
├── cameras.json / rig.json    # 完整运行中可复用参数
└── run.log
```

NPZ 可用 `numpy.load(...)` 读取。调试输出是阶段计算结果，不逐次保存优化器的全部迭代状态。

## 8. 无相机合成演示和测试

```powershell
python scripts/synthetic_demo.py
python -m dual_fisheye run --config configs/synthetic.json --mode debug --output outputs/synthetic-debug
python -m dual_fisheye run --config configs/synthetic.json --mode normal --output outputs/synthetic-normal
python -m unittest discover -s tests -v
python -m black --check src scripts tests
```

演示会生成真实光线/平面求交的棋盘格图，以及具有有限深度的纹理房间双鱼眼图，写入被忽略的 `data/synthetic/`。合成棋盘格使用已知模型初值，这是可复现实验，不证明任意镜头能从默认初值收敛。两镜头使用相同合成内参以减少演示复杂度；真实数据仍分别标定。

生成器不会覆盖已有 `data/synthetic`。`truth.json` 保存几何真值。测试检查超过 180° 的投影往返、含外点的背靠背 R,t 恢复、内参留出射线精度、全景覆盖及输入配对拒绝。详细实际测试记录见 VALIDATION.md。

## 9. 验证、维护与后续扩展

必须用独立实拍数据评估：边缘重投影误差、环带各方向误差、重复标定的 R/t 稳定性，以及远中近景接缝。默认阈值只是初始配置，不能当作产品验收指标。

若只有近景错位而远景正常，首先考虑基线视差，不应放开内参硬拟合。若所有距离都错位，检查变换方向、内参边缘、同步及粗重叠区域。若优化报预算耗尽，可先检查图像覆盖/初值，再调整 max_nfev。

后续可替换 features.py 为局部透视块匹配，可在 extrinsics.py 增加旋转专用模式或重投影 BA，可在 stitching.py 增加深度补偿和动态接缝；保持 I/O 与算法分离。当前内参使用稀疏数值雅可比；外参使用球面极线优化，并未宣称实现 Ceres/OpenGV 或完整 BA。

参考资料（模型公式独立用 NumPy 实现，未复制 Basalt 源码）：

- [Double Sphere 论文与说明](https://cvg.cit.tum.de/research/vslam/double-sphere)
- [Basalt DS 参考模型](https://github.com/VladyslavUsenko/basalt-headers)
- [Kalibr 模型选型](https://github.com/ethz-asl/kalibr/wiki/supported-models)
- [OpenGV 单位射线几何](https://laurentkneip.github.io/opengv/page_how_to_use.html)
