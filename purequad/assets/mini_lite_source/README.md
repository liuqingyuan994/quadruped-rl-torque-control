# Mini Lite 模型整理说明

整理日期：2026-10-19  
保存位置：`E:\机器狗文件\model`  
模型来源：用户提供的 `E:\机器狗文件\Webots_Project`。

## 先使用哪些文件

| 需要 | 文件 | 说明 |
|---|---|---|
| 机器人结构、坐标和关节 | `mini_lite/urdf/MiniLiteC.urdf` | 18 个 link、17 个 joint，其中 12 个活动关节；网格路径已统一为相对于 URDF 的 `../meshes/` |
| 机器人外观网格 | `mini_lite/meshes/` | BASE、FL_HIP、FR_HIP、HL_HIP、HR_HIP、L_THIGH、R_THIGH、SHANK，共 8 个 DAE |
| Webots 完整外观 PROTO | `mini_lite/webots/protos/JueyingMiniLite.proto` | 原始 R2022a 模型；几何数据内嵌，没有额外纹理依赖 |
| Webots 原始简化初始场景 | `mini_lite/webots/worlds/mark_0_init_model.wbt` | 原始场景自带简化机器人；无需外部控制器；不是加载上述完整外观 PROTO 的演示 |
| 其他 Mini Lite 场景参考 | `mini_lite/webots/reference_worlds/` | 5 个原始场景，保留传感器、接触和 Supervisor 配置以供参考 |
| 坡面和粗糙地形 | `terrain/meshes/` | 17 个原始 STL；可按实验需要选择，不是机器人本体的加载依赖 |
| 地形 CAD 源文件 | `terrain/cad/` | 7 个 SLDPRT，保留以便后续修改地形；Webots 不直接加载此格式 |
| 完全未改动的 URDF | `provenance/MiniLiteC.urdf.original` | 原始字节副本，仅用于追溯；使用时优先选择上面的整理版 URDF |

本次完成文件筛选、复制、去重和静态检查；未运行 Webots 仿真或真机。

**原始 URDF 有已确认的动力学参数错误：HR_HIP 的 `izz=-4.1425e-7 kg·m²`，惯量矩阵存在负特征值。该值在整理版中原样保留。开始动力学仿真或训练前，必须核定此参数；本包不属于可直接训练的已验证模型。**

## 为什么选择这组模型

运动 SDK 手册第 2-3 页描述 Mini Lite 为四腿、12 关节，按 FL / FR / HL / HR 分腿，每条腿依次为 HipX、HipY、Knee。所选 URDF 的机器人名称为 `JueyingMiniLite`，12 个活动关节及 PROTO 中的 12 个电机名称与此对应；URDF 的大腿、小腿长度均为 0.18 m，足底碰撞球半径为 0.02 m，也与手册第 5 页相符。

学习路径中有 A2、机械臂、灵巧手等平台描述，但这些不能作为将其他机器狗模型替换为 Mini Lite 的依据。本次以用户提供的 Mini Lite 手册、SDK 命名和实际模型结构筛选文件；未把学习路径中的平台名称当作新的执行指令。

感知手册包含深度相机和激光雷达开发方向，因此保留坡面与粗糙地形资源作为可选实验素材。但是所选 Mini Lite URDF/PROTO 不含相机、激光雷达或完整传感器外参，不能据此认为已经具备感知仿真配置。

## 来源与唯一修改

- URDF：`Webots_Project/JY_webots_rev/urdf/MiniLiteC.urdf`。
- 8 个 DAE：`Webots_Project/JY_webots_rev/meshes/`。原 `urdf/meshes/` 中的 8 个文件与它们逐一同哈希，因此只保留一份。
- PROTO：`Webots_Project/JY_webots_rev/protos/JueyingMiniLite.proto`。找到的 5 份同名 PROTO 完全同哈希，只保留一份，原位置列在 `docs/duplicate_proto_sources.csv`。
- 场景：`Webots_Project/JY_webots_rev/worlds/`，保留含 Mini Lite 的 6 个场景。
- 地形及 CAD：`Webots_Project` 顶层的 STL 和 SLDPRT。
- 仅整理版 URDF 修改了网格引用：把 `filename="./meshes/..."` 统一为 `filename="../meshes/..."`。原本 BASE 已使用 `../meshes/BASE.dae`。关节、质量、惯量、限位、碰撞体和单位均保持原值。
- 其余 40 个选中文件均按原始字节复制。全部来源、修改标记、源 SHA-256 和整理后 SHA-256 见 `docs/source_manifest.csv`。

## Webots 场景的使用范围

源文件声明 Webots R2022a。`worlds/mark_0_init_model.wbt` 可以作为查看原始简化模型的入口；本次只检查文件内容和依赖，没有启动 Webots 验证。

`protos/JueyingMiniLite.proto` 是独立的完整外观模型，可在自己的 Webots 世界中导入使用；现有 `mark_0_init_model.wbt` 内嵌了简化机器人，不能把二者视为同一套完全一致的动力学配置。

参考场景的作用：

| 文件 | 保留原因和依赖 |
|---|---|
| `Figure.wbt` | 含 3 个 Mini Lite 实例，用于外观或场景配置参考，无显式外部控制器 |
| `mark_1_basic_ops.wbt` | 含 IMU、陀螺仪、足端 TouchSensor 等配置；机器人使用 `<extern>` 控制器 |
| `mark_1.5_Supervisor.wbt` | 含机器人与 Supervisor，依赖 `<extern>` 和原项目 `Supervisor` 控制器 |
| `mark_1.5_Supervisor - origin_BK.wbt` | 原始备份场景，独立保留；同样依赖上述控制器 |
| `mark_1.5_Supervisor修改.wbt` | 原始修改版本，独立保留；同样依赖上述控制器 |

控制程序未纳入本模型包。若要运行参考场景，应先回到原 `JY_webots_rev` 项目核对控制器和运行环境。Webots 自带的 RectangleArena、TexturedBackground 等资源仍由 Webots 安装提供，未复制通用纹理库。

## 已发现的参数差异

这是一组按名称和结构匹配的本地模型，不能确认它与当前实机是同一硬件版本，因此未擅自按手册改参数。

| 项目 | 提供的运动手册 | 当前 URDF / PROTO | 本次处理 |
|---|---|---|---|
| 大腿、小腿长度 | 均 0.18 m | 均 0.18 m | 原值保留 |
| 足底半径 | 0.02 m | URDF 足底碰撞球为 0.02 m | 原值保留 |
| 身体质量 | 第 5 页为 8.5 kg，含电池 | URDF 中 BASE_INER 为 5.6352 kg；URDF 显式质量总和约 10.00939 kg | 标记差异，未重标定 |
| HipX / HipY 力矩 | 第 6 页额定 10 Nm、峰值 26 Nm | URDF effort / PROTO maxTorque 为 13.2 Nm | 标记差异，未修改 |
| Knee 力矩 | 第 6 页额定 17 Nm、峰值 51 Nm | URDF effort / PROTO maxTorque 为 28 Nm | 标记差异，未修改 |
| 右后髋惯量 | 手册未给出可逐项核对的该连杆完整矩阵 | URDF 的 HR_HIP `izz=-4.1425e-7` 为负；PROTO 对应处为 `0.00014559` | 标记源文件错误，未自动替换 |
| 传感器 | 感知手册描述相机、激光雷达等实机功能 | 核心 URDF/PROTO 未配置相机、雷达或 IMU；部分 WBT 场景有 IMU 和足端传感器 | 保留实际配置，未编造外参 |

URDF 的 BASE 未给出 inertial，足端使用极小质量，部分连杆未设置 collision；PROTO 的根部还使用空的 `Physics {}`。因此，URDF 的显式质量求和不应直接当成所有 Webots 场景的实际总质量。文件可作为模型开发素材；用于 RL/Sim2Real 前还需要核对这些定义。

## 地形的尺寸与坐标

STL 本身不声明单位。这批文件的原始包围盒尺度差异较大，部分坐标跨度达数百，不能统一假定为米或统一乘 0.001。原坐标轴和比例全部保留。

`docs/terrain_dimensions.csv` 列出各 STL 的三角形数量、XYZ 最小值、最大值和尺寸。选择地形后应先确认预期尺寸、上方向和导入比例，再放入世界。文件名中的 rad、BIG 等仅保留原命名，不作为已核实的坡度或尺度结论。

## 排除的内容

- `unknown/protos/Spot*.proto` 和 Spot 场景：Boston Dynamics Spot，非 Mini Lite。
- `Quasi-static-control-of-quadruped-robot-main` 的 `dog_c.urdf` / `anymal.urdf`：其他机器人；未作为 Mini Lite 本体复制。
- `my_first_simulation`、`my_surpervisor`：轮式机器人和基础教学示例。
- `mark_1.5_Supervisor_test_torque.wbt`：球体/关节力矩实验，不含 Mini Lite 本体。
- 空世界、重复子项目、.git、虚拟环境、二进制程序、缓存、日志、控制器和无模型引用的通用纹理：本次模型整理不需要。
- `jy_sdk_ml`、`Lite_SDK_master` 与 `ros2_21_tutorials`：用于识别接口或教学背景，未从中混入其他本体模型。

## 校验文件

- `docs/source_manifest.csv`：逐文件来源与哈希。
- `docs/joint_interfaces.csv`：12 关节在 URDF 中的顺序、父子连杆、轴向、原点和限位；此顺序不等于已验证的实机数组映射。
- `docs/duplicate_proto_sources.csv`：5 个相同 PROTO 的原位置。
- `docs/terrain_dimensions.csv`：17 个 STL 的原始尺寸。
- `docs/validation.json`：URDF 连通树、13 处网格引用、8 个 DAE 单位/纹理依赖、显式惯量检查（包含 HR_HIP 失败项）、Webots 文本括号和 STL 结构检查结果。
- `docs/package_checksums.json`：本模型包各文件的 SHA-256，不包含该清单本身。

以上检查属于静态检查；不代表 Webots 解析、ROS/MuJoCo 导入、行走、训练或实机验证通过。SLDPRT 只完成复制及哈希校验，未用 SolidWorks 打开。
