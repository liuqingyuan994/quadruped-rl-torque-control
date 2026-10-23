from pathlib import Path
from zipfile import ZipFile
import argparse
import ast
import difflib
import hashlib
import json
import re


HERE = Path(__file__).resolve().parent
PROJECT = Path('D:/Download/桌面/rl')
TARGET = Path('E:/obsidian/lqy的知识库/机器狗学习/相关项目/Mujoco仿真/概述.md')
START = '#### 五个奖励版本分别做什么\n'
END = '#### 版本 2、4 在哪里生效\n'

SECTION = r'''#### 五个奖励版本分别做什么

下面给出 **`reward_version=1～5` 各自最终生效的完整单步奖励函数**。公式按 `stage='path'`、`reference_reward=False` 整理；站立与基础运动任务的区别在本段末尾说明。当前默认策略采用版本 5，前面的 20 项表与下述 $r_t^{(5)}$ 对应。

| 奖励版本 | 主要目标 |
| --- | --- |
| 1 | 基础姿态、高度、速度与沿路径前进 |
| 2 | 加强持续速度跟踪、横向纠偏和竖直速度约束 |
| 3 | 更严格地保持机身水平与目标高度 |
| 4 | 增加关节和机身运动的平滑性惩罚 |
| 5 | 保留平滑要求，针对当前 URDF 调整两项差分惩罚系数 |

**公式中的符号。** 沿用本节前面的 $h,\rho,\vartheta,e_y,e_\psi,q,q_0,u_t,\Delta s$ 等定义；$q_0$ 为中立关节姿态，$\operatorname{mean}$ 对十二个关节求平均。所有状态项对应当前控制步；为便于阅读，部分状态量省略下标 $t$。定义：

$$
S=\rho^2+\vartheta^2,\qquad
E_0=(v_x-v^\star)^2+v_y^2
$$

$$
E_{\mathrm{avg}}=E_v=(\bar v_x-v^\star)^2+\bar v_y^2
$$

$E_0$ 使用瞬时速度，$E_{\mathrm{avg}}$ 使用前文定义的一阶平均速度。路径任务的目标速度在前 0.6 s 为 0，之后为 0.13 m/s；直立、高度项使用前进相关系数 $m_t=0.1+0.9\operatorname{clip}(v_x/0.13,0,1)$。

**五个版本共用的部分。** 将各版相同的九项合并为 $C_t$：

$$
\begin{aligned}
C_t={}&\Delta t\Big[
1-40n_{\mathrm{bad}}
-0.0002\sum_i\dot q_i^2
-0.00005P\\
&\quad-0.02\operatorname{mean}(u_t^2)
-8e_\psi^2\\
&\quad+2\exp\left(-\frac{\operatorname{mean}((q-q_0)^2)}{0.02}\right)
\Big]\\
&-0.02\operatorname{mean}\big((u_t-u_{t-1})^2\big)
+r_{\mathrm{terminal}}
\end{aligned}
$$

这九项依次对应持续运行、非足部触地、关节速度、机械功率、动作幅值、航向误差、中立姿态、相邻动作突变和终止反馈。一次性终止奖励为：

$$
r_{\mathrm{terminal}}=
\begin{cases}
-50,&\text{失败}\\
+100,&\text{未失败且到达路径终点}\\
0,&\text{其他情况，包括单纯超时}
\end{cases}
$$

$P=\sum_i\lvert\tau_i\dot q_i^{\mathrm{pre}}\rvert$ 使用物理步进前的关节速度。$u_t=0.4\operatorname{clip}(a_t,-1,1)$ 是缩放动作，尚未乘关节力矩限值。共同项 $C_t$ 已包含航向误差与终止奖励，下面的总公式不再重复添加它们。

**版本 1：基础奖励。**

$$
\boxed{
\begin{aligned}
r_t^{(1)}=C_t+\Delta t\Big[&6m_t\exp\left(-\frac{S}{0.02}\right)\\
&+4m_t\exp\left(-\left(\frac{h-0.31}{0.02}\right)^2\right)\\
&+10\exp\left(-\frac{E_0}{0.04^2}\right)
-120e_y^2
\Big]+20\Delta s
\end{aligned}}
$$

该版鼓励身体水平、高度合适、瞬时速度接近目标，并沿路径前进。速度评分使用 $E_0$，前进增量系数为 20。代码见 [env.py 基础奖励](D:/Download/桌面/rl/purequad/env.py:128)。

**版本 2：加强平均速度跟踪与纠偏。**

$$
\boxed{
\begin{aligned}
r_t^{(2)}=C_t+\Delta t\Big[&6m_t\exp\left(-\frac{S}{0.02}\right)\\
&+4m_t\exp\left(-\left(\frac{h-0.31}{0.02}\right)^2\right)\\
&+10\exp\left(-\frac{E_{\mathrm{avg}}}{0.04^2}\right)\\
&-150E_{\mathrm{avg}}
-120e_y^2-10v_z^2
\Big]+2\Delta s
\end{aligned}}
$$

速度奖励改用一阶平均速度，并增加显式速度误差惩罚 $-150E_{\mathrm{avg}}\Delta t$ 和竖直速度惩罚 $-10v_z^2\Delta t$。前进增量系数降为 2，减少单纯追求前进距离的激励。代码见 [env.py 版本 2 分支](D:/Download/桌面/rl/purequad/env.py:141)。

**版本 3：加强机身稳定性。**

$$
\boxed{
\begin{aligned}
r_t^{(3)}=C_t+\Delta t\Big[&8m_t\exp\left(-\frac{S}{0.002}\right)\\
&+5m_t\exp\left(-\left(\frac{h-0.31}{0.008}\right)^2\right)\\
&+10\exp\left(-\frac{E_{\mathrm{avg}}}{0.04^2}\right)\\
&-150E_{\mathrm{avg}}
-120e_y^2-10v_z^2\\
&+2\exp\left(-\left(\frac{v_z}{0.015}\right)^2\right)
\Big]+2\Delta s
\end{aligned}}
$$

直立奖励峰值从 6 增至 8，姿态误差增大后奖励衰减更快；高度奖励宽度从 20 mm 收紧至 8 mm，并增加竖直速度接近零的正奖励。指数奖励的宽度决定评分衰减速度，本身不构成终止阈值。代码见 [env.py 版本 3 分支](D:/Download/桌面/rl/purequad/env.py:148)。

**版本 4：增加关节与机身平滑性惩罚。**

$$
\boxed{
\begin{aligned}
r_t^{(4)}=C_t+\Delta t\Big[&8m_t\exp\left(-\frac{S}{0.002}\right)\\
&+6m_t\exp\left(-\left(\frac{h-0.31}{0.006}\right)^2\right)\\
&+10\exp\left(-\frac{E_{\mathrm{avg}}}{0.04^2}\right)\\
&-150E_{\mathrm{avg}}
-120e_y^2-40v_z^2\\
&+2\exp\left(-\left(\frac{v_z}{0.015}\right)^2\right)\\
&-0.0025\operatorname{mean}(\ddot q^2)\\
&-0.1\operatorname{mean}(\dot u_t^2)
-2(\omega_1^2+\omega_2^2)
\Big]+2\Delta s
\end{aligned}}
$$

新增的后三项分别抑制关节速度的剧烈变化、相邻动作的剧烈变化，以及机身横滚、俯仰方向的快速转动。高度奖励宽度收紧至 6 mm，竖直速度惩罚系数增至 40。其中：

$$
\ddot q=\frac{\dot q_t-\dot q_{t-1}}{\Delta t},\qquad
\dot u_t=\frac{u_t-u_{t-1}}{\Delta t}
$$

$\omega_1,\omega_2$ 直接对应 `qvel[3:5]` 两个分量。`torque_rate` 奖励实际使用 $\dot u_t$，不是按 N·m 计算的实际力矩变化率。代码见 [env.py 版本 4 分支](D:/Download/桌面/rl/purequad/env.py:154)。

**版本 5：针对当前 URDF 调整平滑惩罚强度。**

$$
\boxed{
\begin{aligned}
r_t^{(5)}=C_t+\Delta t\Big[&8m_t\exp\left(-\frac{S}{0.002}\right)\\
&+6m_t\exp\left(-\left(\frac{h-0.31}{0.006}\right)^2\right)\\
&+10\exp\left(-\frac{E_{\mathrm{avg}}}{0.04^2}\right)\\
&-150E_{\mathrm{avg}}
-120e_y^2-40v_z^2\\
&+2\exp\left(-\left(\frac{v_z}{0.015}\right)^2\right)\\
&-10^{-6}\operatorname{mean}(\ddot q^2)\\
&-10^{-4}\operatorname{mean}(\dot u_t^2)
-2(\omega_1^2+\omega_2^2)
\Big]+2\Delta s
\end{aligned}}
$$

版本 5 与版本 4 只有两项系数不同，其余奖励保持相同：

| 惩罚项 | 版本 4 的系数 | 版本 5 最终系数 | 降低比例 |
| --- | ---: | ---: | ---: |
| $\operatorname{mean}(\ddot q^2)\Delta t$ | 0.0025 | $10^{-6}$ | 2500 倍 |
| $\operatorname{mean}(\dot u_t^2)\Delta t$ | 0.1 | $10^{-4}$ | 1000 倍 |

源码注明这一调整针对当前提供的 URDF。控制周期只有 2 ms，差分求导再平方可能产生较大数值，惩罚过强可能压制正常抬腿和动作调整；版本 5 通过较小系数调整这一权衡。它保留机身转动速度、高度和竖直速度要求，也保留两项差分惩罚。代码见 [env.py 版本 5 分支](D:/Download/桌面/rl/purequad/env.py:161)。

这些系数是代码设置的奖励参数，训练过程中不会因选择版本 5 而自动学习新的奖励权重。

**站立与基础运动任务怎样对应上述公式。**

| 环境任务 | 实际使用规则 |
| --- | --- |
| `path` | 使用上面对应版本的公式；满足终点条件且未失败时，终止奖励为 +100 |
| `motion` | 版本 1 的横向误差项为 0；版本 2～5 仍使用横向误差惩罚；不启用到达终点的成功判定，因此没有成功终止奖励 |
| `stand` | 使用基础站立奖励：$m_t=1$、目标速度为 0、瞬时速度正奖励系数为 2；航向、横向和前进奖励为 0；跳过版本 2～5 的全部运动调整分支 |

因此，站立任务即使设置较高的奖励版本编号，也仍使用基础站立奖励，不能直接将上面某个路径总公式代入站立任务。不同任务均有失败终止惩罚，具体结束判断见 4.4 节。

上式均为环境返回的单步奖励。大多数连续项乘以 $\Delta t$；前进项使用距离增量，动作突变和一次性终止项没有再乘 $\Delta t$。CEM 在回合结束后计算的抬脚、腾空比例和前进量等约束扣分另外计入候选评分，详见 5.1 节。

'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def norm(data):
    return data.decode('utf-8-sig').replace('\r\n', '\n')


def prepare():
    original = TARGET.read_bytes()
    before = norm(original)
    assert before.count(START) == before.count(END) == 1
    start, end = before.index(START), before.index(END)
    after = before[:start] + SECTION + before[end:]
    # Repair the missing opening backtick in this reward section only.
    typo = '\ntorque` 与 `torque_rate` 的名字需要结合公式理解：'
    if typo in after:
        assert after.count(typo) == 1
        after = after.replace(typo, '\n`torque` 与 `torque_rate` 的名字需要结合公式理解：', 1)
    sources = {}
    for name in ['purequad/env.py', 'purequad/config.py', 'purequad/train.py', 'purequad/smooth_train.py', 'new_train.py']:
        path = PROJECT / name
        content = path.read_bytes()
        sources[str(path)] = sha(content)
        ast.parse(content.decode('utf-8-sig'))
    env = (PROJECT / 'purequad/env.py').read_text(encoding='utf-8')
    for fragment in ["'progress':20*delta", "terms['progress']=2*delta", "terms['speed_error']=-150*error", "terms['height']=5*np.exp", "terms['height']=6*np.exp", "terms['joint_acceleration']=-.0025", "terms['joint_acceleration']=-1e-6", "terms['torque_rate']=-.1", "terms['torque_rate']=-1e-4", "terms['body_rotation_rate']=-2", "'smooth':-.02*float(np.mean((self.action-previous)**2))"]:
        assert fragment in env, fragment
    policy = PROJECT / 'runs/mini_lite_urdf/final/policy.zip'
    with ZipFile(policy) as archive:
        cfg = json.loads(archive.read('data'))['pure_config']
    expected = {'stage':'path', 'reward_version':5, 'reference_reward':False, 'physics_dt':.002, 'frame_skip':1, 'speed':.13, 'height':.31, 'action_scale':.4, 'fall_penalty':50.}
    assert all(cfg[k] == v for k, v in expected.items())
    prefix = before[:start].replace(typo, '\n`torque` 与 `torque_rate` 的名字需要结合公式理解：')
    assert after[:after.index(START)] == prefix
    assert after[after.index(END):] == before[end:]
    assert re.findall(r'!\[\[.*?\]\]', after) == re.findall(r'!\[\[.*?\]\]', before)
    assert after.count('```') % 2 == 0
    assert sum(line.strip() == '$$' for line in after.splitlines()) % 2 == 0
    for version in range(1, 6):
        assert SECTION.count(f'r_t^{{({version})}}=C_t') == 1
    assert SECTION.count('\\boxed{') == 5
    for display in re.findall(r'^\$\$\s*\n(.*?)^\$\$', SECTION, flags=re.MULTILINE | re.DOTALL):
        assert display.count('\\begin{aligned}') == display.count('\\end{aligned}')
        assert display.count('\\begin{cases}') == display.count('\\end{cases}')
        depth = 0
        for char in display:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            assert depth >= 0
        assert depth == 0
    for block in re.findall(r'(?:^\|.*\n)+', SECTION, flags=re.MULTILINE):
        assert len({line.count('|') for line in block.splitlines()}) == 1
    for link in re.findall(r'\]\((D:/[^)]+)\)', after):
        assert Path(re.sub(r':\d+$', '', link)).exists(), link
    for embed in re.findall(r'!\[\[(.*?)\]\]', after):
        assert (Path('E:/obsidian/lqy的知识库') / embed).is_file(), embed
    newline = '\r\n' if b'\r\n' in original else '\n'
    updated = after.replace('\n', newline).encode('utf-8')
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'overview.before.md').write_bytes(original)
    (HERE / 'overview.updated.md').write_bytes(updated)
    (HERE / 'overview.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='overview.before.md', tofile='overview.updated.md')), encoding='utf-8')
    plan = {'target':str(TARGET), 'before_normalized_sha256':sha(before.encode('utf-8')),
            'updated_sha256':sha(updated), 'updated_normalized_sha256':sha(after.encode('utf-8')),
            'sources':sources, 'policy_sha256':sha(policy.read_bytes()),
            'checks':{'five_complete_formula_combinations':True, 'formula_braces_and_environments_balanced':True,
                      'remaining_note_and_figures_preserved':True, 'project_links_and_embeds_exist':True,
                      'new_tables_consistent':True, 'source_and_policy_configuration_checked':True},
            'before_lines':len(before.splitlines()), 'after_lines':len(after.splitlines())}
    (HERE / 'publication_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:plan[k] for k in ['target','before_lines','after_lines','checks']}, ensure_ascii=False, indent=2))


def publish():
    plan = json.loads((HERE / 'publication_plan.json').read_text(encoding='utf-8'))
    assert sha(norm(TARGET.read_bytes()).encode('utf-8')) == plan['before_normalized_sha256'], 'Note changed after preparation'
    for path, expected in plan['sources'].items():
        assert sha(Path(path).read_bytes()) == expected, f'Source changed: {path}'
    assert sha((PROJECT / 'runs/mini_lite_urdf/final/policy.zip').read_bytes()) == plan['policy_sha256']
    updated = (HERE / 'overview.updated.md').read_bytes()
    assert sha(updated) == plan['updated_sha256']
    TARGET.write_bytes(updated)
    observed = TARGET.read_bytes()
    assert sha(norm(observed).encode('utf-8')) == plan['updated_normalized_sha256']
    result = {'written':True, 'verified':True, 'target':str(TARGET), 'sha256':sha(observed)}
    (HERE / 'publication_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    publish() if args.publish else prepare()
