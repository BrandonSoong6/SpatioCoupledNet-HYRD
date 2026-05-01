# 🧠 Robot Brain Memory Tool (记忆维护工具)

`Memory Tool` 是 `robot_brain` 终身学习系统的配套维护脚本。它用于安全地管理机器人的“记忆”（训练数据和索引），确保 `memory_index.json` 与物理文件系统始终保持同步。

使用此工具可以避免手动删除文件夹导致的索引报错，或因环境变化引发的模型训练崩溃。

## 📂 脚本位置

```plaintext
~/brandon/hyrd_robot/src/robot_brain/scripts/memory_tool.py

```

## 🚀 快速开始

在使用之前，建议赋予脚本执行权限：

```bash
chmod +x ~/brandon/hyrd_robot/src/robot_brain/scripts/memory_tool.py

```

---

## 🛠️ 功能与用法

### 1. 查看记忆状态 (`--status`)

查看当前经验池中有多少个 Batch，有多少精英样本，以及当前的平均 Loss 基准。

```bash
python3 memory_tool.py --status

```

### 2. 删除指定 Batch (`--del_batch`)

* **场景**：某一轮次（例如 `batch_012`）因为硬件故障（相机遮挡、电机断电、撞到障碍物）产生了脏数据，需要将其剔除。
* **操作**：
```bash
python3 memory_tool.py --del_batch batch_012

```


* **效果**：
* 从 `memory_index.json` 的 `history` 和 `elites` 列表中移除该 ID。
* 彻底删除 `lifelong_data/batch_012` 文件夹。



### 3. 切换环境 / 重置短期记忆 (`--new_place`)

* **场景**：机器人从“实验室”移动到了“走廊”，或负载发生变化。旧的短期记忆（Short-term Memory）不再适用，但你想保留已训练好的模型权重和精英样本。
* **操作**：
```bash
python3 memory_tool.py --new_place

```


* **效果**：
* 清空 `memory_index.json` 中的 `history`（短期滑动窗口）。
* **保留** `elites`（精英/困难样本）。
* **保留** 所有物理文件（不删除文件夹，仅让训练器忽略它们）。
* 下次运行 Orchestrator 时，将触发“冷启动保护”，重新积累适应新环境的数据。



### 4. 恢复出厂设置 (`--nuke`)

* **场景**：模型彻底练废，或想从 Loop 0 开始全新的实验。
* **操作**：
```bash
python3 memory_tool.py --nuke

```


* **⚠️ 警告**：此操作不可逆！
* **效果**：
* 彻底删除整个 `lifelong_data` 目录。
* 删除所有数据、模型 checkpoints 和索引。
* 下次运行 Orchestrator 时，系统将从 `Loop 000` 重新初始化。



---

## ❓ 常见问题 (FAQ)

> **Q: 运行脚本时需要关闭 Orchestrator 吗？**
> **A:** **必须关闭**。为了防止文件读写冲突，请先在运行 Orchestrator 的终端按 `Ctrl+C` 停止程序，然后再运行此维护工具。

> **Q: `--new_place` 会删除我的模型吗？**
> **A:** **不会**。它只清除“用于训练的数据索引”。模型权重文件（`.pth`）会保留，机器人将在新的一轮训练中基于原有模型进行微调（Fine-tuning）。

> **Q: 我可以直接在文件管理器里删文件夹吗？**
> **A:** **不建议**。如果你手动删除了 `batch_xxx` 文件夹但没更新 JSON 索引，Orchestrator 会在加载数据时因找不到文件而报错崩溃。请始终使用此工具进行操作。
