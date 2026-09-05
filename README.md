# ComfyUI-ki-MiniMax-H3-Ref — ki MiniMax H3 Reference to Video

ComfyUI 官方 `MiniMaxH3ReferenceToVideo` 节点的**魔改增强版**：行为 100% 兼容官方，唯一区别是把参考图缩放选项 `ref_image_size` 从官方的两档（`match` / `max`）扩展为 **22 档连续可控的短边缩放**，让你在「生成速度」与「参考图身份保真度」之间精细调节。

节点 ID：`ki_MiniMaxH3ReferenceToVideo`（`ki_` 前缀，与官方内置节点**不冲突**，可同时存在）

## 魔改核心算法：短边比例缩放

官方只有两个极端：`match`（把参考图缩到和生成画布面积相当，最快但细节丢失多）和 `max`（短边固定 2048，保真最高但每步 token 开销最大）。

本插件在两者之间插入 1.2 ~ 3.1 共 20 档中间值，算法与官方 `max` 同族：

```
gen_short = min(生成宽, 生成高)          # 生成画布短边
target    = gen_short × (ratio / 1.2)   # 1.2 档恰好等于生成短边（基线）
target    = round(target / 32) × 32     # 对齐 32 的倍数
target    = min(target, 2048)           # 不超过官方上限
scale     = min(1.0, target / min(原宽, 原高))  # 只缩小、绝不放大
```

以 1.0MP 生成（1376×768，gen_short=768）为例：

| 选项 | 算法 | 目标短边 |
|--------|-----------|------|
| `match` | 面积匹配生成画布 | ≈ 输出面积 |
| `1.2` | gen_short×(1.2/1.2) | 768 |
| `1.3` | gen_short×(1.3/1.2) | 832 |
| `1.4` | gen_short×(1.4/1.2) | 896 |
| `1.5` | gen_short×(1.5/1.2) | 960 |
| `1.6` | gen_short×(1.6/1.2)（默认） | 1024 |
| `1.7` | gen_short×(1.7/1.2) | 1088 |
| `1.8` | gen_short×(1.8/1.2) | 1152 |
| `1.9` | gen_short×(1.9/1.2) | 1216 |
| `2.0` | gen_short×(2.0/1.2) | 1280 |
| `2.1` | gen_short×(2.1/1.2) | 1344 |
| `2.2` | gen_short×(2.2/1.2) | 1408 |
| `2.3` | gen_short×(2.3/1.2) | 1472 |
| `2.4` | gen_short×(2.4/1.2) | 1536 |
| `2.5` | gen_short×(2.5/1.2) | 1600 |
| `2.6` | gen_short×(2.6/1.2) | 1664 |
| `2.7` | gen_short×(2.7/1.2) | 1728 |
| `2.8` | gen_short×(2.8/1.2) | 1792 |
| `2.9` | gen_short×(2.9/1.2) | 1856 |
| `3.0` | gen_short×(3.0/1.2) | 1920 |
| `3.1` | gen_short×(3.1/1.2) | 1984 |
| `max` | 官方固定值 | 2048 |

**ratio 越大 = 参考图身份细节越保真，但每步去噪的视觉 token 越多、速度越慢。**

## 节点功能（与官方一致）

多模态参考生视频条件节点，支持在提示词中用 `<Picture i>` / `<Video k>` / `<Audio j>` 标签引用参考素材：

- **参考图**（最多 9 张，Autogrow 槽位）：按上述算法短边缩放、32 对齐后 VAE 编码为 latent 块注入 DiT
- **参考视频**（最多 3 个，2–15s @ 24fps）：`adapt_canvas` 自适应画布、帧数对齐到 `17k+5`、Qwen 视觉端按 2fps 抽帧并附时间戳；可索引配对**原声轨**（`ref_video_audio_N` 属于 `ref_video_N`，会先发 `<Audio j>` 标签再发 `<Video k>`）
- **独立参考音频**（最多 3 条）：经 audio VAE 编码进条件
- 输出 `positive` 条件 + 空 AV latent，直接接 KSampler

## 实现方式

不复制官方逻辑，而是直接 `from comfy_extras.nodes_minimax_h3 import ...` 复用官方的 `adapt_canvas`、`_resize`、`_encode_ref_audio`、`_empty_av_latent` 等辅助函数与常量，仅重写 `_ref_image_scale` 缩放决策。官方行为升级时自动跟随；若 ComfyUI 版本不含 MiniMax H3 支持会启动时报错提示更新。

## 安装

1. 将本文件夹放入 `ComfyUI/custom_nodes/`
2. 重启 ComfyUI
3. 搜索节点「ki MiniMax H3 Reference to Video」（分类：`model/conditioning/minimax`）

## 文件结构

```
ComfyUI-ki-MiniMax-H3-Ref/
├── __init__.py   # 节点注册
└── nodes.py      # ki 节点 + 短边比例缩放算法
```

## License

MIT
