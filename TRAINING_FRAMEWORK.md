# EEG 独立片段训练与结果框架

框架读取现有 `data/eeg/brainlat/*.npy` 与
`dataset_description.json`，动态加载
`model_design/eeg_hc_ad_model.py`。训练输入已经与当前模型保持一致：一个 batch
是形状为 `[N, C, T]` 的独立片段集合，不包含被试 ID，也没有同被试 bag 或片段
间聚合。

`data/` 和 `model_design/` 只会被读取；框架不会修改其中内容。

## 执行方式

所有命令均使用 `conda` 环境 `cgz`：

```bash
# 1. 只读检查数据、划分、图、模型和损失配置
conda run -n cgz python train_eeg.py --check

# 2. 在固定的小规模真实片段集合上做过拟合诊断
conda run -n cgz python train_eeg.py --sanity-overfit

# 3. 正式训练；默认依次使用 2026–2035 共 10 个数据划分种子
conda run -n cgz python train_eeg.py

# 覆盖随机数种子的起点和数量
conda run -n cgz python train_eeg.py \
  --split-seed-start 2030 --split-seed-count 5
```

可以用 `--run-name` 和 `--device cuda:0` 覆盖实验名与设备。正式训练会针对每个
数据划分分别重新初始化模型、训练和测试，最后汇总所有数值指标。SSH 后台运行
方式：

```bash
./run_eeg_background.sh
./run_eeg_background.sh start --split-seed-start 2030 --split-seed-count 5
./run_eeg_background.sh sanity
./run_eeg_background.sh status
./run_eeg_background.sh tail
```

## 数据划分与片段输入

数据仍先按被试划分，以阻止同一被试的片段跨越训练、验证和测试集。划分按
“诊断 × 机构”联合分层。默认从 `split_seed_start=2026` 开始使用 10 个连续
种子，即 2026–2035；每个种子产生一套独立的训练、验证和测试划分，并完成一次
完整训练。下表是种子 2026 的结果：

| split | 被试数 | 片段数 |
|---|---:|---:|
| train | 46 | 40,235 |
| validation | 11 | 9,629 |
| test | 10 | 9,674 |

每个 split 内随后转换为扁平片段索引：

1. 每个数据集条目只对应一个 `[C,T]` 片段；
2. 每轮将所有条目作一次确定性全局重排，不按被试连续组织；
3. DataLoader 直接堆叠成 `[N,C,T]`，只把 EEG 张量传给模型；
4. 标签是逐片段标签，分类损失直接作用于 `segment_logits`；
5. 每个 epoch 使用训练集全部 40,235 个片段，无重复、无遗漏；
6. 类别权重按 HC/AD 的片段数计算，不再按被试或 bag 加权。

被试 ID、机构和原始片段序号仅作为 batch 外元数据保留，用于划分审计以及预测
完成后的被试级汇总，不参与前向传播。

## 训练、验证与选模

- Epoch 1–5：仅优化片段分类损失；
- Epoch 6–10：神经生理辅助损失权重从 0.02 线性升至 0.10；
- Epoch 11 以后：辅助损失权重保持 0.10；
- 没有质量标签，因此质量监督权重固定为 0；
- 学习率调度监控验证集片段交叉熵；
- 默认最佳 checkpoint 按“被试内 logits 平均”的 fixed-0.5 balanced accuracy
  选择，并列时使用片段交叉熵。

每轮都会同时计算片段级、被试多数投票和被试 logits 平均三组指标，但后两组只
用于评估/选模，不进入训练图，也不会把同一被试的片段联合送入模型。

多划分实验开始时会先冻结输入配置、模型源码、预处理 JSON 和训练源码。之后的
所有子实验都从这份父实验快照派生；即使长时间训练期间这些配置或源码在工作区
发生变化，也不会让不同划分使用不同版本的配置或代码。

## 三类预测结果

最佳 checkpoint 会在验证集和测试集各输出三张表：

1. `*_segments.csv`：每个片段一行，含被试 ID、片段序号、真实标签、预测标签、
   HC/AD 概率与两个 logits；
2. `*_subject_majority_vote.csv`：每个被试一行，先按 0.5 将各片段变成硬标签，
   再取多数票；表中保留 HC/AD 票数、票差和片段数。票数完全相同时固定判为
   AD，并通过 `vote_tie=1` 明确标记；
3. `*_subject_logit_mean.csv`：每个被试一行，分别平均其所有片段的 HC/AD
   logits，再做 softmax 和分类。

框架针对上述三类结果分别在验证集选择最大化 balanced accuracy 的阈值，并把
对应阈值冻结后用于测试集。CSV 同时保留固定 0.5 的结果和验证集阈值结果；
每个子实验的 `metrics/final_metrics.json` 同时报告两套指标。父实验会对所有
子实验中的数值指标计算均值、样本标准差、最小值、最大值和有效样本数，字段分别
为 `mean`、`std`、`min`、`max` 和 `n`。

## 结果目录

```text
exp/YYYYMMDD_HHMMSS_eeg_hc_ad_independent_segments_multi_split/
├── config/
├── snapshots/
│   ├── data_json/                         # 整组实验冻结的预处理 JSON
│   ├── model_source/                      # 整组实验冻结的模型源码
│   ├── training_code/                     # 整组实验冻结的训练源码
│   └── manifest.json
├── metrics/
│   ├── aggregate_final_metrics.json       # 跨划分综合指标
│   ├── aggregate_run_metrics.json
│   └── split_summaries.json               # 各子实验完整 summary
├── logs/train.log
├── status.json
├── summary.json
├── split_seed_2026/
│   ├── checkpoints/best.pt
│   ├── checkpoints/last.pt
│   ├── metrics/history.csv
│   ├── metrics/history.jsonl
│   ├── metrics/coverage.jsonl             # 全量覆盖、顺序哈希、重复/缺失数
│   ├── metrics/final_metrics.json
│   ├── predictions/
│   ├── artifacts/
│   ├── splits.json
│   └── summary.json
├── split_seed_2027/
│   └── ...
└── ...
```

每个 `split_seed_*/metrics/coverage.jsonl` 中的
`subject_grouped_batches=false`、`coverage_ratio=1.0`、
`duplicate_segments=0` 和 `missing_segments=0` 可用于审计片段级输入策略。

## 显存调整

默认 `batch_size=64`、`eval_batch_size=64`；当前 RTX 4090 上的真实 AMP
前向、辅助损失和反向烟雾测试峰值约为 16.99 GiB。若设备显存不足，可以减小
batch，并增加梯度累积步数以保持近似的有效 batch：

```json
{
  "data": {
    "batch_size": 32,
    "eval_batch_size": 32
  },
  "training": {
    "gradient_accumulation_steps": 2
  }
}
```

默认配置位于 `configs/eeg_hc_ad.json`。
