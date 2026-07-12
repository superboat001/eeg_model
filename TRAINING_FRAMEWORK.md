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

# 3. 正式训练
conda run -n cgz python train_eeg.py
```

可以用 `--run-name` 和 `--device cuda:0` 覆盖实验名与设备。SSH 后台运行方式：

```bash
./run_eeg_background.sh
./run_eeg_background.sh sanity
./run_eeg_background.sh status
./run_eeg_background.sh tail
```

## 数据划分与片段输入

数据仍先按被试划分，以阻止同一被试的片段跨越训练、验证和测试集。划分按
“诊断 × 机构”联合分层，默认结果为：

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
`metrics/final_metrics.json` 同时报告两套指标。

## 结果目录

```text
exp/YYYYMMDD_HHMMSS_eeg_hc_ad_independent_segments/
├── config/
├── snapshots/
│   ├── data_json/                 # 预处理 JSON 快照
│   ├── model_source/              # 本次实际加载的模型源码快照
│   ├── training_code/             # 训练框架快照
│   └── manifest.json
├── checkpoints/best.pt
├── checkpoints/last.pt
├── metrics/history.csv
├── metrics/history.jsonl
├── metrics/coverage.jsonl         # 全量覆盖、顺序哈希、重复/缺失数
├── metrics/final_metrics.json
├── predictions/
│   ├── validation_segments.csv
│   ├── validation_subject_majority_vote.csv
│   ├── validation_subject_logit_mean.csv
│   ├── test_segments.csv
│   ├── test_subject_majority_vote.csv
│   └── test_subject_logit_mean.csv
├── artifacts/channel_graph.json
├── artifacts/class_weights.json   # 按片段数计算
├── logs/train.log
├── environment.json
├── splits.json
├── status.json
└── summary.json
```

`coverage.jsonl` 中的 `subject_grouped_batches=false`、`coverage_ratio=1.0`、
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
