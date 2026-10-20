# 离线评测数据

把现场录制的视频放进 `videos/`，并为每段视频准备：

1. 标注 JSON（参考 [labels.example.json](labels.example.json)）
2. 可选的警戒区 JSON（与监控端 `zone_config.json` 同格式），用 `--zone` 传入

```bash
python scripts/evaluate.py --video eval/videos/a.mp4 --labels eval/a.json --zone eval/a_zone.json
python scripts/evaluate.py --video eval/videos/a.mp4 --labels eval/a.json \
    --trackers strongsort botsort deepocsort boosttrack --output reports/a.json
```

建议至少覆盖：遮挡、并排行走、边界徘徊、长时间逗留。报告里的 `extra_ids_in_events` 大于 0 说明同一事件内发生了 ID 断裂。
