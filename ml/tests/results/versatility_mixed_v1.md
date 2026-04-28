# Versatility evaluation — mixed_v1

- **Model**: `models/versatility_mixed_v1.pth`
- **Seed**: 42 · **test_ratio**: 0.2
- **Trained on 691 zip(s)**
- **similar**: 3 pattern(s) — ['data/zips/*_spider.zip', 'data/zips/*_tiled.zip', 'data/zips/*_twitch_*.zip']
- **out_of_box**: 1 pattern(s) — ['data/zips/*_unknown.zip']

## trained

| bucket | tester | num_samples | overall_acc | idle_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|---|
| trained | chrome | 34 | 1.0000 | N/A | 1.0000 | N/A |
| trained | hearthstone | 160 | 1.0000 | N/A | N/A | 1.0000 |
| trained | miro | 365 | 1.0000 | N/A | 1.0000 | N/A |
| trained | rocket_league | 227 | 1.0000 | N/A | N/A | 1.0000 |
| trained | valorant | 330 | 1.0000 | 1.0000 | N/A | 1.0000 |
| trained | wuthering_waves | 264 | 1.0000 | N/A | N/A | 1.0000 |
| trained | ALL | 1380 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## similar

| bucket | tester | num_samples | overall_acc | idle_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|---|
| similar | spider | 250 | 0.8000 | N/A | N/A | 0.8000 |
| similar | tiled | 238 | 1.0000 | N/A | 1.0000 | N/A |
| similar | twitch_(cs2) | 87 | 0.0345 | N/A | 0.0345 | N/A |
| similar | ALL | 575 | 0.7670 | N/A | 0.7415 | 0.8000 |

## out_of_box

| bucket | tester | num_samples | overall_acc | idle_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|---|
| out_of_box | unknown | 100 | 0.9600 | N/A | N/A | 0.9600 |
| out_of_box | ALL | 100 | 0.9600 | N/A | N/A | 0.9600 |

