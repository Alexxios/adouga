# Versatility evaluation — games_only_v1

- **Model**: `models/versatility_games_only_v1.pth`
- **Seed**: 42 · **test_ratio**: 0.2
- **Trained on 271 zip(s)**
- **similar**: 1 pattern(s) — ['data/zips/*_chrome.zip']
- **out_of_box**: 1 pattern(s) — ['data/zips/*_rocket_league.zip']

## trained

| bucket | tester | num_samples | overall_acc | idle_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|---|
| trained | hearthstone | 126 | 1.0000 | N/A | N/A | 1.0000 |
| trained | miro | 372 | 1.0000 | N/A | 1.0000 | N/A |
| trained | valorant | 42 | 1.0000 | 1.0000 | N/A | N/A |
| trained | ALL | 540 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## similar

| bucket | tester | num_samples | overall_acc | idle_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|---|
| similar | chrome | 190 | 0.2000 | N/A | 0.2000 | N/A |
| similar | ALL | 190 | 0.2000 | N/A | 0.2000 | N/A |

## out_of_box

| bucket | tester | num_samples | overall_acc | idle_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|---|
| out_of_box | rocket_league | 390 | 0.0000 | N/A | N/A | 0.0000 |
| out_of_box | ALL | 390 | 0.0000 | N/A | N/A | 0.0000 |

