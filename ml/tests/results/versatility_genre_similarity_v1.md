# Versatility evaluation — genre_similarity_v1

- **Model**: `models/versatility_genre_similarity_v1.pth`
- **Seed**: 42 · **test_ratio**: 0.2
- **Trained on 612 zip(s)**
- **similar**: 3 pattern(s) — ['data/zips/*_valorant.zip', 'data/zips/*_riotclient.zip', 'data/zips/*_twitchcs2.zip']
- **out_of_box**: 2 pattern(s) — ['data/zips/*_rocketleague.zip', 'data/zips/*_paint.zip']

## trained

| bucket | tester | num_samples | overall_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|
| trained | adobe | 26 | 1.0000 | 1.0000 | N/A |
| trained | apex | 104 | 1.0000 | N/A | 1.0000 |
| trained | cs2 | 149 | 1.0000 | N/A | 1.0000 |
| trained | hearthstone | 165 | 1.0000 | 1.0000 | 1.0000 |
| trained | pycharm | 82 | 1.0000 | 1.0000 | N/A |
| trained | steam | 106 | 1.0000 | 1.0000 | 1.0000 |
| trained | twitchwuwa | 127 | 1.0000 | 1.0000 | N/A |
| trained | wutheringwaves | 234 | 1.0000 | N/A | 1.0000 |
| trained | wutheringwavespinball | 227 | 1.0000 | N/A | 1.0000 |
| trained | ALL | 1220 | 1.0000 | 1.0000 | 1.0000 |

## similar

| bucket | tester | num_samples | overall_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|
| similar | riotclient | 310 | 0.6871 | 0.6871 | N/A |
| similar | twitchcs2 | 1640 | 1.0000 | 1.0000 | N/A |
| similar | valorant | 380 | 1.0000 | N/A | 1.0000 |
| similar | ALL | 2330 | 0.9584 | 0.9503 | 1.0000 |

## out_of_box

| bucket | tester | num_samples | overall_acc | not_gaming_acc | gaming_acc |
|---|---|---|---|---|---|
| out_of_box | paint | 167 | 1.0000 | 1.0000 | N/A |
| out_of_box | rocketleague | 1340 | 0.9985 | 1.0000 | 0.9985 |
| out_of_box | ALL | 1507 | 0.9987 | 1.0000 | 0.9985 |

