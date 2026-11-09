# runs/single_afab rank 0

trace entries: 8696

| backward | grads | MB | freed in _accumulate_grad | reusable before next forward | late MB | where late ones become reusable |
|---|---|---|---|---|---|---|
| step0 mb0 | 26 | 1089 | 26/26 | 26/26 | 0 | - |
| step0 mb1 | 26 | 1089 | 26/26 | 26/26 | 0 | - |
| step0 mb2 | 26 | 1089 | 26/26 | 26/26 | 0 | - |
| step1 mb0 | 26 | 1089 | 26/26 | 26/26 | 0 | - |
| step1 mb1 | 26 | 1089 | 26/26 | 26/26 | 0 | - |
| step1 mb2 | 26 | 1089 | 26/26 | 26/26 | 0 | - |
| step2 mb0 | 26 | 1089 | 26/26 | 26/26 | 0 | - |
| step2 mb1 | 26 | 1089 | 26/26 | 26/26 | 0 | - |
| step2 mb2 | 26 | 1089 | 26/26 | 26/26 | 0 | - |

## Python side at each forward start

| forward | prev grads still alive | MB | allocated MB | active MB | reserved MB |
|---|---|---|---|---|---|
| step0 mb0 | 0 | 0 | 5450 | 5450 | 5474 |
| step0 mb0 | 0 | 0 | 7404 | 7404 | 8972 |
| step0 mb0 | 0 | 0 | 9343 | 9343 | 10976 |
| step1 mb0 | 0 | 0 | 9824 | 9824 | 12980 |
| step1 mb0 | 0 | 0 | 11764 | 11764 | 12980 |
| step1 mb0 | 0 | 0 | 13703 | 13703 | 15486 |
| step2 mb0 | 0 | 0 | 9824 | 9824 | 17490 |
| step2 mb0 | 0 | 0 | 11764 | 11764 | 17490 |
| step2 mb0 | 0 | 0 | 13703 | 13703 | 17490 |
