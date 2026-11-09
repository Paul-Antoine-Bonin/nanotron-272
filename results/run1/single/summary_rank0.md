# runs/single rank 0

trace entries: 8704

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
| step0 mb1 | 0 | 0 | 5467 | 5467 | 8972 |
| step0 mb2 | 0 | 0 | 5467 | 5467 | 9974 |
| step1 mb0 | 0 | 0 | 9824 | 9824 | 10318 |
| step1 mb1 | 0 | 0 | 9824 | 9824 | 13360 |
| step1 mb2 | 0 | 0 | 9824 | 9824 | 13360 |
| step2 mb0 | 0 | 0 | 9824 | 9824 | 13360 |
| step2 mb1 | 0 | 0 | 9824 | 9824 | 13360 |
| step2 mb2 | 0 | 0 | 9824 | 9824 | 13360 |
