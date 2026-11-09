# runs/dp2_expandable rank 0

trace entries: 8730

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
| step0 mb0 | 0 | 0 | 6534 | 6534 | 6582 |
| step0 mb1 | 0 | 0 | 6552 | 6552 | 10026 |
| step0 mb2 | 0 | 0 | 6552 | 6552 | 10526 |
| step1 mb0 | 0 | 0 | 10908 | 10908 | 10946 |
| step1 mb1 | 0 | 0 | 10908 | 10908 | 14386 |
| step1 mb2 | 0 | 0 | 10908 | 10908 | 14386 |
| step2 mb0 | 0 | 0 | 10908 | 10908 | 14386 |
| step2 mb1 | 0 | 0 | 10908 | 10908 | 14386 |
| step2 mb2 | 0 | 0 | 10908 | 10908 | 14386 |
