# runs/dp2 rank 0

trace entries: 8727

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
| step0 mb0 | 0 | 0 | 6540 | 6540 | 6564 |
| step0 mb1 | 0 | 0 | 6557 | 6557 | 10062 |
| step0 mb2 | 0 | 0 | 6557 | 6557 | 11064 |
| step1 mb0 | 0 | 0 | 10914 | 10914 | 11408 |
| step1 mb1 | 0 | 0 | 10914 | 10914 | 14466 |
| step1 mb2 | 0 | 0 | 10914 | 10914 | 14466 |
| step2 mb0 | 0 | 0 | 10914 | 10914 | 14466 |
| step2 mb1 | 0 | 0 | 10914 | 10914 | 14466 |
| step2 mb2 | 0 | 0 | 10914 | 10914 | 14466 |
