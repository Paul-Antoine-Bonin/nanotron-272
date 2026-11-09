# runs/tp2 rank 0

trace entries: 9482

| backward | grads | MB | freed in _accumulate_grad | reusable before next forward | late MB | where late ones become reusable |
|---|---|---|---|---|---|---|
| step0 mb0 | 26 | 545 | 26/26 | 26/26 | 0 | - |
| step0 mb1 | 26 | 545 | 26/26 | 26/26 | 0 | - |
| step0 mb2 | 26 | 545 | 26/26 | 26/26 | 0 | - |
| step1 mb0 | 26 | 545 | 26/26 | 26/26 | 0 | - |
| step1 mb1 | 26 | 545 | 26/26 | 26/26 | 0 | - |
| step1 mb2 | 26 | 545 | 26/26 | 26/26 | 0 | - |
| step2 mb0 | 26 | 545 | 26/26 | 26/26 | 0 | - |
| step2 mb1 | 26 | 545 | 26/26 | 26/26 | 0 | - |
| step2 mb2 | 26 | 545 | 26/26 | 26/26 | 0 | - |

## Python side at each forward start

| forward | prev grads still alive | MB | allocated MB | active MB | reserved MB |
|---|---|---|---|---|---|
| step0 mb0 | 0 | 0 | 2731 | 2731 | 2738 |
| step0 mb1 | 0 | 0 | 2764 | 2764 | 4502 |
| step0 mb2 | 0 | 0 | 2764 | 2921 | 4502 |
| step1 mb0 | 0 | 0 | 4947 | 4947 | 5048 |
| step1 mb1 | 0 | 0 | 4947 | 5102 | 6714 |
| step1 mb2 | 0 | 0 | 4947 | 5101 | 7216 |
| step2 mb0 | 0 | 0 | 4947 | 4947 | 7216 |
| step2 mb1 | 0 | 0 | 4947 | 5103 | 7216 |
| step2 mb2 | 0 | 0 | 4947 | 5103 | 7216 |
