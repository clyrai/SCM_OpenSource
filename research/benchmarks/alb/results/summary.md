# ALB v0.1 pilot results

## SCM

### idle_on = True

| persona | pdr | css | cgc_id | cgc_fill | crai_current | crai_old | wsi_f1 |
|---|---|---|---|---|---|---|---|
| persona_001 | 0.500 | 0.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.222 |
| persona_002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |

### idle_on = False (NIAL ablation)

| persona | pdr | css | cgc_id | cgc_fill | crai_current | crai_old | wsi_f1 |
|---|---|---|---|---|---|---|---|
| persona_001 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| persona_002 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

### NIAL lift (idle_on - idle_off)

| metric | mean lift | n |
|---|---|---|
| pdr | +0.250 | 2 |
| css | -1.000 | 2 |
| cgc_id | +0.500 | 2 |
| cgc_fill | +0.250 | 2 |
| crai_current | +0.000 | 2 |
| crai_old | -1.000 | 2 |
| wsi_f1 | +0.361 | 2 |
