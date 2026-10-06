# Evaluation

Lines the demo never uses (seeds 21, 31, 41). Training as in the demo; each line scored on a fresh
3,000-frame stream with 10% defects. Thresholds set for 1% false rejects on held-back good parts. Reproduce with `python -m qi.evaluate`.

## Detection, against the simulation truth

| Product | Seed | AUROC model / template | False rejects model / template | Escapes model / template | Escapes by type (model) | Type right, of caught | Box on the defect |
|---|---|---|---|---|---|---|---|
| pcb | 21 | 0.941 / 0.791 | 1.4% / 1.1% | 20.6% / 56.6% | missing component 0/106, solder bridge 4/110, scratch 61/100 | 225/251 | 251/251 |
| pcb | 31 | 0.938 / 0.772 | 2.8% / 0.3% | 19.2% / 65.5% | missing component 0/89, solder bridge 7/121, scratch 53/103 | 213/253 | 248/253 |
| pcb | 41 | 0.930 / 0.801 | 1.6% / 0.6% | 20.5% / 63.3% | missing component 0/90, solder bridge 2/94, scratch 55/94 | 191/221 | 220/221 |
| pouch | 21 | 0.938 / 0.823 | 1.3% / 2.5% | 32.3% / 50.6% | seal gap 0/106, contamination 20/110, wrinkle 82/100 | 203/214 | 213/214 |
| pouch | 31 | 0.917 / 0.811 | 0.3% / 0.9% | 52.4% / 60.1% | seal gap 0/89, contamination 61/121, wrinkle 103/103 | 146/149 | 149/149 |
| pouch | 41 | 0.910 / 0.808 | 1.8% / 0.9% | 36.0% / 57.2% | seal gap 0/90, contamination 16/94, wrinkle 84/94 | 166/178 | 176/178 |
| bracket | 21 | 0.879 / 0.790 | 1.3% / 0.7% | 38.3% / 64.6% | scratch 76/106, dent 45/110, missing hole 0/100 | 177/195 | 194/195 |
| bracket | 31 | 0.877 / 0.794 | 0.9% / 2.0% | 42.2% / 63.6% | scratch 66/89, dent 66/121, missing hole 0/103 | 168/181 | 180/181 |
| bracket | 41 | 0.890 / 0.797 | 0.6% / 0.6% | 36.3% / 63.7% | scratch 67/90, dent 34/94, missing hole 0/94 | 161/177 | 176/177 |

Single-frame inspection on one core, 2,700 frames: p50 0.69 ms, p95 7.27 ms, p99 7.77 ms against the 33.3 ms a 30 FPS frame allows.

## The supplier change: what 40 reviewed rejects buy

Board lines. A 3,000-frame window with the second supplier's capacitors from frame 1,500 is inspected by version 1; 40 of its rejects are reviewed
(labels from the truth) and version 2 adds the reviewed good boards where they looked wrong. Both versions then inspect a fresh 3,000 frames with the
new part on 70% of boards. Escapes are split by supplier because version 1 rejects every new-supplier board, which catches their defects for the wrong reason.

| Seed | Chosen by | Good boards reviewed | False rejects, new-supplier boards | False rejects, old-supplier boards | Escapes, old-supplier boards | Escapes, new-supplier boards |
|---|---|---|---|---|---|---|
| 21 | nobody (version 1) | 0 | 100.0% | 0.8% | 7/53 | 0/116 |
| 21 | cluster representatives (the product) | 35 | 27.1% | 1.3% | 7/53 | 10/116 |
| 21 | nearest the threshold | 33 | 51.7% | 1.2% | 7/53 | 10/116 |
| 21 | at random | 37 | 20.1% | 0.9% | 7/53 | 14/116 |
| 21 | highest score | 4 | 88.7% | 0.8% | 7/53 | 1/116 |
| 31 | nobody (version 1) | 0 | 100.0% | 2.4% | 6/44 | 0/119 |
| 31 | cluster representatives (the product) | 35 | 24.5% | 2.3% | 6/44 | 18/119 |
| 31 | nearest the threshold | 31 | 68.9% | 1.9% | 6/44 | 8/119 |
| 31 | at random | 33 | 27.5% | 2.0% | 6/44 | 19/119 |
| 31 | highest score | 4 | 96.3% | 2.3% | 6/44 | 2/119 |
| 41 | nobody (version 1) | 0 | 100.0% | 2.1% | 10/44 | 0/117 |
| 41 | cluster representatives (the product) | 32 | 20.1% | 1.5% | 10/44 | 9/117 |
| 41 | nearest the threshold | 29 | 52.6% | 1.3% | 9/44 | 6/117 |
| 41 | at random | 38 | 19.3% | 1.7% | 10/44 | 14/117 |
| 41 | highest score | 6 | 90.1% | 2.1% | 11/44 | 1/117 |

Run time 217 s.
