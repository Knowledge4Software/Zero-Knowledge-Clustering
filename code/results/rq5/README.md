# RQ5 — scaling and parameter sensitivity

`runs/native/` contains 200 one-factor scaling rows. `runs/gamma/` contains 80 controlled-separation rows. `runs/precision_20ref/` contains the older 120-row precision experiment. `runs/precision_24ref/` contains the 150-row precision experiment with one shared 24-bit physical reference. `analysis/` contains separate scaling, separation, and precision summaries and charts.

The two precision runs use different normalization references and must not be merged into one curve. The scaling `t_total_s` is a sum of cached native component timings, not ZK proving time or a complete transaction stopwatch. Eight-bit quantization collapsed all tested matrices under the 24-bit reference; sixteen bits has not been established as optimal.
