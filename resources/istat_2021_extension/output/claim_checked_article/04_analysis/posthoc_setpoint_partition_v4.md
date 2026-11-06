# Post-hoc corrected setpoint partition — v4

Descriptive diagnostic requested after the original primary endpoints were complete, now recomputed only from current v4 files. No extra annual simulation or primary test was added. Positive heating hours use load >1e-6 W; energy is the one-hour sum of max(load,0). The 4/4/1 reported on-hour inputs become 22°C high-setpoint durations under the source mapping, not constraints enforcing the same actual heater runtime. All schedules assign the same calendar hours to 19/22/0°C; positive load hours and energy can differ.

| Coordinate case | Placement | Heating at 19°C kWh | Positive hours at 19°C | Heating at 22°C kWh | Positive hours at 22°C | Heating at 0°C kWh |
|---|---|---:|---:|---:|---:|---:|
|Padova|seeded|2326.804|1350|7462.222|1545|0.000|
|Padova|early|1862.918|1255|8007.972|1527|0.000|
|Padova|late|2166.359|1311|7652.752|1544|0.000|
|Bolzano|seeded|5284.586|2111|10656.094|1638|0.000|
|Bolzano|early|4676.994|2115|11251.683|1638|0.000|
|Bolzano|late|5080.247|2100|10863.165|1638|0.000|
|Roma|seeded|1130.868|841|5739.414|1523|0.000|
|Roma|early|736.681|605|6251.162|1489|0.000|
|Roma|late|986.356|812|5912.774|1520|0.000|
|Palermo|seeded|559.900|540|4808.187|1419|0.000|
|Palermo|early|411.923|422|5125.125|1359|0.000|
|Palermo|late|484.060|514|4924.705|1408|0.000|