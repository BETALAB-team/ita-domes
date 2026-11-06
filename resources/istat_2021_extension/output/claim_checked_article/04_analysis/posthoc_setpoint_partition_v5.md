# Post-hoc candidate setpoint partition — v5

Descriptive diagnostic requested after the original primary endpoints were complete, now recomputed only from candidate v5 files. No extra annual simulation or primary test was added. Positive heating hours use load >1e-6 W; energy is the one-hour sum of max(load,0). The 4/4/1 reported on-hour inputs become 22°C high-setpoint durations under the source mapping, not constraints enforcing the same actual heater runtime. All schedules assign the same calendar hours to 19/22/0°C; positive load hours and energy can differ.

| Coordinate case | Placement | Heating at 19°C kWh | Positive hours at 19°C | Heating at 22°C kWh | Positive hours at 22°C | Heating at 0°C kWh |
|---|---|---:|---:|---:|---:|---:|
|Padova|seeded|2375.617|1323|7198.401|1495|0.000|
|Padova|early|1882.699|1229|7776.934|1456|0.000|
|Padova|late|2207.735|1287|7398.562|1496|0.000|
|Bolzano|seeded|5380.284|2073|10470.930|1635|0.000|
|Bolzano|early|4734.495|2077|11101.188|1628|0.000|
|Bolzano|late|5162.094|2052|10693.371|1635|0.000|
|Roma|seeded|1166.722|839|5435.507|1466|0.000|
|Roma|early|742.253|593|5994.193|1442|0.000|
|Roma|late|1014.224|810|5621.017|1456|0.000|
|Palermo|seeded|590.691|545|4488.147|1307|0.000|
|Palermo|early|422.799|417|4852.494|1270|0.000|
|Palermo|late|508.745|525|4619.939|1302|0.000|