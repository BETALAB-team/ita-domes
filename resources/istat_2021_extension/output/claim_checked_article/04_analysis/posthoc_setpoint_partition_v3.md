# Post-hoc corrected setpoint partition — v3

Descriptive diagnostic requested after the original primary endpoints were complete, now recomputed only from corrected v3 files. No extra annual simulation or primary test was added. Positive heating hours use load >1e-6 W; energy is the one-hour sum of max(load,0). The 4/4/1 reported on-hour inputs become 22°C high-setpoint durations under the source mapping, not constraints enforcing the same actual heater runtime. All schedules assign the same calendar hours to 19/22/0°C; positive load hours and energy can differ.

| Coordinate case | Placement | Heating at 19°C kWh | Positive hours at 19°C | Heating at 22°C kWh | Positive hours at 22°C | Heating at 0°C kWh |
|---|---|---:|---:|---:|---:|---:|
|Padova|seeded|893.218|671|5054.507|1325|0.000|
|Padova|early|600.841|547|5477.872|1298|0.000|
|Padova|late|789.139|669|5198.984|1321|0.000|
|Bolzano|seeded|2553.326|1426|8060.287|1612|0.000|
|Bolzano|early|2040.652|1324|8616.725|1579|0.000|
|Bolzano|late|2374.667|1398|8257.694|1608|0.000|
|Roma|seeded|243.785|266|3081.869|1160|0.000|
|Roma|early|118.355|135|3403.579|1110|0.000|
|Roma|late|184.957|229|3188.539|1157|0.000|
|Palermo|seeded|63.421|89|2257.670|910|0.000|
|Palermo|early|42.859|57|2432.371|913|0.000|
|Palermo|late|49.707|74|2313.367|916|0.000|