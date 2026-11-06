# Post-hoc descriptive setpoint partition

**Provisional diagnostic:** this partitions existing outputs subject to the timestep-dispatch scientific-use hold; it is not a validated runtime/energy result. See `05_simulations/SCIENTIFIC_USE_HOLD.md`.

Requested after the primary simulations and endpoints were complete; no new annual runs or primary tests were added. Positive heating hours use sensible load >1e-6 W. Energy sums include max(load,0) at the one-hour timestep. Columns distinguish calendar hours assigned each lower setpoint, hours with positive modeled heating, and their energy. Every schedule has the same number of hours at 19/22/0°C; the positive-load hours and energy need not match.

The source mapping preserves 4/4/1 hours assigned the high setpoint in the three reported blocks. It does not enforce 4/4/1 modeled heater-runtime hours. The selected timer-control branch imposes 19°C setback outside its high-setpoint hours, so the ideal system may continue supplying heat at setback. Outside the source-defined season the lower setpoint is 0°C. The reported survey on-hours therefore become a temperature-schedule assumption in this adapter, not a constraint guaranteeing equal realized heating runtime. This is a material interpretation limit.

| Site | Schedule | Heating at 19°C kWh | Positive hours at 19°C | Heating at 22°C kWh | Positive hours at 22°C | Heating at 0°C kWh |
|---|---|---:|---:|---:|---:|---:|
|Padova|seeded|1220.322|801|5300.863|1330|0.000|
|Padova|early|792.663|601|5583.887|1317|0.000|
|Padova|late|1082.764|750|5329.792|1321|0.000|
|Bolzano|seeded|2992.172|1546|8341.844|1629|0.000|
|Bolzano|early|2359.347|1379|8619.715|1581|0.000|
|Bolzano|late|2814.972|1499|8468.169|1615|0.000|
|Roma|seeded|475.629|410|3392.440|1156|0.000|
|Roma|early|217.094|187|3674.289|1150|0.000|
|Roma|late|327.846|302|3318.221|1151|0.000|
|Palermo|seeded|122.106|154|2425.293|911|0.000|
|Palermo|early|65.521|76|2658.058|968|0.000|
|Palermo|late|81.051|97|2353.073|912|0.000|
