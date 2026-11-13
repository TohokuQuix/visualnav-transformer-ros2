# Scene1 model2000 10-trial navigation
Goal count: 0/10. Outcomes: {'tip_over': 10}. Outside-course trials: 0.
Start requested(.6,1.7),yaw-y; goal(4.3,.5), radius.2m, heading ignored. Maximum300 simulated seconds. 30s nohistory NN+same Actor, B60deg with explicit course boundary exclusion.
|trial|result|seconds|best distance m|final XY m|region|blocked linear s|blocked turn s|
|---|---|---:|---:|---|---|---:|---:|
|original-t01|tip_over|29.00|2.729|(1.492,0.829)|central_lower_ramps|0.0|0.0|
|original-t02|tip_over|22.60|2.920|(1.357,0.908)|central_lower_ramps|0.0|0.0|
|original-t03|tip_over|10.05|3.016|(1.274,0.393)|left_V_section|0.0|0.0|
|original-t04|tip_over|16.65|2.881|(1.449,0.914)|central_lower_ramps|0.0|0.0|
|original-t05|tip_over|67.00|1.589|(2.981,1.391)|central_upper_ramps|0.0|0.0|
|original-t06|tip_over|32.20|2.751|(1.465,0.810)|central_lower_ramps|0.0|0.0|
|original-t07|tip_over|10.90|2.997|(1.303,0.483)|central_lower_ramps|0.0|0.0|
|original-t08|tip_over|23.80|2.948|(1.354,0.927)|central_lower_ramps|0.0|0.0|
|original-t09|tip_over|23.00|2.953|(1.383,0.954)|central_lower_ramps|0.0|0.0|
|original-t10|tip_over|23.60|2.917|(1.409,0.890)|central_lower_ramps|0.0|0.0|

Blocked diagnostics are post-hoc2s observations, not control rules. Linear: meanabs requestedvx>=.1m/s with total3Dmotion<.04m/2s. Turn: meanabs requestedwz>=.2rad/s with actualyawtravel<.1rad/2s. Commanded STOP is separate and does not guarantee physical stillness. Exact locations/times in motion-diagnostics.csv and trajectories.npz.
Successes involving outside-course detours are not counted. Best video must render recorded root/joints; it does not constitute a fresh deterministic resimulation. Finite terrain visibility is ideal geometric DEM, not actual FAST-LIO.
