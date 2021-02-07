# stereo_paragram_optimizer

This is about optimizering two cameras which has thire own K&D, but no accurate Transform between them, and we have no condition to calibrate these two cameras.

The main idea is from ORB_SLAM2, it looks like tracking two or more stereo images so that we can caculate the Transform, and using many ORB KeyPoints to optimizer it.



### TODO

recode and readme

1. Separate the Mobocular.cpp into many class for better reusing
2. Extract the io from the system, and code a test.cpp for example using 
3. Add check, check the result
4. Add expand io for more data