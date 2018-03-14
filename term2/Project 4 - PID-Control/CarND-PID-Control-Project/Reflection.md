# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Reflection

### Observations/Notes on initialization of PIDs:
* The higher the cruising speed, the more eratic the ride is going to be on the road. At lower speeds, the init values are more forgiving. This just reinforces that twiddle algo is very beneficial if not necessary in the real-world.
* Intentionally set cruising speed to 35, but can go even as high as 45 though it does become a very jerky ride. A value of 55 is disastrous. Lower than 35 is more pleasant.
* My observations on the PID components were as follows via trail and error (manual tuning):
- P: the higher the value such as 0.9, the more the front wheels would turn back and forth although it would typically stay on the track. A minimal value of 0.1 was chosen.
- I: the higher the value such as 0.1, along with reasonable values for P and D would cause the car to go off track. A minimal value of 0.01 was chosen.
- D: the higher the value such as 10, the larger the car swerved back and forth to the right. Also, too small a value such as less than 1.0, the car would go off-track. A minimal value of 2.0 was chosen.
* I have some of the combinations commented in my code in main.cpp for reference.


### Observations/Notes on throttle/speed relationship:
* A high throttle value will allow the car to go faster but without using twiddle (or something similar), the car steering will oscillate left to right with large angles and quickly go off track.
* Revisit twiddle after the course completes, but to complete the project for now, I will submit what I have which is not optimized as I would have liked.
* Used the speed to calculate throttle error since throttle is a constant.
* Used a speed pid and used speed value and a various constant values to create an error value to pass into UpdateError(), but regardless I did not see any advantage or disadvantage.	
