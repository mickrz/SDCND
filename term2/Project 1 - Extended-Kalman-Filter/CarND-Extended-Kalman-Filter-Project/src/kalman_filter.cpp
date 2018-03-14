#include "kalman_filter.h"
#include <iostream>
#include <math.h>
#include "mikeDebug.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  /** 
      predict the state:
      NOTE: For all intents and purposes, this assignment ignores Bu + ν. B is
      the control input matrix and u is the control vector. ν is random noise.

      Unit 5, Lesson 5 states:
      "As an example, let's say we were tracking a car and we knew for certain
	  how much the car's motor was going to accelerate or decelerate over time;
	  in other words, we had an equation to model the exact amount of
	  acceleration at any given moment. Bu would represent the updated position
	  of the car due to the internal force of the motor. We would use ν to
	  represent any random noise that we could not precisely predict like if
	  the car slipped on the road or a strong wind moved the car.

      For the Kalman filter lessons, we will assume that there is no way to
	  measure or know the exact acceleration of a tracked object. For example,
	  if we were in an autonomous vehicle tracking a bicycle, pedestrian or
	  another car, we would not be able to model the internal forces of the
	  other object; hence, we do not know for certain what the other
	  object's acceleration is. Instead, we will set Bu=0 and represent
	  acceleration as a random noise with mean ν."
    */
  x_ = F_ * x_ /*+ Bu + ν */;
  /** NOTE: P​′=FPF(transposed)+Q represents the uncertainty.
      Process noise refers to the uncertainty in the prediction step. We assume
	  the object travels at a constant velocity, but in reality, the object
	  might accelerate or decelerate. The notation ν∼N(0,Q) defines the
	  process noise as a gaussian distribution with mean zero and covariance Q.
	*/
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  
#ifdef EXTRA_DEBUG
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "z", z);
#endif 
 
  /**
      NOTE: K represents the Kalman gain where the higher the gain, the more
      accurate the measurement and less accurate the estimate and vice versa.
      The smaller ther error in the estimate, the more stable. This combines
      uncertainty of where we think we are and where we really are based on
      the sensor measurement R.

      Unit 5, Lesson 5 states:
      "If our sensor measurements are very uncertain (R is high relative to P'),
	  then the Kalman filter will give more weight to where we think we are: x​′​.
	  If where we think we are is uncertain (P' is high relative to R),
	  the Kalman filter will put more weight on the sensor measurement: z.

      Measurement noise refers to uncertainty in sensor measurements. The
	  notation ω~N(0,R) defines the measurement noise as a gaussian
	  distribution with mean zero and covariance R. Measurement noise comes
	  from uncertainty in sensor measurements."	  
	*/
  MatrixXd K = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse();

  // new state
  x_ = x_ + (K * (z - (H_ * x_)));
  P_ = P_ - (K * H_ * P_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  
#ifdef EXTRA_DEBUG
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "z", z);
#endif 

  VectorXd y = z - hx_;
  
  /**
  In C++, atan2() returns values between -pi and pi. When calculating phi in
  y = z - h(x) for radar measurements, the resulting angle phi in the y vector
  should be adjusted so that it is between -pi and pi. The Kalman filter is
  expecting small angle values between the range -pi and pi.
  */  
  bool normalized = false;
  while (normalized == false) {
    if (y(1) > M_PI) {
#ifdef EXTRA_DEBUG
      INFO(__FILE__, __LINE__, __FUNCTION__, "Before: phi: " << y(1));
#endif 

      y(1) = y(1) - 2 * M_PI;

#ifdef EXTRA_DEBUG
      INFO(__FILE__, __LINE__, __FUNCTION__, "After: phi: " << y(1));
#endif 
    }
    else if (y(1) < -M_PI) {
#ifdef EXTRA_DEBUG
      INFO(__FILE__, __LINE__, __FUNCTION__, "Before: phi: " << y(1));
#endif 

      y(1) = y(1) + 2 * M_PI;

#ifdef EXTRA_DEBUG
      INFO(__FILE__, __LINE__, __FUNCTION__, "After: phi: " << y(1));
#endif 
    }
	else {
      normalized = true;
    }
  }
  MatrixXd K = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse();
  // New state
  x_ = x_ + (K * y);
  P_ = P_ - (K * H_ * P_);  
}
