#include "FusionEKF.h"
#include "tools.h"
#include <iostream>
#include "mikeDebug.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225,      0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09,      0,    0,
                 0, 0.0009,    0,
                 0,      0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0,    0,
        	 0, 1, 0,    0,
			 0, 0, 1000, 0,
			 0, 0, 0,    1000;
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;  
			 
  // Initialise H_laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
    
  // Initialise H
  ekf_.H_ = MatrixXd(4, 4);
  ekf_.H_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
	INFO(__FILE__, __LINE__, __FUNCTION__, "Initializing...");
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    ekf_.Q_ = MatrixXd(4, 4);
	
	float px = 0.0;
	float py = 0.0;
	float vx = 0.0;
	float vy = 0.0;
	
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
	  float rho = measurement_pack.raw_measurements_[0];
	  float phi = measurement_pack.raw_measurements_[1];
	  px = rho * cos(phi);
	  py = rho * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
	  px = measurement_pack.raw_measurements_[0];
	  py = measurement_pack.raw_measurements_[1];
	}
    else {
      ERROR(__FILE__, __LINE__, __FUNCTION__, "Invalid or unknown sensor type!");
    }
	
	if (fabs(px) < EPSILON) {
	  px = EPSILON;
	  INFO(__FILE__, __LINE__, __FUNCTION__, "Initial px too small");
	}
	
	if (fabs(py) < EPSILON) {
	  py = EPSILON;
	  INFO(__FILE__, __LINE__, __FUNCTION__, "Initial py too small");
	}
	
	ekf_.x_ << px, py, vx, vy;
	previous_timestamp_ =  measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / MICROSECONDS_TO_SECONDS;
  float dt_2 = dt * dt;
  float dt_3 = (dt_2 * dt)/2;
  float dt_4 = (dt_3 * dt)/4;
  float noise_ax = 9.0;
  float noise_ay = 9.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Update the process covariance matrix Q
  ekf_.Q_ <<  dt_4 * noise_ax, 0,               dt_3 * noise_ax, 0,
              0,               dt_4 * noise_ay, 0,               dt_3 * noise_ay,
              dt_3 * noise_ax, 0,               dt_2 * noise_ax, 0,
              0,               dt_3 * noise_ay, 0,               dt_2 * noise_ay;

#ifdef EXTRA_DEBUG
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "ekf_.F_", ekf_.F_);
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "ekf_.Q_", ekf_.Q_);
#endif
			  
  ekf_.Predict();
  
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
#ifdef  MINIMAL_DEBUG
	INFO(__FILE__, __LINE__, __FUNCTION__, "Radar update");
#endif
      
      ekf_.hx_ = VectorXd(3);
      
      float px = ekf_.x_[0];
      float py = ekf_.x_[1];
      float vx = ekf_.x_[2];
      float vy = ekf_.x_[3];
      float rho = 0.0;
      float phi = 0.0;
      float rho_dot = 0.0;

      if(fabs(px) < EPSILON){
        px = EPSILON;
	    INFO(__FILE__, __LINE__, __FUNCTION__, "px too small");
      }

      if(fabs(py) < EPSILON){
        py = EPSILON;
	    INFO(__FILE__, __LINE__, __FUNCTION__, "py too small");
      }
        
      rho = sqrt(pow(px, 2) + pow(py, 2));
      phi = atan2(py, px);
      rho_dot = (px * vx + py * vy) / rho;

      ekf_.hx_ << rho, phi, rho_dot;
      
      // set H_ to Hj when updating with a radar measurement
      Hj_ = tools.CalculateJacobian(ekf_.x_);
      
      // don't update measurement if we can't compute the Jacobian
      if (Hj_.isZero(0)){
	    ERROR(__FILE__, __LINE__, __FUNCTION__, "Hj is zero");
        return;
      }
     
      ekf_.H_ = Hj_;
      ekf_.R_ = R_radar_;
	  
#ifdef EXTRA_DEBUG
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "Hj_", Hj_);
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "R_radar_", R_radar_);
#endif
	  
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);   
   } 
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER){
#ifdef  MINIMAL_DEBUG
    INFO(__FILE__, __LINE__, __FUNCTION__, "Lidar Update");	  
#endif
    // Laser updates
	ekf_.H_ = H_laser_;
	ekf_.R_ = R_laser_;
	
#ifdef EXTRA_DEBUG
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "H_laser_", H_laser_);
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "R_laser_", R_laser_);
#endif	
	
	ekf_.Update(measurement_pack.raw_measurements_);
  }
  else {
    ERROR(__FILE__, __LINE__, __FUNCTION__, "Invalid or unknown sensor type!"); 	
  }

#ifdef MINIMAL_DEBUG
  // print the output
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "ekf_.x_", ekf_.x_);
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "ekf_.P_", ekf_.P_);
#endif	
}
