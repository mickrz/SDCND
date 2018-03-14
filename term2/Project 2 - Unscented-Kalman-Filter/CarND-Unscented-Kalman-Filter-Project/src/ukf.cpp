#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "mikeDebug.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/** Initializes Unscented Kalman filter */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // State dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);  
  
  // Augmented state dimension
  n_aug_ = 7;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  //create sigma point matrix
  MatrixXd Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);

  //create sigma point matrix
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, n_sig_);

  // Predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 4.5; 
  
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.75;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // R matrices
  n_z_radar_ = 3;
  R_radar_ = MatrixXd(n_z_radar_,n_z_radar_);
  R_radar_ << std_radr_*std_radr_, 0,                       0,
              0,                   std_radphi_*std_radphi_, 0,
              0,                   0,                       std_radrd_*std_radrd_;

  n_z_lidar_ = 2;
  R_lidar_ = MatrixXd(n_z_lidar_,n_z_lidar_);
  R_lidar_ << std_laspx_*std_laspx_, 0,
              0,                     std_laspy_*std_laspy_;
  
  // Weights of sigma points
  weights_ = VectorXd(n_sig_);
  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < n_sig_; i++) {  
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
 
 
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	INFO(__FILE__, __LINE__, __FUNCTION__, "Initializing...");
	
	double px = 0.0;
	double py = 0.0;
	double vx = 0.0;
	double vy = 0.0;
	
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /** Convert radar from polar to cartesian coordinates and initialize state. */
	  double rho = meas_package.raw_measurements_[0];
	  double phi = meas_package.raw_measurements_[1];
	  double rho_dot = meas_package.raw_measurements_[2];
	  px = rho * cos(phi);
	  py = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /** Initialize state. */
	  px = meas_package.raw_measurements_[0];
	  py = meas_package.raw_measurements_[1];
	}
    else {
      ERROR(__FILE__, __LINE__, __FUNCTION__, "Invalid or unknown sensor type!");
    }
	
	x_ << std::max(EPSILON, fabs(px)), std::max(EPSILON, fabs(py)), sqrt(pow(vx, 2) + pow(vy, 2)), 0, 0;
	
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
	return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / MICROSECONDS_TO_SECONDS; 
  time_us_ = meas_package.timestamp_;  

  /** initial state vector */
  VectorXd x_aug = VectorXd(n_aug_);

  /** initial covariance matrix */
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  
  /** create augmented mean state */
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  /** create augmented covariance matrix */
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_) = pow(std_a_, 2);
  P_aug(n_x_ + 1,n_x_ + 1) = pow(std_yawdd_, 2);

  /** create square root matrix */
  MatrixXd L = P_aug.llt().matrixL();

  /** create augmented sigma points */
  Xsig_aug_ = MatrixXd(n_aug_, n_sig_);
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug_.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
#ifdef  MINIMAL_DEBUG
	INFO(__FILE__, __LINE__, __FUNCTION__, "Radar update");
#endif
      if (use_radar_) {
        UpdateRadar(meas_package);
	  }
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
#ifdef  MINIMAL_DEBUG
	INFO(__FILE__, __LINE__, __FUNCTION__, "Lidar update");
#endif
      if (use_laser_) {
        UpdateLidar(meas_package);
	  }
  }
  else {
    ERROR(__FILE__, __LINE__, __FUNCTION__, "Invalid or unknown sensor type!"); 	
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /** predict sigma */ 
  for (int i = 0; i < n_sig_; i++)
  {  
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v =   Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    /** predicted state values */
	double px_p = (fabs(yawd) > EPSILON) ? p_x + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw)) : p_x + v * delta_t * cos(yaw);
	double py_p = (fabs(yawd) > EPSILON) ? p_y + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t)) : p_y + v * delta_t * sin(yaw);
	double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    /** add noise & write predicted sigma points into right column */
    Xsig_pred_(0,i) = px_p + 0.5 * pow(delta_t, 2) * nu_a * cos(yaw);
    Xsig_pred_(1,i) = py_p + 0.5 * pow(delta_t, 2) * nu_a * sin(yaw);
    Xsig_pred_(2,i) = v_p + nu_a * delta_t;
    Xsig_pred_(3,i) = yaw_p + 0.5 * pow(delta_t, 2) * nu_yawdd;
    Xsig_pred_(4,i) = yawd_p + nu_yawdd * delta_t;
  }
  
  /** create vector for predicted state */
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }

  /** create covariance matrix for prediction */
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    /** state difference */
    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
    /** angle normalization */
    while (x_diff(3)> M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2. * M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  x_ = x;
  P_ = P;
  
#ifdef MINIMAL_DEBUG
  // print the output
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "Predict: x", x_);
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "Predict: P", P_);
#endif	  
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /** Use lidar data to update the belief about the object's position. Modify the state vector, x_, and covariance, P_. */
  /** Predict Lidar Measurement */

  // incoming lidar measurement
  z_ = VectorXd(n_z_lidar_);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  
  //create matrix for sigma points in measurement space
  Zsig_ = MatrixXd(n_z_lidar_, n_sig_);  

  /** transform sigma points into measurement space */
  for (int i = 0; i < n_sig_; i++) {
    double p_x = Xsig_pred_(0,i);
	double p_y = Xsig_pred_(1,i);

	bool check_if_greater = ((fabs(p_x) > EPSILON) || (fabs(p_y) > EPSILON));
    // measurement model
    Zsig_(0,i) = (check_if_greater) ? p_x : EPSILON;
    Zsig_(1,i) = (check_if_greater) ? p_y : EPSILON;
  }

  //mean predicted measurement
  z_pred_ = VectorXd(n_z_lidar_);
  z_pred_.fill(0.0);
  for (int i=0; i < n_sig_; i++) {
      z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }

  //measurement covariance matrix S_
  S_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  S_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S_ = S_ + R_lidar_;
 
  /**update state mean and covariance matrix and return calculated NIS */
  NIS_laser_ = UpdateState(n_z_lidar_);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /** Use radar data to update the belief about the object's position. Modify the state vector, x_, and covariance, P_. */
  /** Predict Radar Measurement */

  // incoming radar measurement
  z_ = VectorXd(n_z_radar_);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];

  //create matrix for sigma points in measurement space
  Zsig_ = MatrixXd(n_z_radar_,n_sig_);
  
  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
	double q = sqrt(pow(p_x,2) + pow(p_y,2));
	bool check_if_greater = ((fabs(p_x) > EPSILON) || (fabs(p_y) > EPSILON));
    Zsig_(0,i) = (check_if_greater) ? q: EPSILON;                        //r
    Zsig_(1,i) = (check_if_greater) ? atan2(p_y, p_x) : EPSILON;         //phi
    Zsig_(2,i) = (check_if_greater) ? (p_x*v1 + p_y*v2 ) / q : EPSILON;  //r_dot
  }

  //mean predicted measurement
  z_pred_ = VectorXd(n_z_radar_);
  z_pred_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }

  //measurement covariance matrix S
  S_ = MatrixXd(n_z_radar_, n_z_radar_);
  S_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    /** calculate residual & normailze */
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    while (z_diff(1) > M_PI) z_diff(1) -=2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) +=2. * M_PI;

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  S_ = S_ + R_radar_;

  /**update state mean and covariance matrix and return calculated NIS */
  NIS_radar_ = UpdateState(n_z_radar_);
}

double UKF::UpdateState(int n_z)
{
   //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    /** calculate residual & normailze */
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    /** calculate state difference & normailze */
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  
  //Kalman gain K;
  MatrixXd K = Tc * S_.inverse();

  /** calculate residual & normailze */
  VectorXd z_diff = z_ - z_pred_;
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;	
	
  /** update state mean and covariance matrix */
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();

#ifdef MINIMAL_DEBUG
  /** print the output */
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "x", x_);
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "P", P_);
#endif

  return z_diff.transpose() * S_.inverse() * z_diff; 
}