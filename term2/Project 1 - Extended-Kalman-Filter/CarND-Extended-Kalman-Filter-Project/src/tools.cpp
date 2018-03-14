#include <iostream>
#include "tools.h"
#include "mikeDebug.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::IOFormat;
using std::vector;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  /** Unit 5, Lesson 22 & 23 */ 
  /** Declare and initialize rmse */
  VectorXd rmse(4);
  rmse << 0,0,0,0;
 
  /** Set the size of estimations to a variable */
  unsigned int estimation_size = estimations.size();
  
   /** check the validity of the following inputs:
	* - the estimation vector size should not be zero
	* - the estimation vector size should equal ground truth vector size
	*/
  if (estimation_size == 0) {
	ERROR(__FILE__, __LINE__, __FUNCTION__, "estimation data is invalid!");
  }
  else if (estimation_size != ground_truth.size()) {
	ERROR(__FILE__, __LINE__, __FUNCTION__, "estimation & ground_truth data are different sizes!");  
  }
  else {

#ifdef EXTRA_DEBUG
    /** print sizes of input */
	INFO(__FILE__, __LINE__, __FUNCTION__, "Sizes - Estimations: " << estimation_size << " , Ground truth: " << ground_truth.size());
#endif

    /** accumulate squared residuals */
	for (unsigned int i = 0; i < estimation_size; ++i) {
      VectorXd residual = estimations[i] - ground_truth[i];

#ifdef EXTRA_DEBUG
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "estimations[" << i << "]", estimations[i]);
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "ground_truth[" << i << "]", ground_truth[i]);
#endif	
	  
      residual = residual.array() * residual.array();
      rmse += residual;
   	}
    /** calculate the mean */
    rmse = rmse/estimation_size;
    /** calculate the squared root */
    rmse = rmse.array().sqrt();
  }
    
  return rmse;
}  

VectorXd Tools::CalculateMAE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /** Declare and initialize mae */
  VectorXd mae(4);
  mae << 0,0,0,0;
 
  /** Set the size of estimations to a variable */
  unsigned int estimation_size = estimations.size();
  
   /** check the validity of the following inputs:
	* - the estimation vector size should not be zero
	* - the estimation vector size should equal ground truth vector size
	*/
  if (estimation_size == 0) {
	ERROR(__FILE__, __LINE__, __FUNCTION__, "estimation data is invalid!");
  }
  else if (estimation_size != ground_truth.size()) {
	ERROR(__FILE__, __LINE__, __FUNCTION__, "estimation & ground_truth data are different sizes!");  
  }
  else {

#ifdef EXTRA_DEBUG
    /** print sizes of input */
	INFO(__FILE__, __LINE__, __FUNCTION__, "Sizes - Estimations: " << estimation_size << " , Ground truth: " << ground_truth.size());
#endif

    /** accumulate squared residuals */
	for (unsigned int i = 0; i < estimation_size; ++i) {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array().abs();

#ifdef EXTRA_DEBUG
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "estimations[" << i << "]", estimations[i]);
  PRINTMATRIX(__FILE__, __LINE__, __FUNCTION__, "ground_truth[" << i << "]", ground_truth[i]);
#endif
	  
      mae += residual;
   	}
    /** calculate the mean */
    mae = mae/estimation_size;
  }
    
  return mae;
}  

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  /** Unit 5, Lesson 18 & 19 */
  MatrixXd Hj(3,4);
  /** get state parameters */
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  /** pre-compute a set of terms to avoid repeated calculation */
  float c1 = pow(px, 2) + pow(py, 2);
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  /** check division by zero */
  if(fabs(c1) < EPSILON) {
   	ERROR(__FILE__, __LINE__, __FUNCTION__, "Division by Zero!");
  }
  else {
    /** compute the Jacobian matrix */
    Hj << (px/c2),               (py/c2),               0,     0,
          -(py/c1),              (px/c1),               0,     0,
      	  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
  }
  return Hj;
}
