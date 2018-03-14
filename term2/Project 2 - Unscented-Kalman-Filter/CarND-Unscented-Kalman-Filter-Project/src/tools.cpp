#include <iostream>
#include "tools.h"
#include "mikeDebug.h"
using Eigen::VectorXd;
using Eigen::MatrixXd;
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

