#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"
#include "Eigen/src/Core/IO.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

class Tools {
public:
  Tools() {};
  virtual ~Tools() {};
  /** A helper method to calculate RMSE. */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);
  
  /** A helper method to calculate MAE. Alternative to measuring error. 
      https://en.wikipedia.org/wiki/Mean_absolute_error */
  VectorXd CalculateMAE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);
  
  /** A helper method to calculate Jacobians. */
  MatrixXd CalculateJacobian(const VectorXd& x_state);
};

#endif /* TOOLS_H_ */
