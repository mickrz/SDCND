#include "PID.h"
#include <vector>
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

void PID::Init(double Kp, double Ki, double Kd) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  p_error = 0.0;
  i_error = 0.0;
  d_error = 0.0;
  int_CTE = 0;
  cte_previous = 0;
}

void PID::UpdateError(double cte) {
  /** cross track error only */	
  p_error = -Kp * cte;
  
  /** sum of all cross track errors */	
  int_CTE += cte;
  i_error = -Ki * int_CTE;

  /** difference of current and previous cross track errors */
  d_error = -Kd * (cte - cte_previous);
  cte_previous = cte;
}

double PID::TotalError() {
  /** sum of all errors calculated from UpdateError() */	
  return (p_error + i_error + d_error);	
}

/**
Revisit this later... error increases so we never get out of loop!!!

void PID::Twiddle(double cte)
{
  double p[] = {0.,0.,0.};
  double dp[] = {0.1,0.1,0.1};
  std::vector<double> p_vec (p, p + sizeof(p) / sizeof(double) );
  std::vector<double> dp_vec (dp, dp + sizeof(dp) / sizeof(double) );
  int iteration = 0;
  double dp_sum = 0.;
  double tolerance = 0.2;
  best_error = 0.1;

  
  for (std::vector<double>::iterator it = dp_vec.begin(); it != dp_vec.end(); ++it)
    dp_sum += *it;

  while (dp_sum > tolerance) {
    for (unsigned int i = 0; i < p_vec.size(); i++) {
      p_vec.at(i) += dp_vec.at(i);

      if (cte < best_error) {
        best_error = cte;
        dp_vec.at(i) *= 1.1;
      }
      else {
        p_vec.at(i) -= 2 * dp_vec.at(i);
        if (cte > best_error) {
          p_vec.at(i) += dp_vec.at(i);
          dp_vec.at(i) *= 0.9;
    	}
        else {
          best_error = cte;
          dp_vec.at(i) *= 1.1;
        }
	  }		  
	}
    iteration++;
  }
  Kp = p_vec.at(0);
  Ki = p_vec.at(1);
  Kd = p_vec.at(2);
  cout << "Kp: " << Kp << " Ki: " << Ki << " Kd: " << Kd << endl;
  UpdateError(best_error);
}*/

