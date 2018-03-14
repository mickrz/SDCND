#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_

#include "Eigen/Dense"

class MeasurementPackage {
public:
  long long timestamp_;

  enum SensorType{
    LASER, /** +/-2cm depth accuracy, range up to 60m, good for blind spot warning, collision warning & avoidance */
    RADAR, /** tracks objects as far as 200m away, complimentary to LIDAR */
	SONAR, /** short range of ~5m, good for parking sensors */
	CAMERA /** another sensor which can track objects, but can't track kangaroos well */
  } sensor_type_;

  Eigen::VectorXd raw_measurements_;
};

#endif /* MEASUREMENT_PACKAGE_H_ */
