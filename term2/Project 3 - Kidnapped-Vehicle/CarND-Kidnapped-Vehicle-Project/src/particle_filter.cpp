/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const double EPSILON = 0.0001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


	// Number of particles to draw
	num_particles = 25;
	
	// Flag, if filter is initialized
	is_initialized = false;
	
	// Vector of weights of all particles
	weights.resize(num_particles);

	default_random_engine gen;
	
	// Creates a normal (Gaussian) distribution for x, y, theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for (unsigned int i = 0; i < num_particles; i++) {
      Particle particle;
	  particle.id = i;
	  particle.x = dist_x(gen);
	  particle.y = dist_y(gen);
	  particle.theta = dist_theta(gen);
	  particle.weight = 1.0;
	  particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	
	// Creates a normal (Gaussian) distribution of noise for x, y, theta.
	normal_distribution<double> dist_x_noise(0.0, std_pos[0]);
	normal_distribution<double> dist_y_noise(0.0, std_pos[1]);
	normal_distribution<double> dist_theta_noise(0.0, std_pos[2]);

	yaw_rate = (fabs(yaw_rate) > EPSILON) ? yaw_rate : EPSILON;
	
    for (auto&& particle : particles) {
	  particle.x = particle.x + (velocity / yaw_rate) * (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta)) + dist_x_noise(gen);
	  particle.y = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + (yaw_rate * delta_t))) + dist_y_noise(gen);
	  particle.theta = particle.theta + (yaw_rate * delta_t) + dist_theta_noise(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    /** code is in updateWeights method */
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sigma_x = pow(std_landmark[0], 2);
	double sigma_y = pow(std_landmark[1], 2);
	double cov_xy = std_landmark[0] * std_landmark[1];
	
	for (unsigned int i = 0; i < num_particles; i++) {
      Particle& particle = particles[i];
	  double weight = 1.0;
	  
	  for (auto lm_obs : observations) {

		/** transform coordinates */
		double x_trans = particle.x + (lm_obs.x * cos(particle.theta)) - (lm_obs.y * sin(particle.theta));
		double y_trans = particle.y + (lm_obs.x * sin(particle.theta)) + (lm_obs.y * cos(particle.theta));

        Map::single_landmark_s nearest_neighbor;
        double distance = 0.0;
		double closest_distance = sensor_range;

        /** calculate nearest landmark to observation */		
        for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++) {
          Map::single_landmark_s lm = map_landmarks.landmark_list[k];

		  /** Use helper function to calculate Euclidean distance */
		  distance = dist(x_trans, y_trans, lm.x_f, lm.y_f);
		  
		  /** check if landmark is closer than previous one and update if so */
		  if (distance < closest_distance) {
			  closest_distance = distance;
			  nearest_neighbor = lm;
		  }
		}
        /**Lesson 14, part 18 */
		double x_delta = x_trans - nearest_neighbor.x_f;
		double y_delta = y_trans - nearest_neighbor.y_f;
		weight *= (1/(2 * M_PI * cov_xy)) * (exp(-0.5 * ((x_delta * x_delta)/sigma_x + (y_delta * y_delta)/sigma_y)));
	  }
	  /** update particle and vector that maintains all weights */
	  particles[i].weight = weight;
      weights[i] = weight;
	}
	
	/** normalize to quiet the particles not relevant */
	double weights_sum = 0.0;
	for (auto& w : weights) {
     weights_sum += w;
	} 
	for (unsigned int i = 0; i < num_particles; i++) {
      particles[i].weight /= weights_sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	std::discrete_distribution<int> new_weights(weights.begin(), weights.end());
	std::vector<Particle> new_set_particles;
	
	for (unsigned int i = 0; i < num_particles; i++) {
	  new_set_particles.push_back(particles[new_weights(gen)]);
	}
    particles = new_set_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
