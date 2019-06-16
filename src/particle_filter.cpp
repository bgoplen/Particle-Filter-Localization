/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles
  default_random_engine gen;
  
  // initialize normal distributions
  normal_distribution<double> nd_x(x, std[0]);
  normal_distribution<double> nd_y(y, std[1]);
  normal_distribution<double> nd_theta(theta, std[2]);

  // initialize particles
  for (int i = 0; i < num_particles; i++) {
    Particle part;
    part.id = i;
    part.x = nd_x(gen);
    part.y = nd_y(gen);
    part.theta = nd_theta(gen);
    part.weight = 1.0;

    particles.push_back(part);
    weights.push_back(1.0);
  }
  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  
  // initialize normal distributions
  normal_distribution<double> nd_x(0, std_pos[0]);
  normal_distribution<double> nd_y(0, std_pos[1]);
  normal_distribution<double> nd_theta(0, std_pos[2]);

  // update each particle
  for (int i = 0; i < num_particles; i++) {

    // calculate new state
    if (fabs(yaw_rate) < 0.0001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add random Gaussian noise
    particles[i].x += nd_x(gen);
    particles[i].y += nd_y(gen);
    particles[i].theta += nd_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
	for (unsigned i = 0; i < observations.size(); i++) {
		double min_d = numeric_limits<double>::max();
		int min_id = -1;
		for (unsigned j = 0; j < predicted.size(); j++) {
			double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (d < min_d) {
				min_d = d;
				min_id = predicted[j].id;
			}
		}
		observations[i].id = min_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i = 0; i < num_particles; i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    // find landmarks within sensor range of particle
    vector<LandmarkObs> close_landmarks;
    for (unsigned j = 0; j < map_landmarks.landmark_list.size(); j++) {      
      if ( dist(x,y,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f) <= sensor_range ) {
        close_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }

    // transform sensor observations from vehicle coordinates to map coordinates
    vector<LandmarkObs> trans_observations;
    for (unsigned j = 0; j < observations.size(); j++) {
      double map_x = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
      double map_y = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
      trans_observations.push_back(LandmarkObs{observations[j].id, map_x, map_y});
    }

    // associate within-range landmarks to observations
    dataAssociation(close_landmarks, trans_observations);

    // calculate weights
    particles[i].weight = 1.0;
    for (unsigned j = 0; j < trans_observations.size(); j++) {
      double d_x, d_y;
      for (unsigned k = 0; k < close_landmarks.size(); k++) {
        if (close_landmarks[k].id == trans_observations[j].id) {
          d_x = trans_observations[j].x-close_landmarks[k].x;
          d_y = trans_observations[j].y-close_landmarks[k].y;
        }
      }
      particles[i].weight *= exp( -(pow(d_x/std_landmark[0],2) + pow(d_y/std_landmark[1],2))/2.0 ) / (2.0*M_PI*std_landmark[0]*std_landmark[1]);
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;

  for (unsigned i = 0; i < particles.size(); i++) {
    weights[i] = particles[i].weight;
  }

  double beta = 0.0;
  uniform_real_distribution<double> real_dist(0.0, 2.0 * (*max_element(weights.begin(), weights.end())));
  uniform_int_distribution<int> int_dist(0, num_particles-1);
  int j = int_dist(gen);
  vector<Particle> new_parts;
  for (int i = 0; i < num_particles; i++) {
    beta += real_dist(gen);
    while (beta > weights[j]) {
      beta -= weights[j];
      j = (j + 1) % num_particles;
    }
    new_parts.push_back(particles[j]);
  }

  particles = new_parts;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}