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
#include <limits>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set number of particles to draw
  num_particles = 2;
  weights.resize(num_particles);

  // Create normal distributions for x, y, and theta based on initial GPS data and their uncertainties
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize particles based on GPS with Gaussian noise and set weights to 1
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    weights[i] = 1.0;
    particles.push_back(p);
  }

  // Set initialized flag to true
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Create random_engine generator for adding noise to predictions
  default_random_engine gen;

  // Make predictions based on velocity and yaw_rate
  for (int i = 0; i < num_particles; ++i) {
    particles[i].x += velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
    particles[i].y += velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
    particles[i].theta += yaw_rate*delta_t;
    // Create normal distribution for generating random Gaussian noise
    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
    // Save prediction with added random Gaussian noise
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement
  // and assign the observed measurement to this particular landmark.
  double distance;
  for (int i = 0; i < observations.size(); ++i) {
    double min_dist = numeric_limits<double>::max();
    int min_id = -1;
    for (int j = 0; j < predicted.size(); ++j) {
      distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < min_dist) {
        min_dist = distance;
        min_id = predicted[j].id;
      }
    }
    observations[i].id = min_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
  std::vector<LandmarkObs> observations, Map map_landmarks) {
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  weights.clear();

  for (int i = 0; i < particles.size(); ++i) {
    // Transform observations to MAP's coordinate system
    std::vector<LandmarkObs> observations_map;
    for (int j = 0; j < observations.size(); ++j) {
      LandmarkObs obs;
      obs.x = observations[j].x * cos(particles[i].theta) -
              observations[j].y * sin(particles[i].theta) + particles[i].x;
      obs.y = observations[j].x * sin(particles[i].theta) +
              observations[j].y * cos(particles[i].theta) + particles[i].y;
      obs.id = -1;
      observations_map.push_back(obs);
    }

    // Compute predicted measurements
    std::vector<LandmarkObs> predicted;
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double distance_p_obs;
      distance_p_obs = dist(particles[i].x, particles[i].y,
          map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if (distance_p_obs <= sensor_range) {
        LandmarkObs obs;
        obs.id = map_landmarks.landmark_list[j].id_i;
        obs.x = map_landmarks.landmark_list[j].x_f;
        obs.y = map_landmarks.landmark_list[j].y_f;
        predicted.push_back(obs);
      }
    }

    dataAssociation(predicted, observations_map);

    double prob = 1.0;
    double prob_i;
    for (int j = 0; j < predicted.size(); ++j) {
      double min_dist = numeric_limits<double>::max();
      int min_idx = -1;
      for (int k = 0; k < observations_map.size(); ++k) {
        // Use measurement closest to predicted
        if (predicted[j].id == observations_map[k].id) {
          double check_dist = dist(predicted[j].x, predicted[j].y,
                                  observations_map[k].x, observations_map[k].y);
          if (check_dist < min_dist) {
            min_dist = check_dist;
            min_idx = k;
          }
        }
      }
      if (min_idx != -1) {
        prob_i = exp(-((predicted[j].x - observations_map[min_idx].x) *
                    (predicted[j].x - observations_map[min_idx].x) /
                    (2 * std_landmark[0] * std_landmark[0]) +
                    (predicted[j].y - observations_map[min_idx].y) *
                    (predicted[j].y - observations_map[min_idx].y) /
                    (2 * std_landmark[1] * std_landmark[1]))) /
                    (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
        prob = prob * prob_i;
      }
    }
    weights.push_back(prob);
    particles[i].weight = prob;
  }
}

void ParticleFilter::resample() {
  // Create a discrete distribution for resampling particles with probability proportional
  // to their weight. Random number generation based on the Mersenne Twister algorithm.
  // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  // Resample particles
  std::vector<Particle> resampled;
  for (int i = 0; i < num_particles; ++i) {
    Particle p = particles[d(gen)];
    resampled.push_back(p);
  }
  particles = resampled;

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
