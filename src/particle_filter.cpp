/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    std::default_random_engine gen;
    std::normal_distribution<double> Norm_x(x, std[0]);
    std::normal_distribution<double> Norm_y(y, std[1]);
    std::normal_distribution<double> Norm_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle particle;
        particle.id = 1;
        particle.x = Norm_x(gen);
        particle.y = Norm_y(gen);
        particle.theta = Norm_theta(gen);
        particle.weight = 1;

        weights.push_back(particle.weight);
        particles.push_back(particle);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    //  NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {
        double New_x;
        double New_y;
        double New_theta;
        if (yaw_rate == 0) {
            New_x = particles[i].x + velocity * delta_t * sin(particles[i].theta);
            New_y = particles[i].y + velocity * delta_t * cos(particles[i].theta);
            New_theta = particles[i].theta;
        } else {
            New_x = particles[i].x + velocity / yaw_rate * (sin(New_theta + yaw_rate * delta_t) - sin(New_theta));
            New_y = particles[i].y + velocity / yaw_rate * (cos(New_theta) - cos(New_theta + yaw_rate * delta_t));
            New_theta = particles[i].theta + yaw_rate * delta_t;
        }
        cout << "new x value:" << endl;
        cout << New_x << endl;
        normal_distribution<double> N_x(New_x, std_pos[0]);
        normal_distribution<double> N_y(New_y, std_pos[1]);
        normal_distribution<double> N_theta(New_theta, std_pos[2]);

        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    // implement this method and use it as a helper during the updateWeights phase.

    int number_observations = observations.size();
    int number_predictions = predicted.size();

    for (int i = 0; i < number_observations; i++) {
        double min_dis = numeric_limits<double>::max();
        int map_id = -1;

        for (int j = 0; j < number_predictions; j++) {
            double x_dis = observations[i].x - predicted[j].x;
            double y_dis = observations[i].y - predicted[j].y;

            double dis = x_dis * x_dis + y_dis * y_dis;
            if (dis < min_dis) {
                min_dis = dis;
                map_id = predicted[j].id;
            }
        }
        observations[i].id = map_id;
    }
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    //   NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    weights.clear();
    for (int i = 0; i < num_particles; i++) {
        std::vector<LandmarkObs> predicted;

        double x_p = particles[i].x;
        double y_p = particles[i].y;
        double theta_p = particles[i].theta;

        //Transforming the car observations to map coordinates.
        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();

        double weight = 1;

        for (int j = 0; j < observations.size(); j++) {
            double o_x = observations[j].x;
            double o_y = observations[j].y;
            double o_x_map = o_x * cos(theta_p) - o_y * sin(theta_p) + x_p;
            double o_y_map = o_x * sin(theta_p) + o_y * cos(theta_p) + y_p;
            if (pow(pow(o_x_map - x_p, 2) + pow(o_y_map - y_p, 2), 0.5) > sensor_range) continue;

            particles[i].sense_x.push_back(o_x_map);
            particles[i].sense_y.push_back(o_y_map);

            double min_range = 1000000000;
            int min_k = -1;
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
                double l_x = map_landmarks.landmark_list[k].x_f;
                double l_y = map_landmarks.landmark_list[k].y_f;
                double diff_x = l_x - o_x_map;
                double diff_y = l_y - o_y_map;
                double range = pow(pow(diff_x, 2) + pow(diff_y, 2), 0.5);
                if (range < min_range) {
                    min_range = range;
                    min_k = k;
                }
            }

            double l_x = map_landmarks.landmark_list[min_k].x_f;
            double l_y = map_landmarks.landmark_list[min_k].y_f;

            particles[i].associations.push_back(map_landmarks.landmark_list[min_k].id_i);

            weight = weight * exp(-0.5 * (pow((l_x - o_x_map) / std_landmark[0], 2) +
                                          pow((l_y - o_y_map) / std_landmark[1], 2))) /
                     (2 * M_PI * std_landmark[0] * std_landmark[1]);


        }
        particles[i].weight = weight;
        weights.push_back(weight);
    }
}

void ParticleFilter::resample() {
    // NOTE: You may find std::discrete_distribution helpful here.
    // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    //Declaring new vector for resample particles
    vector<Particle> resample_particles;
   //random sampling generator
    default_random_engine gen;

    discrete_distribution<int> distribution(weights.begin(), weights.end());

    //particles resample
    for (int i = 0; i < num_particles; i++) {
        int chosen = distribution(gen);
        resample_particles.push_back(particles[chosen]);
        weights.push_back(particles[chosen].weight);

    }
    particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

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