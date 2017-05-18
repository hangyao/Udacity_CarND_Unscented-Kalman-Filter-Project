#include <cmath>
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd::Zero(4);
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Invalid input" << endl;
    return rmse;
  }

  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

VectorXd Tools::PolarToCartesianMeasurement(const VectorXd& x_state) {
  float rho = x_state(0);
  float phi = x_state(1);
  float rho_dot = x_state(2);
  float tan_phi = tan(phi);
  float px = sqrt(rho * rho / (1 + tan_phi * tan_phi));
  float py = tan_phi * px;
  float vx = rho_dot * cos(phi);
  float vy = rho_dot * sin(phi);
  float v  = sqrt(vx * vx + vy * vy);

  VectorXd cartesian_vec(3);
  cartesian_vec << px, py, v;
  return cartesian_vec;
}

VectorXd Tools::CartesianToPolarMeasurement(const VectorXd& x_state) {
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  float position_vec_magnitude = sqrt(px * px + py * py);

  if (px == 0) {
    px = 1e-10;
    position_vec_magnitude = sqrt(px * px + py * py);
  }

  if (position_vec_magnitude == 0) position_vec_magnitude = 1e-10;

  VectorXd polar_vec(3);
  polar_vec << position_vec_magnitude,
               atan2(py, px),
               (px * vx + py * vy) / position_vec_magnitude;
  return polar_vec;
}

VectorXd Tools::ProducePredictionWeights(unsigned int num_sigma_points,
                                       unsigned int augmented_state_size,
                                       double lambda) {
  VectorXd weights(num_sigma_points);
  weights.fill(0.5 / (lambda + augmented_state_size));
  weights(0) = lambda / (lambda + augmented_state_size);
  return weights;
}

double Tools::NormaliseAngle(double theta) {
  return fmod(theta, (M_PI + 1e-10));
}

VectorXd Tools::NormaliseAngles(VectorXd thetas) {
  VectorXd normalised_thetas(thetas.size());
  for (unsigned int i = 0; i < thetas.size(); i++) {
    normalised_thetas(i) = NormaliseAngle(thetas(i));
  }
  return normalised_thetas;
}

double Tools::CalculateNIS(VectorXd z_diff, MatrixXd S) {
  return z_diff.transpose() * S.inverse() * z_diff;
}
