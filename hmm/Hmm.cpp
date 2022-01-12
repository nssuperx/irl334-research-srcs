#include "Hmm.hpp"
#include <iostream>
#include <random>
#include <vector>
using namespace std;

Hmm::Hmm(int n, double sigma) {
    this->n = n;
    this->sigma = sigma;

    pow_sigma = 2 * sigma * sigma;

    x = vector<int>(n);
    xmap = vector<int>(n);
    y = vector<double>(n);

    C = vector<vector<double>>(n, vector<double>(2));
    S = vector<vector<int>>(n, vector<int>(2));

    random_device seed;
    mt = mt19937(seed());
    rand_uniform = uniform_real_distribution<>(0.0, 1.0); // [0.0, 1.0] 範囲の一様乱数
    rand_dist = normal_distribution<>(0.0, sigma);        // 平均0.0, 標準偏差sigmaの正規分布
}

void Hmm::generate_x() {
    // TODO: 分布の設定にマジックナンバー使ってるので読みにくい
    x.at(0) = rand_uniform(mt) < 0.5 ? 0 : 1;

    for(int i = 1; i < n; i++) {
        if(x.at(i - 1) == 0) {
            x.at(i) = rand_uniform(mt) < 0.99 ? 0 : 1;
        } else {
            x.at(i) = rand_uniform(mt) < 0.97 ? 1 : 0;
        }
    }
}

void Hmm::generate_y() {
    for(int i = 0; i < n; i++) {
        y.at(i) = x.at(i) + rand_dist(mt);
    }
}

void Hmm::calc_xmap() {
    C.at(0).at(0) = -pow(y.at(0) - 0, 2) / pow_sigma;
    C.at(0).at(1) = -pow(y.at(0) - 1, 2) / pow_sigma;

    double t00, t01, t10, t11;

    for(int i=1;i<n;i++){
        C.at(i).at(0) = -pow(y.at(i) - 0, 2) / pow_sigma;
        C.at(i).at(1) = -pow(y.at(i) - 1, 2) / pow_sigma;

        
    }
}
