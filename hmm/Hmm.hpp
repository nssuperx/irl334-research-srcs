#pragma once
#include <random>
#include <vector>
using namespace std;

class Hmm {
  private:
    int n;
    double sigma;
    double pow_sigma;

    vector<vector<double>> C;
    vector<vector<int>> S;

    // 乱数まわり
    mt19937 mt;                               // 疑似乱数生成器
    uniform_real_distribution<> rand_uniform; // 一様乱数
    normal_distribution<> rand_dist;          // 正規分布

    
  public:
    // NOTE: 実験用なのでpublicにした
    vector<int> x, xmap;
    vector<double> y;

  public:
    Hmm(int n, double sigma);
    void generate_x();
    void generate_y();
    void calc_xmap();
};
