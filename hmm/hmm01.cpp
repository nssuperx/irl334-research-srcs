#include "Hmm.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace std;

int main() {
    int n = 100;
    Hmm hmm_test = Hmm(n, 0.7);
    vector<int> t(n);
    iota(t.begin(), t.end(), 0);
    hmm_test.generate_x();
    hmm_test.generate_y();
    hmm_test.calc_xmap();

    ofstream outfile("out.csv");

    for(int i = 0; i < n; i++) {
        // printf("%d %d %f\n", t.at(i), hmm_test.x.at(i), hmm_test.y.at(i));
        outfile << t.at(i) << ',' << hmm_test.x.at(i) << ',' << fixed << setprecision(6) << hmm_test.y.at(i) << endl;
    }
    return 0;
}
