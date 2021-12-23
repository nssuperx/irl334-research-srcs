#include <iostream>
#include <vector>
using namespace std;

class Hmm {
  private:
    int n;
    double sigma;
  public:
    Hmm(int n, double sigma);
    ~Hmm();
};

Hmm::Hmm(int n, double sigma) {
    this->n = n;
    this->sigma = sigma;
}

Hmm::~Hmm() {
}

int main() {
    return 0;
}