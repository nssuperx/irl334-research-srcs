#include<iostream>
#include<list>
#include<string>
using namespace std;

class Invariance{
    public:
    Invariance(string s);
    void addDown(Invariance* i);
    bool isPrimitive();
    bool isTop();
    list<Invariance*> searchCycle();

    private:
    string name;
    list<Invariance*> upList;
    list<Invariance*> downList;
};

Invariance::Invariance(string s){
    name = s;
}

void Invariance::addDown(Invariance* i){
    downList.push_back(i);
    i->upList.push_back(this);
}

bool Invariance::isPrimitive(){
    return downList.empty();
}

bool Invariance::isTop(){
    return upList.empty();
}

list<Invariance*> searchCycle(){
    list<Invariance*> cycleList;
    
    return cycleList;
}

int main(){
    return 0;
}