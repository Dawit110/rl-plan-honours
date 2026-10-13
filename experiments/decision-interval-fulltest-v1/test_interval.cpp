#include "interval_control.h"
#include <cassert>
#include <iostream>
#include <limits>
using namespace decision_interval;
Stats sample(double value=100) {
    Stats s;
    for(int i=0;i<4;++i) s[i]={{"Min Value",value},{"Open List Entries",10}};
    return s;
}
void start(Control &c, Stats &s) {
    assert(c.due(s,0)); c.choose(0,0);
    assert(c.due(s,1)); c.choose(0,1); // one-removal FF warm-up
}
int main() {
    {Control c(0,16,4,8,64);auto s=sample();start(c,s);assert(c.due(s,1));}
    {Control c(1,16,4,8,64);auto s=sample();start(c,s);
     for(int n=0;n<100;++n)assert(!c.due(s,1)); // closed entries do not advance
     for(int e=2;e<17;++e) { assert(!c.due(s,e)); }
     assert(c.due(s,17));}
    {Control c(2,16,4,8,64);auto s=sample();start(c,s);
     for(int e=2;e<9;++e) { assert(!c.due(s,e)); }
     assert(c.due(s,9));assert(c.reason=="stagnation");}
    {Control c(2,16,4,8,64);auto s=sample();start(c,s);
     for(int e=2;e<65;++e){s[0]["Min Value"]=100-e;assert(!c.due(s,e));}
     s[0]["Min Value"]=35;assert(c.due(s,65));assert(c.reason=="maximum");}
    {Control c(2,16,4,8,64);auto s=sample();start(c,s);
     for(int e=2;e<9;++e){s[0]["Min Value"]=(e%2)?100:110;assert(!c.due(s,e));}
     assert(c.due(s,9));} // oscillation back to an old best is not progress
    {Control c(2,16,4,8,64);auto s=sample();start(c,s);
     for(int e=2;e<7;++e)assert(!c.due(s,e));
     s[0]["Min Value"]=99;assert(!c.due(s,7));
     for(int e=8;e<15;++e) { assert(!c.due(s,e)); }
     assert(c.due(s,15));}
    {Control c(2,16,4,8,64);auto s=sample();start(c,s);
     s[0]["Min Value"]=std::numeric_limits<double>::infinity();
     s[0]["Open List Entries"]=0;assert(c.due(s,1));assert(c.reason=="empty");
     assert(c.best[0]==100);}
    {Control c(2,16,4,8,64);auto s=sample();start(c,s);
     s[1]["Min Value"]=1;assert(!c.due(s,2));c.choose(1,2);
     for(int e=3;e<10;++e) { assert(!c.due(s,e)); }
     assert(c.due(s,10));}
    {bool failed=false;try{Control c(1,0,4,8,64);}catch(const std::invalid_argument&){failed=true;}assert(failed);}
    std::cout<<"9 interval control checks passed\n";
}
