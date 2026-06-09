#include "perception_tracker.h"
#include <cassert>
#include <cstdio>
#include <cmath>
using Vec3 = Eigen::Vector3f;
static std::vector<Vec3> one(const Vec3& v){ return {v}; }
static const Vec3 BASE(-1.6f, 0.f, 0.793f);

// helper: feed the same incoming ball candidate N frames, return final output
static PTOutput feed(PerceptionTracker& t, const Vec3& ball, int n){
    PTOutput o;
    for(int i=0;i<n;i++){ t.update(one(ball), BASE, true, Vec3::Zero()); o=t.output(); }
    return o;
}

void run_volume_gate(){
    PerceptionTracker t;  // default config
    // A point far outside the playable volume (way behind robot) must never engage.
    PTOutput o = feed(t, Vec3(-5.0f, 0.f, 1.0f), 10);
    assert(o.engaged == false);
    printf("  run_volume_gate OK\n");
}

int main(){
    run_volume_gate();
    printf("ALL TESTS PASSED\n");
    return 0;
}
