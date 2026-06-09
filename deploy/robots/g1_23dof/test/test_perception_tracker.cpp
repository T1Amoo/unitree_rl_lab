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

void run_kf_smooth_and_coast(){
    PerceptionTracker t;
    // Feed a noisy straight ballistic-ish ball moving toward robot; KF should track it,
    // and during a dropout it should COAST (keep producing a moving estimate, not freeze).
    Vec3 p(1.0f, 0.f, 1.0f); Vec3 v(-3.0f, 0.f, 0.f);
    for(int i=0;i<6;i++){ p += v*0.02f; t.update(one(p), BASE, true, Vec3::Zero()); }
    Vec3 last_seen = t.ball_estimate();
    // dropout: no candidates for 3 frames
    for(int i=0;i<3;i++){ t.update({}, BASE, true, Vec3::Zero()); }
    Vec3 coasted = t.ball_estimate();
    // coasted x must have advanced in -x (moved), not frozen at last_seen
    assert(coasted.x() < last_seen.x() - 0.01f);
    printf("  run_kf_smooth_and_coast OK\n");
}

int main(){
    run_volume_gate();
    run_kf_smooth_and_coast();
    printf("ALL TESTS PASSED\n");
    return 0;
}
