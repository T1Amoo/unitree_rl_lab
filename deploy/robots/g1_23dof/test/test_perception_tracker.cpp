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

void run_reflection_rejection(){
    PerceptionTracker t;
    Vec3 p(1.0f,0.f,1.0f), v(-3.0f,0.f,0.f);
    for(int i=0;i<6;i++){ p+=v*0.02f; t.update(one(p), BASE, true, Vec3::Zero()); }
    // Now feed TWO candidates: the true continuation + a far reflection blip.
    p += v*0.02f;
    Vec3 reflection(0.5f, 0.8f, 1.5f);   // in-volume but far from predicted track
    std::vector<Vec3> cands = { reflection, p };
    t.update(cands, BASE, true, Vec3::Zero());
    // tracker must follow the true ball (near p), not jump to the reflection
    assert((t.ball_estimate() - p).norm() < 0.2f);
    assert((t.ball_estimate() - reflection).norm() > 0.5f);
    printf("  run_reflection_rejection OK\n");
}

void run_dead_ball(){
    {   // rolling on the table: z at table height, vz~0, moving horizontally -> dead
        PerceptionTracker t;
        Vec3 p(-0.4f, 0.f, 0.785f);             // on table (z≈0.78), in own half
        Vec3 v(-1.0f, 0.f, 0.f);                // rolling toward robot, no vertical motion
        PTOutput o;
        for(int i=0;i<12;i++){ p+=v*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == false);                // dead ball -> not a live target
    }
    {   // resting: zero velocity -> dead
        PerceptionTracker t;
        Vec3 p(-0.5f, 0.2f, 0.78f);
        PTOutput o;
        for(int i=0;i<12;i++){ t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == false);
    }
    {   // going away (vx>0, post-hit) -> dead
        PerceptionTracker t;
        Vec3 p(-0.5f,0.f,1.0f), v(+4.0f,0.f,0.f);
        PTOutput o;
        for(int i=0;i<12;i++){ p+=v*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == false);
    }
    printf("  run_dead_ball OK\n");
}

void run_double_bounce(){
    PerceptionTracker t;
    // Simulate two bounces in the own half (x<0) without a paddle hit.
    // Bounce = vz goes negative (down) then positive (up) near table height.
    auto frame=[&](Vec3 p){ t.update(one(p),BASE,true,Vec3::Zero()); };
    // descend to 1st bounce
    frame(Vec3(-0.6f,0.f,1.0f)); frame(Vec3(-0.62f,0.f,0.85f)); frame(Vec3(-0.64f,0.f,0.78f));
    frame(Vec3(-0.66f,0.f,0.85f)); frame(Vec3(-0.68f,0.f,0.95f));   // up (bounce 1)
    frame(Vec3(-0.70f,0.f,0.85f)); frame(Vec3(-0.72f,0.f,0.78f));   // down again
    frame(Vec3(-0.74f,0.f,0.85f));                                  // up (bounce 2)
    // after the 2nd own-half bounce with no paddle hit, ball is dead
    for(int i=0;i<6;i++) frame(Vec3(-0.76f,0.f,0.9f));
    assert(t.output().live == false);
    printf("  run_double_bounce OK\n");
}

void run_volley_reject(){
    {   // flat fast ball that will NOT bounce in own half (flies long/over) -> not live
        PerceptionTracker t;
        Vec3 p(1.2f,0.f,1.05f), v(-10.0f,0.f,1.5f);  // hard rising drive -> first table contact x≈-3 (out, past own-half end)
        PTOutput o;
        for(int i=0;i<5;i++){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == false);   // volley/out -> robot must NOT engage
    }
    {   // normal serve that WILL bounce in own half (~x=-0.6) -> live
        PerceptionTracker t;
        Vec3 p(1.0f,0.f,1.0f), v(-4.0f,0.f,1.7f);
        PTOutput o;
        for(int i=0;i<5;i++){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == true);
    }
    printf("  run_volley_reject OK\n");
}

void run_hysteresis(){
    PerceptionTracker t;   // confirm_frames=3, coast_frames=8 by default
    Vec3 p(1.0f,0.f,1.0f), v(-4.0f,0.f,1.7f);
    auto step=[&](){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f; };
    // 1) a single live frame must NOT engage (needs K=3 consecutive)
    step(); t.update(one(p),BASE,true,Vec3::Zero());
    assert(t.output().engaged == false);
    // 2) after >=3 consecutive live frames, engage
    for(int i=0;i<4;i++){ step(); t.update(one(p),BASE,true,Vec3::Zero()); }
    assert(t.output().engaged == true);
    // 3) a brief 2-frame dropout must NOT disengage (coast M=8)
    t.update({},BASE,true,Vec3::Zero()); t.update({},BASE,true,Vec3::Zero());
    assert(t.output().engaged == true);
    printf("  run_hysteresis OK\n");
}

int main(){
    run_volume_gate();
    run_kf_smooth_and_coast();
    run_reflection_rejection();
    run_dead_ball();
    run_double_bounce();
    run_volley_reject();
    run_hysteresis();
    printf("ALL TESTS PASSED\n");
    return 0;
}
