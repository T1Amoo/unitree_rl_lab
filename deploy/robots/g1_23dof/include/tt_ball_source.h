#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

struct TTPerception { float ball_pos[3]; float robot_pos[3]; float heading; };

// Abstract source of ball + robot-base perception, per control step.
class TTBallSource {
public:
    virtual ~TTBallSource() = default;
    virtual TTPerception get(long step) = 0;
};

// Replays perception from a CSV with header: bx,by,bz,rx,ry,rz,heading
class ReplayBallSource : public TTBallSource {
public:
    explicit ReplayBallSource(const std::string& csv) {
        std::ifstream f(csv);
        std::string line;
        std::getline(f, line); // skip header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            TTPerception p; char c;
            ss >> p.ball_pos[0] >> c >> p.ball_pos[1] >> c >> p.ball_pos[2] >> c
               >> p.robot_pos[0] >> c >> p.robot_pos[1] >> c >> p.robot_pos[2] >> c >> p.heading;
            rows_.push_back(p);
        }
    }
    TTPerception get(long step) override {
        if (rows_.empty()) return TTPerception{};
        return rows_[std::min<long>(step, (long)rows_.size() - 1)];
    }
    size_t size() const { return rows_.size(); }
private:
    std::vector<TTPerception> rows_;
};
