#pragma once
#include <deque>
#include <vector>
#include <string>
#include "isaaclab/algorithms/algorithms.h"   // isaaclab::OrtRunner

// Runs the learned ball-trajectory predictor (64-64 MLP) exported to ONNX.
// Maintains a ring buffer of the most recent H ball positions and, each step,
// feeds [p_{t-H} ... p_{t-1}] (oldest->newest, flattened) to predictor.onnx.
class TTPredictor {
public:
    TTPredictor(const std::string& onnx_path, int history_len = 5)
        : runner_(onnx_path), H_(history_len) {}

    // push newest ball position (x,y,z); returns predicted future ball point (3 floats)
    std::vector<float> update(const std::vector<float>& ball_xyz) {
        buf_.push_back(ball_xyz);
        while ((int)buf_.size() > H_) buf_.pop_front();
        // left-pad with the oldest available frame until we have H frames
        while ((int)buf_.size() < H_)
            buf_.push_front(buf_.empty() ? std::vector<float>{0.f, 0.f, 0.f} : buf_.front());
        std::vector<float> x;                       // oldest -> newest, flattened
        for (const auto& p : buf_) x.insert(x.end(), p.begin(), p.end());
        return runner_.act({{"ball_history", x}});
    }
private:
    isaaclab::OrtRunner runner_;
    std::deque<std::vector<float>> buf_;
    int H_;
};
