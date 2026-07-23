#include <rbdl/rbdl.h>
#include <rbdl/addons/urdfreader/urdfreader.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kJoints = 7;

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            out.push_back(cell);
            cell.clear();
        } else {
            cell.push_back(c);
        }
    }
    out.push_back(cell);
    return out;
}

double parseDouble(const std::string& text) {
    if (text.empty() || text == "nan" || text == "NaN") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(text);
}

std::string num(double v) {
    if (!std::isfinite(v)) return "nan";
    std::ostringstream os;
    os << std::fixed << std::setprecision(9) << v;
    return os.str();
}

std::string getCell(
        const std::vector<std::string>& cells,
        const std::map<std::string, size_t>& index,
        const std::string& key) {
    const auto it = index.find(key);
    if (it == index.end() || it->second >= cells.size()) return "";
    return cells[it->second];
}

double getDouble(
        const std::vector<std::string>& cells,
        const std::map<std::string, size_t>& index,
        const std::string& key) {
    return parseDouble(getCell(cells, index, key));
}

std::string requireArg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    }
    return argv[++i];
}

std::array<std::string, kJoints> parseBodyNames(const std::string& text) {
    std::array<std::string, kJoints> names{};
    if (text.empty()) {
        for (int j = 0; j < kJoints; ++j) names[j] = "Link" + std::to_string(j + 1);
        return names;
    }
    std::istringstream is(text);
    for (int j = 0; j < kJoints; ++j) {
        if (!(is >> names[j])) {
            throw std::runtime_error("--body-names expects seven body names");
        }
    }
    return names;
}

void usage() {
    std::cerr
        << "Usage: rbdl_gravity_compare --input CSV --urdf A1/a1_r.urdf --output CSV\n"
        << "  [--gravity-x 9.81 --gravity-y 0 --gravity-z 0]\n"
        << "  [--torque-ff-scale '1.3 1.3 1.2 1 1 1 1']\n"
        << "  [--body-names 'Link1 Link2 Link3 Link4 Link5 Link6 Link7']\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input;
        std::string output;
        std::string urdf;
        std::string body_names_arg;
        Eigen::Vector3d gravity(9.81, 0.0, 0.0);
        std::array<double, kJoints> torque_scale{1.3, 1.3, 1.2, 1.0, 1.0, 1.0, 1.0};

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--input") {
                input = requireArg(i, argc, argv);
            } else if (arg == "--output") {
                output = requireArg(i, argc, argv);
            } else if (arg == "--urdf") {
                urdf = requireArg(i, argc, argv);
            } else if (arg == "--gravity-x") {
                gravity.x() = std::stod(requireArg(i, argc, argv));
            } else if (arg == "--gravity-y") {
                gravity.y() = std::stod(requireArg(i, argc, argv));
            } else if (arg == "--gravity-z") {
                gravity.z() = std::stod(requireArg(i, argc, argv));
            } else if (arg == "--torque-ff-scale") {
                std::istringstream is(requireArg(i, argc, argv));
                for (int j = 0; j < kJoints; ++j) {
                    if (!(is >> torque_scale[j])) {
                        throw std::runtime_error("--torque-ff-scale expects seven values");
                    }
                }
            } else if (arg == "--body-names") {
                body_names_arg = requireArg(i, argc, argv);
            } else if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        if (input.empty() || output.empty() || urdf.empty()) {
            usage();
            return 2;
        }

        RigidBodyDynamics::Model model;
        if (!RigidBodyDynamics::Addons::URDFReadFromFile(urdf.c_str(), &model, false)) {
            throw std::runtime_error("failed to load URDF: " + urdf);
        }
        model.gravity = gravity;
        if (model.q_size != kJoints) {
            std::cerr << "warning: RBDL q_size=" << model.q_size
                      << " expected " << kJoints << "\n";
        }
        const auto body_names = parseBodyNames(body_names_arg);
        std::array<unsigned int, kJoints> q_indices{};
        for (int j = 0; j < kJoints; ++j) {
            const unsigned int body_id = model.GetBodyId(body_names[j].c_str());
            if (body_id == std::numeric_limits<unsigned int>::max() ||
                body_id >= model.mJoints.size()) {
                throw std::runtime_error("body not found or fixed in RBDL model: " + body_names[j]);
            }
            const unsigned int q_index = model.mJoints[body_id].q_index;
            if (q_index >= static_cast<unsigned int>(model.q_size)) {
                throw std::runtime_error("invalid q_index for body: " + body_names[j]);
            }
            q_indices[j] = q_index;
            std::cerr << "body " << body_names[j] << " -> q_index " << q_index << "\n";
        }

        std::ifstream in(input);
        if (!in) throw std::runtime_error("failed to open input: " + input);
        std::ofstream out(output);
        if (!out) throw std::runtime_error("failed to open output: " + output);

        std::string header_line;
        if (!std::getline(in, header_line)) {
            throw std::runtime_error("empty input CSV: " + input);
        }
        const auto header = splitCsvLine(header_line);
        std::map<std::string, size_t> index;
        for (size_t i = 0; i < header.size(); ++i) index[header[i]] = i;

        out << "sample_id,ros_time_s,stage";
        for (int j = 0; j < kJoints; ++j) out << ",q_actual_" << j;
        for (int j = 0; j < kJoints; ++j) out << ",tau_rbdl_g_" << j;
        for (int j = 0; j < kJoints; ++j) out << ",tau_rbdl_g_scaled_" << j;
        for (int j = 0; j < kJoints; ++j) out << ",tau_isaac_g_" << j;
        for (int j = 0; j < kJoints; ++j) out << ",rbdl_minus_isaac_" << j;
        for (int j = 0; j < kJoints; ++j) out << ",rbdl_scaled_minus_isaac_" << j;
        for (int j = 0; j < kJoints; ++j) out << ",tau_ff_" << j;
        for (int j = 0; j < kJoints; ++j) out << ",tau_ff_minus_rbdl_scaled_" << j;
        out << "\n";

        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.q_size);
        Eigen::VectorXd qdot = Eigen::VectorXd::Zero(model.qdot_size);
        Eigen::VectorXd qddot = Eigen::VectorXd::Zero(model.qdot_size);
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(model.qdot_size);

        std::string line;
        size_t count = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const auto cells = splitCsvLine(line);
            q.setZero();
            for (int j = 0; j < kJoints; ++j) {
                q(q_indices[j]) = getDouble(cells, index, "q_actual_" + std::to_string(j));
            }
            tau.setZero();
            RigidBodyDynamics::InverseDynamics(model, q, qdot, qddot, tau);

            out << getCell(cells, index, "sample_id")
                << "," << getCell(cells, index, "ros_time_s")
                << "," << getCell(cells, index, "stage");
            for (int j = 0; j < kJoints; ++j) {
                out << "," << num(q(q_indices[j]));
            }
            for (int j = 0; j < kJoints; ++j) {
                out << "," << num(tau(q_indices[j]));
            }
            for (int j = 0; j < kJoints; ++j) {
                out << "," << num(torque_scale[j] * tau(q_indices[j]));
            }
            for (int j = 0; j < kJoints; ++j) {
                out << "," << getCell(cells, index, "tau_isaac_g_" + std::to_string(j));
            }
            for (int j = 0; j < kJoints; ++j) {
                const double isaac = getDouble(cells, index, "tau_isaac_g_" + std::to_string(j));
                out << "," << num(tau(q_indices[j]) - isaac);
            }
            for (int j = 0; j < kJoints; ++j) {
                const double isaac = getDouble(cells, index, "tau_isaac_g_" + std::to_string(j));
                out << "," << num(torque_scale[j] * tau(q_indices[j]) - isaac);
            }
            for (int j = 0; j < kJoints; ++j) {
                out << "," << getCell(cells, index, "tau_ff_" + std::to_string(j));
            }
            for (int j = 0; j < kJoints; ++j) {
                const double ff = getDouble(cells, index, "tau_ff_" + std::to_string(j));
                out << "," << num(ff - torque_scale[j] * tau(q_indices[j]));
            }
            out << "\n";
            ++count;
        }

        std::cerr << "wrote " << count << " rows to " << output << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
