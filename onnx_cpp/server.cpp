// server/main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "httplib.h"               // single-header cpp-httplib
#include "json.hpp"                // nlohmann::json single-header
#include <onnxruntime_cxx_api.h>

using json = nlohmann::json;
using namespace httplib;

static std::string MODEL_PATH = "/models/mnist_classifier.onnx";
static const char* LISTEN_ADDR = "0.0.0.0";
static const int LISTEN_PORT = 8000;

bool nested_list_to_flat(const json &j, std::vector<float> &out, std::vector<int64_t> &shape) {
    // Determine nested depth and sizes recursively
    json cur = j;
    shape.clear();
    // Determine shape by walking nested arrays
    while (cur.is_array()) {
        shape.push_back((int64_t)cur.size());
        if (cur.size() == 0) break;
        cur = cur[0];
    }
    // Compute total size
    int64_t total = 1;
    for (auto d : shape) total *= d;
    out.reserve(total);

    // Flatten via recursive lambda
    std::function<void(const json&)> flatten = [&](const json &v) {
        if (v.is_array()) {
            for (auto &el : v) flatten(el);
        } else if (v.is_number()) {
            out.push_back(v.get<float>());
        } else {
            // unsupported type
            throw std::runtime_error("Unsupported JSON type inside data array");
        }
    };

    flatten(j);
    if ((int64_t)out.size() != total) {
        std::cerr << "Warning: flattened size != computed total. flattened=" << out.size() << " total=" << total << "\n";
    }
    return true;
}

int main(int argc, char** argv) {
    // Optional: allow override model path from argv
    if (argc > 1) MODEL_PATH = argv[1];

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_cpp_server");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Create Session
    std::cerr << "Loading model: " << MODEL_PATH << std::endl;
    Ort::Session session(env, MODEL_PATH.c_str(), session_options);

    // Get IO names & info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    for (size_t i = 0; i < num_inputs; ++i) {
        char* name = session.GetInputNameAllocated(i, allocator).release();
        input_names.emplace_back(name);
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        char* name = session.GetOutputNameAllocated(i, allocator).release();
        output_names.emplace_back(name);
    }

    std::cerr << "Input names: ";
    for (auto &n : input_names) std::cerr << n << " ";
    std::cerr << "\nOutput names: ";
    for (auto &n : output_names) std::cerr << n << " ";
    std::cerr << std::endl;

    // Start HTTP server
    Server svr;

    svr.Post("/infer", [&](const Request& req, Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"empty body\"}", "application/json");
                return;
            }

            json j = json::parse(req.body);
            if (!j.contains("data")) {
                res.status = 400;
                res.set_content("{\"error\":\"missing 'data' field\"}", "application/json");
                return;
            }

            json data = j["data"];

            // Convert nested list -> flat vector and get shape
            std::vector<float> input_flat;
            std::vector<int64_t> input_shape;
            nested_list_to_flat(data, input_flat, input_shape);

            if (input_shape.size() == 3) {
                // maybe user sent [1,28,28] => add channel dimension
                // expected ONNX input: [batch, channels, H, W]
                // if shape == [batch, 1, 28, 28] that's 4 dims. If user sends [1,28,28] assume batch=1 and add channel=1
                // but here shape==3 -> interpret as [batch, H, W] and convert to [batch,1,H,W] if sensible.
                // We'll adopt: if last two dims are 28,28 -> insert channel=1 at pos 1
                if (input_shape[1] == 28 && input_shape[2] == 28) {
                    input_shape.insert(input_shape.begin()+1, 1);
                } else {
                    // possible other format - treat as [batch, channels, H] (odd). We'll let ONNX fail if mismatch.
                }
            } else if (input_shape.size() == 2) {
                // maybe single image [1, 28*28] -> convert to [1,1,28,28] if possible
                if (input_shape[1] == 28*28) {
                    input_shape = { input_shape[0], 1, 28, 28 };
                }
            }

            // If the outermost dimension is not batch (user maybe sent without batch), try to guess
            // Ensure at least 4D for MNIST model
            if (input_shape.size() == 3) {
                // insert batch dim 1
                input_shape.insert(input_shape.begin(), 1);
            }

            // Prepare input shape as int64_t vector
            std::vector<int64_t> onnx_shape = input_shape;

            // Create Ort tensor
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

            // Check sizes match
            int64_t total = 1;
            for (auto d : onnx_shape) total *= d;
            if ((int64_t)input_flat.size() != total) {
                // mismatch -> respond error
                json jr;
                jr["error"] = "input size mismatch";
                jr["expected_elements"] = total;
                jr["received_elements"] = input_flat.size();
                res.status = 400;
                res.set_content(jr.dump(), "application/json");
                return;
            }

            // Input name pointer arr
            std::vector<const char*> c_input_names;
            for (auto &n : input_names) c_input_names.push_back(n.c_str());
            std::vector<const char*> c_output_names;
            for (auto &n : output_names) c_output_names.push_back(n.c_str());

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info,
                input_flat.data(),
                input_flat.size(),
                onnx_shape.data(),
                onnx_shape.size()
            );

            // Run
            auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                              c_input_names.data(), &input_tensor, 1,
                                              c_output_names.data(), c_output_names.size());

            // Expect single output (N, classes)
            if (output_tensors.size() == 0) {
                res.status = 500;
                res.set_content("{\"error\":\"no outputs from model\"}", "application/json");
                return;
            }

            float* out_data = output_tensors[0].GetTensorMutableData<float>();
            auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto out_shape = out_info.GetShape();
            int64_t out_elements = 1;
            for (auto d : out_shape) out_elements *= d;

            // pack into json
            json jr;
            jr["shape"] = out_shape;

            // convert to nested list by batch
            int64_t batch = out_shape.size() > 0 ? out_shape[0] : 1;
            int64_t classes = out_shape.size() > 1 ? out_shape[1] : out_elements;
            jr["pred"] = json::array();

            for (int64_t b = 0; b < batch; ++b) {
                json row = json::array();
                for (int64_t c = 0; c < classes; ++c) {
                    row.push_back(out_data[b * classes + c]);
                }
                jr["pred"].push_back(row);
            }

            res.status = 200;
            res.set_content(jr.dump(), "application/json");
            return;
        } catch (const std::exception &e) {
            json jr;
            jr["error"] = e.what();
            res.status = 500;
            res.set_content(jr.dump(), "application/json");
            return;
        }
    });

    std::cerr << "Server listening on " << LISTEN_ADDR << ":" << LISTEN_PORT << std::endl;
    svr.listen(LISTEN_ADDR, LISTEN_PORT);

    return 0;
}
