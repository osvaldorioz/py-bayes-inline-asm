// Bayes.cpp
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

//g++ -O2 -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` bayes.cpp -o naive_bayes`python3.12-config --extension-suffix`
//c++ -O3 -Wall -shared -std=c++17 -fPIC `python3.12 -m pybind11 --includes` bayes.cpp -o naive_bayes`python3.12-config --extension-suffix`

namespace py = pybind11;

// Función en ensamblador para realizar multiplicaciones
/*extern "C" double asm_multiply(double a, double b);
__asm__(
    ".global asm_multiply\n"
    "asm_multiply:\n"
    "    fldl 8(%esp)\n"      // Cargar b
    "    fldl 4(%esp)\n"      // Cargar a
    "    fmul %st(1), %st\n"   // Multiplicar
    "    fstp %st(1)\n"        // Guardar resultado
    "    ret\n"
);*/

// Función en ensamblador para realizar multiplicaciones ver. 64bits
/*extern "C" double asm_multiply(double a, double b);
__asm__(
    ".global asm_multiply\n"
    "asm_multiply:\n"
    "   movsd  xmm0, qword ptr [rdi]\n"      // Cargar el primer argumento (a) en xmm0
    "   movsd  xmm1, qword ptr [rsi]\n"      // Cargar el segundo argumento (b) en xmm1
    "   mulsd  xmm0, xmm1           \n"      // Multiplicar xmm0 por xmm1
    "   movsd  qword ptr [rax], xmm0\n"      // Guardar el resultado en rax
    "   ret\n"
);*/

// Función en ensamblador para realizar sumas
/*extern "C" double asm_add(double a, double b);
__asm__(
    ".global asm_add\n"
    "asm_add:\n"
    "    fldl 8(%esp)\n"      // Cargar b
    "    fldl 4(%esp)\n"      // Cargar a
    "    fadd %st(1), %st\n"   // Sumar
    "    fstp %st(1)\n"        // Guardar resultado
    "    ret\n"
);*/

// Función en ensamblador para realizar sumas version 64 bits
/*extern "C" double asm_add(double a, double b);
__asm__(
    ".global asm_add\n"
    "asm_add:\n"
    "   movsd  xmm0, qword ptr [rdi]\n"        // Cargar el primer argumento (a) en xmm0
    "   movsd  xmm1, qword ptr [rsi]\n"        // Cargar el segundo argumento (b) en xmm1
    "   addsd  xmm0, xmm1           \n"        // Sumar xmm0 y xmm1
    "   movsd  qword ptr [rax], xmm0\n"        // Guardar el resultado en rax
    "   ret\n"
);*/

inline double asm_multiply(double a, double b) {
    return a * b;
}

inline double asm_add(double a, double b) {
    return a + b;
}


class NaiveBayes {
public:
    void fit(const std::vector<std::vector<std::string>>& X, const std::vector<std::string>& y) {
        for (size_t i = 0; i < X.size(); ++i) {
            std::string label = y[i];
            label_count[label]++;

            for (size_t j = 0; j < X[i].size(); ++j) {
                feature_count[label][X[i][j]]++;
            }
        }

        total_samples = X.size();
    }

    std::map<std::string, double> predict_proba(const std::vector<std::string>& X) {
        std::map<std::string, double> probabilities;

        for (const auto& label : label_count) {
            double label_prob = label.second / static_cast<double>(total_samples);
            double prob = label_prob;

            for (const auto& feature : X) {
                double feature_prob = (feature_count[label.first][feature] + 1.0) / (label_count[label.first] + feature_count[label.first].size());
                prob = asm_multiply(prob, feature_prob);
            }

            probabilities[label.first] = prob;
        }

        // Normalizar probabilidades
        double total_prob = 0.0;
        for (const auto& prob : probabilities) {
            total_prob = asm_add(total_prob, prob.second);
        }

        for (auto& prob : probabilities) {
            prob.second /= total_prob;
        }

        return probabilities;
    }

private:
    std::map<std::string, int> label_count;
    std::map<std::string, std::map<std::string, int>> feature_count;
    int total_samples = 0;
};

PYBIND11_MODULE(naive_bayes, m) {
    py::class_<NaiveBayes>(m, "NaiveBayes")
        .def(py::init<>())
        .def("fit", &NaiveBayes::fit)
        .def("predict_proba", &NaiveBayes::predict_proba);
}
