/**
 * @file    gauss.h
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include "../matrix/matrix.h"

namespace s21 {

/**
 * @class   Gauss
 * @brief   Defines class for the solution of linear equation by gauss method
 */

class Gauss {
 private:
  Matrix<double> m_matrix;
  size_t m_size = 0;

 public:
  Gauss() {}
  ~Gauss() {}

  auto loadMatrixFromFile(const std::string &path) -> void;
  auto getMatrix() -> Matrix<double>;
  auto getResultWithParallelAlgorithm() -> std::vector<double>;
  auto getResultWithoutParallelAlgorithm() -> std::vector<double>;

 private:
  auto straighStroke(Matrix<double> *inputMatrix) -> void;
  auto reverseStroke(Matrix<double> *inputMatrix, std::vector<double> *result)
      -> void;

  auto straighStrokeWithParallels(Matrix<double> *inputMatrix) -> void;
  auto reverseStrokeWithParallels(Matrix<double> *inputMatrix,
                                  std::vector<double> *result) -> void;
};

}  // namespace s21
