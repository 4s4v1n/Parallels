/**
 * @file    gauss.cpp
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#include "gauss.h"

namespace s21 {

auto Gauss::getResultWithoutParallelAlgorithm() -> std::vector<double> {
  std::vector<double> result;
  result.resize(m_size);
  Matrix<double> tempMatrix = getMatrix();

  straighStroke(&tempMatrix);
  reverseStroke(&tempMatrix, &result);

  return result;
}

auto Gauss::straighStroke(Matrix<double> *inputMatrix) -> void {
  double factor;

  for (int k = 0; k < inputMatrix->cols() - 2; k++) {
    for (int i = 0; i < inputMatrix->rows() - 1 - k; i++) {
      if (inputMatrix->operator()(i + 1, k) != -inputMatrix->operator()(i, k)) {
        factor = -(inputMatrix->operator()(i, k) /
                   inputMatrix->operator()(i + 1, k));
      } else {
        factor = 1;
      }

      for (int j = 0; j < inputMatrix->cols(); j++) {
        inputMatrix->operator()(i, j) =
            inputMatrix->operator()(i, j) +
            inputMatrix->operator()(i + 1, j) * factor;
      }
    }
  }
}

auto Gauss::reverseStroke(Matrix<double> *inputMatrix,
                          std::vector<double> *result) -> void {
  for (int k = 0; k < inputMatrix->rows(); k++) {
    double sum = inputMatrix->operator()(k, inputMatrix->cols() - 1);

    for (int j = inputMatrix->rows() - 1, i = 0; i < k; j--, i++) {
      sum -= inputMatrix->operator()(k, j) * result->at(j);
    }

    if (inputMatrix->operator()(k, inputMatrix->rows() - k - 1) == 0 ||
        sum == 0) {
      result->at(inputMatrix->rows() - k - 1) = 0;
    } else {
      result->at(inputMatrix->rows() - k - 1) =
          sum / inputMatrix->operator()(k, inputMatrix->rows() - k - 1);
    }
  }
}

auto Gauss::getResultWithParallelAlgorithm() -> std::vector<double> {
  std::vector<double> result;
  result.resize(m_size);
  Matrix<double> tempMatrix = getMatrix();

  straighStrokeWithParallels(&tempMatrix);
  reverseStrokeWithParallels(&tempMatrix, &result);

  return result;
}

auto Gauss::straighStrokeWithParallels(Matrix<double> *inputMatrix) -> void {
  std::vector<std::thread> threads;
  auto function = [](int k, Matrix<double> *inputMatrix) {
    double factor;

    for (int i = 0; i < inputMatrix->rows() - 1 - k; i++) {
      if (inputMatrix->operator()(i + 1, k) != -inputMatrix->operator()(i, k)) {
        factor = -(inputMatrix->operator()(i, k) /
                   inputMatrix->operator()(i + 1, k));
      } else {
        factor = 1;
      }

      for (int j = 0; j < inputMatrix->cols(); j++) {
        inputMatrix->operator()(i, j) =
            inputMatrix->operator()(i, j) +
            inputMatrix->operator()(i + 1, j) * factor;
      }
    }
  };

  for (int k = 0; k < inputMatrix->rows() - 1; k++) {
    threads.push_back(std::thread(function, k, inputMatrix));
    threads.back().join();
  }
}

auto Gauss::reverseStrokeWithParallels(Matrix<double> *inputMatrix,
                                       std::vector<double> *result) -> void {
  auto function = [](int k, Matrix<double> *inputMatrix,
                     std::vector<double> *result) {
    double sum = inputMatrix->operator()(k, inputMatrix->cols() - 1);

    for (int j = inputMatrix->rows() - 1, i = 0; i < k; j--, i++) {
      sum -= inputMatrix->operator()(k, j) * result->at(j);
    }

    if (inputMatrix->operator()(k, inputMatrix->rows() - k - 1) == 0 ||
        sum == 0) {
      result->at(inputMatrix->rows() - k - 1) = 0;
    } else {
      result->at(inputMatrix->rows() - k - 1) =
          sum / inputMatrix->operator()(k, inputMatrix->rows() - k - 1);
    }
  };

  std::vector<std::thread> threads;

  for (int k = 0; k < inputMatrix->rows(); k++) {
    threads.push_back(std::thread(function, k, inputMatrix, result));
    threads.back().join();
  }
}

auto Gauss::loadMatrixFromFile(const std::string &path) -> void {
  std::ifstream fs;
  fs.open(path, std::fstream::in);
  if (!fs.is_open()) {
    throw std::runtime_error{"Incorrect file path"};
  }

  fs >> m_size;

  m_matrix.resize(m_size, m_size + 1);

  for (int i{0}; i < m_matrix.rows(); ++i) {
    for (int j{0}; j < m_matrix.cols(); ++j) {
      if (fs.peek() == std::ifstream::traits_type::eof()) {
        throw std::runtime_error{"Incorrect file"};
      }

      fs >> m_matrix(i, j);
    }
  }

  fs.close();
}

auto Gauss::getMatrix() -> Matrix<double> { return m_matrix; }
}  // namespace s21
