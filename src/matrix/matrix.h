/**
 * @file   matrix.h
 * @author Savin A. S. <sav1nru<at>mail.ru>
 * @date   18/08/22
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

/**
 * @class Matrix
 * @brief Realization of matrix on cpp language
 */

template <class T>

class Matrix final {
 private:
  typedef const T& const_reference;

 public:
  Matrix();
  Matrix(int rows, int cols);
  Matrix(const Matrix& other);
  Matrix(Matrix&& other) noexcept;
  ~Matrix();
  auto operator=(const Matrix& other) -> Matrix&;

 public:
  auto sum_matrix(const Matrix& other) -> void;
  auto sub_matrix(const Matrix& other) -> void;
  auto mul_number(double num) -> void;
  auto mul_matrix(const Matrix& other) -> void;
  auto resize(int rows, int cols) -> void;
  auto clear() -> void;
  auto fill(const T& value) -> void;
  auto rand(double min, double max) -> void;
  auto print() -> void;
  [[nodiscard]] auto eq_matrix(const Matrix& other) const -> bool;
  [[nodiscard]] auto transpose() const -> Matrix;
  [[nodiscard]] auto calc_complements() const -> Matrix;
  [[nodiscard]] auto inverse_matrix() const -> Matrix;
  [[nodiscard]] auto determinant() const -> double;

 public:
  [[nodiscard]] auto rows() const -> int;
  [[nodiscard]] auto cols() const -> int;
  auto operator()(int i, int j) -> T&;
  const_reference operator()(int i, int j) const;
  auto operator+(const Matrix<T>& rhs) -> Matrix<T>;
  auto operator-(const Matrix<T>& rhs) -> Matrix<T>;
  auto operator*(const Matrix<T>& rhs) -> Matrix<T>;
  auto operator*(double rhs) -> Matrix<T>;
  auto operator==(const Matrix<T>& rhs) -> bool;
  auto operator!=(const Matrix<T>& rhs) -> bool;
  auto operator+=(const Matrix<T>& rhs) -> void;
  auto operator-=(const Matrix<T>& rhs) -> void;
  auto operator*=(const Matrix<T>& rhs) -> void;
  auto operator*=(double rhs) -> void;

 private:
  auto newMatrixWithoutRowAndCol(const Matrix& other, int row, int col) -> void;
  auto memoryFree() -> void;

 private:
  static constexpr int standard_size = 3;

 private:
  int m_rows{};
  int m_cols{};
  T** m_matrix{};
};

#include "matrix.inl"

#endif  // MATRIX_H
