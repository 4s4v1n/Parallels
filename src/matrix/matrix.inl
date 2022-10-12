/**
 * @file   matrix.inl
 * @author Savin A. S. <sav1nru<at>mail.ru>
 * @date   18/08/22
 */

#include <random>
#include <type_traits>

#include "matrix.h"

template <class T>
Matrix<T>::Matrix() : m_rows{standard_size}, m_cols{standard_size} {
  m_matrix = new T*[m_rows];
  for (int i{0}; i < m_rows; ++i) {
    m_matrix[i] = new T[m_cols]{};
  }
}

template <class T>
Matrix<T>::Matrix(const int rows, const int cols) {
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument{"Invalid size of matrix"};
  }
  m_rows = rows;
  m_cols = cols;
  m_matrix = new T*[m_rows];
  for (int i{0}; i < m_rows; ++i) {
    m_matrix[i] = new T[m_cols]{};
  }
}

template <class T>
Matrix<T>::~Matrix() {
  memoryFree();
}

template <class T>
Matrix<T>::Matrix(const Matrix& other) : Matrix{other.m_rows, other.m_cols} {
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      m_matrix[i][j] = other.m_matrix[i][j];
    }
  }
}

template <class T>
Matrix<T>::Matrix(Matrix&& other) noexcept {
  if (this != &other) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_matrix = other.m_matrix;
    other.m_matrix = nullptr;
  }
}

template <class T>
auto Matrix<T>::newMatrixWithoutRowAndCol(const Matrix& other, const int row,
                                          const int col) -> void {
  int offsetRow{0};
  for (int i{0}; i < other.m_rows - 1; ++i) {
    if (i == row) {
      offsetRow = 1;
    }
    int offsetCol{0};
    for (int j{0}; j < other.m_cols - 1; ++j) {
      if (j == col) {
        offsetCol = 1;
      }
      m_matrix[i][j] = other.m_matrix[i + offsetRow][j + offsetCol];
    }
  }
}

template <class T>
auto Matrix<T>::memoryFree() -> void {
  if (m_matrix != nullptr) {
    for (int i{0}; i < m_rows; ++i) {
      delete[] m_matrix[i];
    }
    delete[] m_matrix;
    m_matrix = nullptr;
    m_cols = 0;
    m_rows = 0;
  }
}

template <class T>
auto Matrix<T>::rows() const -> int {
  return m_rows;
}

template <class T>
auto Matrix<T>::cols() const -> int {
  return m_cols;
}

template <class T>
auto Matrix<T>::eq_matrix(const Matrix& other) const -> bool {
  if (m_rows != other.m_rows || m_cols != other.m_rows) {
    return false;
  }
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      if (m_matrix[i][j] != other.m_matrix[i][j]) {
        return false;
      }
    }
  }
  return true;
}

template <class T>
auto Matrix<T>::sum_matrix(const Matrix& other) -> void {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    throw std::logic_error{"Size of matrices are not the same"};
  }
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      m_matrix[i][j] += other.m_matrix[i][j];
    }
  }
}

template <class T>
auto Matrix<T>::sub_matrix(const Matrix& other) -> void {
  if (m_rows != other.m_rows || m_cols != other.m_cols) {
    throw std::logic_error{"Size of matrices are not the same"};
  }
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      m_matrix[i][j] -= other.m_matrix[i][j];
    }
  }
}

template <class T>
auto Matrix<T>::mul_number(const double num) -> void {
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      m_matrix[i][j] *= num;
    }
  }
}

template <class T>
auto Matrix<T>::calc_complements() const -> Matrix {
  if (m_rows != m_cols) {
    throw std::logic_error{"Matrix is not square"};
  }
  if (m_rows == 1) {
    throw std::logic_error{"It's not a matrix, it's scalar"};
  }
  Matrix A{*this};
  Matrix B{m_rows - 1, m_cols - 1};
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      B.newMatrixWithoutRowAndCol(*this, i, j);
      A.m_matrix[i][j] = B.determinant() * pow(-1, i + j + 2);
    }
  }
  return A;
}

template <class T>
auto Matrix<T>::transpose() const -> Matrix {
  Matrix A{m_cols, m_rows};
  for (int i{0}; i < A.m_rows; ++i) {
    for (int j{0}; j < A.m_cols; ++j) {
      A.m_matrix[i][j] = m_matrix[j][i];
    }
  }
  return A;
}

template <class T>
auto Matrix<T>::mul_matrix(const Matrix& other) -> void {
  if (m_cols != other.m_rows) {
    throw std::invalid_argument{"Rows from 1st matrix != cols from 2nd matrix"};
  }
  Matrix A{m_rows, other.m_cols};
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < other.m_cols; ++j) {
      for (int k{0}; k < m_cols; ++k) {
        A.m_matrix[i][j] += m_matrix[i][k] * other.m_matrix[k][j];
      }
    }
  }
  memoryFree();
  m_matrix = new T*[A.m_rows];
  for (int i{0}; i < A.m_rows; ++i) {
    m_matrix[i] = new T[A.m_cols]{};
  }
  *this = A;
}

template <class T>
auto Matrix<T>::determinant() const -> double {
  if (m_rows != m_cols) {
    throw std::logic_error{"Matrix is not square"};
  }
  double det{0.};
  if (m_rows == 1) {
    det = m_matrix[0][0];
  } else if (m_rows == 2) {
    det = m_matrix[0][0] * m_matrix[1][1] - m_matrix[0][1] * m_matrix[1][0];
  } else {
    int degree{1};
    Matrix A{m_rows - 1, m_cols - 1};
    for (int i{0}; i < m_rows; ++i) {
      A.newMatrixWithoutRowAndCol(*this, 0, i);
      det += degree * m_matrix[0][i] * A.determinant();
      degree = -degree;
    }
  }
  return det;
}

template <class T>
auto Matrix<T>::inverse_matrix() const -> Matrix {
  if (determinant() == 0) {
    throw std::logic_error{"Determinant is equal 0"};
  }
  Matrix A{m_rows, m_cols};
  if (m_rows == 1) {
    A.m_matrix[0][0] = 1. / m_matrix[0][0];
  } else {
    Matrix B{calc_complements()};
    B = B.transpose();
    B *= 1. / determinant();
    A = B;
  }
  return A;
}

template <class T>
auto Matrix<T>::resize(const int rows, const int cols) -> void {
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument{"Wrong size"};
  }
  if (rows != m_rows || cols != m_cols) {
    for (int i{0}; i < m_rows; ++i) {
      delete[] m_matrix[i];
    }
    delete[] m_matrix;
    m_rows = rows;
    m_cols = cols;
    m_matrix = new T*[m_rows];
    for (int i{0}; i < m_rows; ++i) {
      m_matrix[i] = new T[m_cols]{};
    }
  }
}

template <class T>
auto Matrix<T>::clear() -> void {
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      m_matrix[i][j] = 0;
    }
  }
}

template <class T>
auto Matrix<T>::operator+(const Matrix<T>& rhs) -> Matrix<T> {
  Matrix lhs{*this};
  lhs.sum_matrix(rhs);
  return lhs;
}

template <class T>
auto Matrix<T>::operator-(const Matrix<T>& rhs) -> Matrix<T> {
  Matrix lhs{*this};
  lhs.sub_matrix(rhs);
  return lhs;
}

template <class T>
auto Matrix<T>::operator*(const Matrix<T>& rhs) -> Matrix<T> {
  Matrix lhs{*this};
  lhs.mul_matrix(rhs);
  return lhs;
}

template <class T>
auto Matrix<T>::operator*(const double rhs) -> Matrix<T> {
  Matrix lhs{*this};
  lhs.mul_number(rhs);
  return lhs;
}

template <class T>
auto Matrix<T>::operator==(const Matrix<T>& rhs) -> bool {
  return this->eq_matrix(rhs);
}

template <class T>
auto Matrix<T>::operator!=(const Matrix<T>& rhs) -> bool {
  return !this->eq_matrix(rhs);
}

template <class T>
auto Matrix<T>::operator+=(const Matrix<T>& rhs) -> void {
  this->sum_matrix(rhs);
}

template <class T>
auto Matrix<T>::operator-=(const Matrix<T>& rhs) -> void {
  this->sub_matrix(rhs);
}

template <class T>
auto Matrix<T>::operator*=(const Matrix<T>& rhs) -> void {
  this->mul_matrix(rhs);
}

template <class T>
auto Matrix<T>::operator*=(const double rhs) -> void {
  this->mul_number(rhs);
}

template <class T>
T& Matrix<T>::operator()(const int i, const int j) {
  if (i < 0 || j < 0 || i >= m_rows || j >= m_cols) {
    throw std::invalid_argument{"Index outside matrix"};
  }
  return m_matrix[i][j];
}

template <class T>
const T& Matrix<T>::operator()(const int i, const int j) const {
  if (i < 0 || j < 0 || i >= m_rows || j >= m_cols) {
    throw std::invalid_argument{"Index outside matrix"};
  }
  return m_matrix[i][j];
}

template <class T>
auto Matrix<T>::operator=(const Matrix& other) -> Matrix& {
  if (this == &other) {
    throw std::invalid_argument{"Self copy"};
  }
  memoryFree();
  m_rows = other.m_rows;
  m_cols = other.m_cols;
  m_matrix = new T*[m_rows];
  for (int i{0}; i < m_rows; ++i) {
    m_matrix[i] = new T[m_cols]{};
  }
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      m_matrix[i][j] = other.m_matrix[i][j];
    }
  }
  return *this;
}

template <class T>
auto Matrix<T>::fill(const T& value) -> void {
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      m_matrix[i][j] = value;
    }
  }
}

template <class T>
auto Matrix<T>::rand(const double min, const double max) -> void {
  if (min >= max) {
    throw std::invalid_argument{"Wrong range"};
  }
  std::random_device random_device;
  std::mt19937 random_generator{random_device()};
  std::uniform_real_distribution<double> distribution{min, max};
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      m_matrix[i][j] = distribution(random_generator);
    }
  }
}

template <class T>
auto Matrix<T>::print() -> void {
  for (int i{0}; i < m_rows; ++i) {
    for (int j{0}; j < m_cols; ++j) {
      std::cout << m_matrix[i][j] << " ";
    }

    std::cout << std::endl;
  }
}
