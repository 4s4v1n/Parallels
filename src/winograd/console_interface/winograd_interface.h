/**
 * @file    winograd_interface.h
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#ifndef CONSOLE_INTERFACE_H
#define CONSOLE_INTERFACE_H

#include <string>

#include "../algorithms/winograd.h"

namespace s21 {

/**
 * @class   Winograd
 * @brief   Defines class with console interface for winograd class
 */

class WinogradInterface {
 public:
  WinogradInterface() = default;
  ~WinogradInterface() = default;

 public:
  auto show() -> void;

 private:
  auto mainMenu() -> int;
  auto setMatrices() -> void;
  auto setRepeatNumber() -> void;
  auto runCase(const std::string& message, RunType type) -> void;
  auto runAllCases() -> void;
  auto showResult(Matrix<double>& result) -> void;
  auto printError(const std::exception& error) -> void;

 private:
  Winograd m_winograd{};
  std::uint32_t m_nRepeat{};
  bool m_bMatricesSet{};
};

}  // namespace s21

#endif  // CONSOLE_INTERFACE_H
