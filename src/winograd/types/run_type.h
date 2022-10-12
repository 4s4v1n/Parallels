/**
 * @file    run_type.h
 * @authors telvina<telvina<at>student.21-school.ru>
 * @authors sreanna<sreanna<at>student.21-school.ru>
 * @authors jmadie<jmadie<at>student.21-school.ru>
 * @date    02/10/22
 */

#ifndef RUN_TYPE_H
#define RUN_TYPE_H

namespace s21 {

/**
 * @enum    RunType
 * @brief   Defines type of current thread model for Winograd class
 */

enum class RunType : std::uint8_t {
  SINGLE = 0,
  MULTITHREADING_CLASSIC,
  MULTITHREADING_NUMBER,
  MULTITHREADING_CONVEYOR
};

}  // namespace s21

#endif  // RUN_TYPE_H
