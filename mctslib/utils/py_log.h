#pragma once

using PyLogFn = void (*)(const char* level, const char* s);

struct PyLog {
  static PyLog& instance() {
    static PyLog pylog;
    return pylog;
  }

  void initialize(PyLogFn log_fn) { log_fn_ = log_fn; }

  static void INFO(const char* str) {
    auto log_fn = instance().log_fn_;
    if (log_fn != nullptr) {
      log_fn("info", str);
    }
  }

  PyLog(const PyLog&) = delete;
  PyLog& operator=(const PyLog&) = delete;

 private:
  PyLog() {}

  PyLogFn log_fn_ = nullptr;
};