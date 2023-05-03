#pragma once

#include <sstream>

using PyLogFn = void (*)(const char* level, const char* s);

struct PyLog {
  static PyLog& instance() {
    static PyLog pylog;
    return pylog;
  }

  void initialize(PyLogFn log_fn) { log_fn_ = log_fn; }

  static void INFO(const char* msg) { log("info", msg); }

  static void WARNING(const char* msg) { log("warning", msg); }

  static void ERROR(const char* msg) { log("error", msg); }

  PyLog(const PyLog&) = delete;
  PyLog& operator=(const PyLog&) = delete;

 private:
  static void log(const char* level, const char* msg) {
    auto log_fn = instance().log_fn_;
    if (log_fn != nullptr) {
      log_fn(level, msg);
    }
  }
  PyLog() {}

  PyLogFn log_fn_ = nullptr;
};

struct PyLogIf {
  PyLogIf() {}

  template <typename T>
  PyLogIf& operator<<(const T& value) {
    stream_ << value;
    return *this;
  }
  ~PyLogIf() { PyLog::INFO(stream_.str().c_str()); }

 private:
  std::ostringstream stream_;
};

#define PYLOG PyLogIf()