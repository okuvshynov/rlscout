#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <vector>
#include <iostream>

const int k = 5;

struct Generator {
  int64_t m, n;
  std::vector<std::set<uint64_t>> patterns;

  Generator(uint64_t m, uint64_t n) : m(m), n(n) {
    patterns.resize(m * n);
  }

  uint64_t index(uint64_t x, uint64_t y) {  return x * n + y; }
  uint64_t mask(uint64_t x, uint64_t y)  {  return (1ULL << index(x, y)); }
  bool is_valid_x(int x) { return x >= 0 && x < m; }
  bool is_valid_y(int y) { return y >= 0 && y < n; }

  void append_if_valid(int x, int y, int dx, int dy) {
    uint64_t mm = 0LL;
    for (int kk = 0; kk < k; kk++) {
      int xx = x + dx * kk;
      int yy = y + dy * kk;
      if (!is_valid_x(xx) || !is_valid_y(yy)) {
        return;
      }
      mm |= (mask(xx, yy));
    }
    for (int kk = 0; kk < k; kk++) {
      int xx = x + dx * kk;
      int yy = y + dy * kk;
      patterns[index(xx, yy)].insert(mm);
    }
  }

  void run() {
    for (int x = 0; x < m; x++) {
      for (int y = 0; y < n; y++) {
        for (int dx = -1; dx < 2; dx++) {
          for (int dy = -1; dy < 2; dy++) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            append_if_valid(x, y, dx, dy);
          }
        }
      }
    }
  }

  void print_check(uint64_t mask) {
    std::cout << "((" << mask << "ULL & board) == " << mask << "ULL)";
  }

  void print() {
    std::cout << "template<>" << std::endl;
    std::cout << "bool is_winning<" << m << "," << n << "," << k << ">(uint64_t index, uint64_t board) {" << std::endl;
    std::cout << "  switch (index) {" << std::endl; 

    for (int i = 0; i < m * n; i++) {
      std::cout << "    case " << i << ":" << std::endl;
      std::cout << "      return ";
        bool f = false;
        for (uint64_t m : patterns[i]) {
          if (f) std::cout << " || ";
          f = true;
          print_check(m);
        }
      std::cout << ";" << std::endl;
      }
      
      std::cout << "    default: return false;" << std::endl;
      std::cout << "  };" << std::endl;
      std::cout << "  return false;" << std::endl;
      std::cout << "}" << std::endl;
    }
};

int main(int argc, char **argv) {
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  
  auto g = Generator(m, n);
  g.run();
  g.print();

  return 0;
}
