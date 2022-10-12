.PHONY: all, clean, tests, gauss, clang
XX=g++
FLAGS= -Wall -Wextra -Werror -std=c++17 -pthread
FILES= gauss.cpp \
	   gauss_interface.cpp \

all: clean gauss

clean:
	rm -rf *.o *.out *.cfg *.gch *.dSYM *.gcda *.gcno *.info *.html .clang-format report

gauss: clean
	$(XX) $(FLAGS) main.cpp $(FILES) -o ../gauss.out

clang:
	cp ../materials/linters/.clang-format .clang-format
	clang-format -n */*.cpp */*.h */*.inl

cppcheck:
	cppcheck -q --std=c++17 --enable=all --language=c++ */*.cpp */*.h

valgrind:
	valgrind --trace-children=yes --track-fds=yes --track-origins=yes --leak-check=full --show-leak-kinds=all -s ./gaussInterface.o