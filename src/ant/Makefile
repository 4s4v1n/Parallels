CXX=g++
OS=$(shell uname -s)

ifeq ($(OS), Linux)
	LINUX_FLAGS=-lpthread -lcheck_pic -lrt -lm -lsubunit
endif

CFLAGS=#-Wall -Werror -Wextra

SOURCES= travelling_salesman_problem.cpp \
         graph/s21_graph.cpp view/*.cpp \
		 ants_colony/*.cpp

.PHONY: all clean ant

ant:
	$(CXX) $(CFLAGS) $(SOURCES) -o ../ant.out $(LINUX_FLAGS)

cppcheck:
	cppcheck --enable=all --language=c++ --suppress=unusedFunction --suppress=missingInclude $(SOURCES)

clean:
	rm -rf *.o *.out