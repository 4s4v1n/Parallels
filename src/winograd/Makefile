CXX=g++
CXX_FLAGS=-std=c++17 -pthread
INL=matrix/matrix.inl
HDR=matrix/matrix.h       \
	types/run_type.h      \
	algorithms/winograd.h \
	console_interface/winograd_inteface.h
SRC=algorithms/winograd.cpp                  \
	console_interface/winograd_interface.cpp \
	winograd_main.cpp

.PHONY: winograd clean

winograd:
	$(CXX) $(CXX_FLAGS) $(SRC) -o ../winograd.out

clean:
	rm -rf *.o *.out
