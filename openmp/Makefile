CXX=CC #mpic++
BIN=bin
OBJ=obj
SRC=src
IDIR=include
TESTS=tests
CXXFLAGS= -O3 -g -fopenmp
LIBPATH=/software/projects/pawsey0001/cdipietrantonio/kmeans/cpp
.PHONY : directories all tests clean

all : directories tests

clean :
	rm -r obj
	rm -r bin

directories :
	[ -d $(OBJ) ] || mkdir $(OBJ)
	[ -d $(BIN) ] || mkdir $(BIN)
	
tests : $(BIN)/tests

_DEPS = kmeans.hpp
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

$(BIN)/tests : $(TESTS)/tests.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(OBJ)/%.o : $(SRC)/%.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run_test : $(BIN)/tests
	./$(BIN)/tests
