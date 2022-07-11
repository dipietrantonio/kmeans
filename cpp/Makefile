CXX=CC #mpic++
BIN=bin
OBJ=obj
SRC=src
IDIR=include
TESTS=tests
CXXFLAGS= -O2 -g
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
	$(CXX) $(CXXFLAGS) -o $@ $< # -L$(LIBPATH)/obj -lmap-sampler-pmpi -lmap-sampler -Wl,--eh-frame-hdr -Wl,-rpath=$(LIBPATH)/obj -lmpi

$(OBJ)/%.o : $(SRC)/%.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run_test : $(BIN)/tests
	./$(BIN)/tests
