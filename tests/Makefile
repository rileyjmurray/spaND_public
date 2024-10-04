include Makefile.conf

INCDIR = ../include
SRCDIR = ../src
OBJDIR = ../obj
BUILDDIR = ../build

## Include folder
INCLUDE = -I$(INCDIR) $(REQUIRED_INCDIRS)

## Link folder
LINK = $(REQUIRED_LIBDIRS)

## Compiler flags
CFLAGS = -std=c++20
ifdef DEBUG
    CFLAGS += -g -O0 -D_GLIBCXX_DEBUG
else
    CFLAGS += -O3 -DEIGEN_NO_DEBUG
endif
CFLAGS += $(EXTRA_CFLAGS)

## Linker flags
LDFLAGS = -lmetis -lblaspp -llapackpp

DEPS = $(INCDIR)/tree.h $(INCDIR)/util.h $(INCDIR)/is.h $(INCDIR)/partition.h $(INCDIR)/operations.h $(INCDIR)/cluster.h $(INCDIR)/edge.h $(INCDIR)/spaND.h
OBJ  = $(OBJDIR)/tree.o $(OBJDIR)/util.o $(OBJDIR)/is.o $(OBJDIR)/partition.o $(OBJDIR)/operations.o $(OBJDIR)/cluster.o $(OBJDIR)/edge.o

all: icesheet mfem mfem2 mfem5 mfem9 tests spaND

# Source
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS) $(INCLUDE)

# Executables
tests: tests.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -pthread -isystem -I$(GTESTDIR) -L$(GTESTLIB) -lgtest

mfem: mfem.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem2: mfem2.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem5: mfem5.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

mfem9: mfem9.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(MFEMDIR) -L$(MFEMLIB) -lmfem

spaND: spaND.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) 

icesheet: icesheet.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LINK) -I$(UNSUPPORTED_EIGEN_DIR)

.PHONY: clean libspaND

clean:
	rm -f $(OBJDIR)/*.o icesheet spemfem mfem2 mfem9 tests spaND

libspaND:
	cp $(INCDIR)/tree.h $(INCDIR)/util.h $(INCDIR)/is.h $(INCDIR)/partition.h $(INCDIR)/operations.h $(INCDIR)/cluster.h $(INCDIR)/edge.h $(INCDIR)/spaND.h $(BUILDDIR)/include/
	ar -rv $(BUILDDIR)/libspaND.a $(OBJDIR)/tree.o $(OBJDIR)/util.o $(OBJDIR)/is.o $(OBJDIR)/partition.o $(OBJDIR)/operations.o $(OBJDIR)/cluster.o $(OBJDIR)/edge.o

