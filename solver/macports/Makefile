
OBJS = test.o SOCP.o QCQP.o QP.o PrimalDualIPM.o
DEPS = $(OBJS:.o=.d)
BIN  = test.exe

CPPFLAGS  = -Wall -I../src -I/opt/local/include/eigen3
CPPFLAGS += -msse2 -O2
#CPPFLAGS += -g -DIPM_ENABLE_ASSERT
#CPPFLAGS += -DIPM_DISABLE_LOG
#CPPFLAGS += -DIPM_USE_FLOAT
#CPPFLAGS += -DIPM_DECOMP_TYPE=ColPivHouseholderQR

VPATH=../src

all: $(BIN)

$(BIN): $(OBJS) Makefile
	g++ $(CPPFLAGS) $(OBJS) -o $@

%.d: %.cpp Makefile
	g++ $(CPPFLAGS) $< -MM -o $@

%.o: %.cpp Makefile
	g++ $(CPPFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS) $(DEPS) $(BIN)

-include $(DEPS)
