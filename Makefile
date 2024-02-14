CCFLAGS :=
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CCFLAGS += -lm -lOpenCL
endif
ifeq ($(UNAME_S),Darwin)
	CCFLAGS += -framework OpenCL 
endif

all: main.c
	cc -o output main.c $(CCFLAGS) 

run: all
	./output

clean:
	rm output
