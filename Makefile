all: main.c
	clang -framework OpenCL -o output main.c

run: all
	./output

clean:
	rm output
