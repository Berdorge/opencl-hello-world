#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

static size_t ceil_division(size_t a, size_t b) {
    if (a % b) {
        return a / b + 1;
    } else {
        return a / b;
    }
}

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static const size_t playground_buffer_size = 8192;

int main(int argc, char* argv[]) {
    cl_uint N;
    cl_uint M;
    cl_uint K;

    scanf("%u%u%u", &N, &K, &M);

    size_t a_size = sizeof(float) * M * K;
    size_t b_size = sizeof(float) * K * N;
    size_t c_size = sizeof(float) * M * N;

    float* host_a = (float*)malloc(a_size);
    float* host_b = (float*)malloc(b_size);
    float* host_c = (float*)malloc(c_size);
    float* expected_c = (float*)malloc(c_size);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            scanf("%f", host_a + i * K + j);
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            scanf("%f", host_b + i * N + j);
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += host_a[i * K + k] * host_b[k * N + j];
            }
            expected_c[i * N + j] = sum;
        }
    }

    cl_int err;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, NULL);

    char* playground_buffer = (char*)malloc(playground_buffer_size);
    clGetPlatformInfo(
        platform_id,
        CL_PLATFORM_NAME,
        playground_buffer_size,
        playground_buffer,
        NULL
    );
    printf("Platform name is %s\n", playground_buffer);

    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);

    clGetDeviceInfo(device_id, CL_DEVICE_NAME, playground_buffer_size, playground_buffer, NULL);
    printf("Device name is %s\n", playground_buffer);

    cl_context context;
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    FILE* program_source = fopen("program.cl", "rb");
    fseek(program_source, 0, SEEK_END);
    size_t file_length = ftell(program_source);
    rewind(program_source);

    char* kernel_source = (char*)malloc(file_length * sizeof(char));
    fread(kernel_source, sizeof(char), file_length, program_source);
    fclose(program_source);

    cl_program program;
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &err);

    char build_options[] = "";
    err = clBuildProgram(program, 0, NULL, build_options, NULL, NULL);

    if (err == 0) {
        puts("Program compiled successfully!");
    } else {
        clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            playground_buffer_size,
            playground_buffer,
            NULL
        );
        puts("Program compilation failed.");
        puts("Build log:");
        puts(playground_buffer);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel;
    kernel = clCreateKernel(program, "matrix_multiply", &err);

    cl_mem device_a;
    cl_mem device_b;
    cl_mem device_c;
    device_a = clCreateBuffer(context, CL_MEM_READ_ONLY, a_size, NULL, NULL);
    device_b = clCreateBuffer(context, CL_MEM_READ_ONLY, b_size, NULL, NULL);
    device_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_size, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_c);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &N);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &M);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &K);

    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

    err = clEnqueueWriteBuffer(queue, device_a, CL_FALSE, 0, a_size, host_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, device_b, CL_FALSE, 0, b_size, host_b, 0, NULL, NULL);

    const size_t global_work_size[] = {N, M};

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &global_work_size, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, c_size, host_c, 0, NULL, NULL);

    printf("\nActual output:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%f ", host_c[i * M + j]);
        }
        printf("\n");
    }

    printf("\nExpected output:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%f ", expected_c[i * M + j]);
        }
        printf("\n");
    }

    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(host_a);
    free(host_b);
    free(host_c);
    free(expected_c);

    return 0;
}
