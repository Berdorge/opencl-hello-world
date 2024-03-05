#include <math.h>
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
static const size_t elements = 100000;

int main(int argc, char* argv[]) {
    float* host_augend;
    float* host_addend;
    float* host_sum;

    size_t bytes = elements * sizeof(float);

    host_augend = (float*)malloc(bytes);
    host_addend = (float*)malloc(bytes);
    host_sum = (float*)malloc(bytes);

    for (size_t i = 0; i < elements; i++) {
        host_augend[i] = sinf(i) * sinf(i);
        host_addend[i] = cosf(i) * cosf(i);
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
    kernel = clCreateKernel(program, "add", &err);

    cl_mem device_augend;
    cl_mem device_addend;
    cl_mem device_sum;
    device_augend = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    device_addend = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    device_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_augend);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_addend);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_sum);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &elements);

    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

    err =
        clEnqueueWriteBuffer(queue, device_augend, CL_FALSE, 0, bytes, host_augend, 0, NULL, NULL);
    err |=
        clEnqueueWriteBuffer(queue, device_addend, CL_FALSE, 0, bytes, host_addend, 0, NULL, NULL);

    size_t local_work_size;
    clGetKernelWorkGroupInfo(
        kernel,
        device_id,
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(local_work_size),
        &local_work_size,
        NULL
    );
    printf("Kernel work group size is %zu\n", local_work_size);

    size_t work_groups = ceil_division(elements, local_work_size);
    printf("There are %zu work groups\n", work_groups);

    size_t global_work_size = work_groups * local_work_size;
    printf("Global work size is %zu\n", global_work_size);

    err = clEnqueueNDRangeKernel(
        queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );

    clEnqueueReadBuffer(queue, device_sum, CL_TRUE, 0, bytes, host_sum, 0, NULL, NULL);

    float sum = 0;
    for (size_t i = 0; i < elements; i++) {
        sum += host_sum[i];
    }

    printf("sum: %f\n", sum);

    clReleaseMemObject(device_augend);
    clReleaseMemObject(device_addend);
    clReleaseMemObject(device_sum);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(host_augend);
    free(host_addend);
    free(host_sum);

    return 0;
}
