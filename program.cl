kernel void matrix_multiply(
    const global float* a,
    const global float* b,
    global float* c,
    const uint N,
    const uint M,
    const uint K
) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    float sum = 0;
    for (uint k = 0; k < K; ++k) {
        sum += a[y * K + k] * b[k * N + x];
    }
    c[y * N + x] = sum;
}
