kernel void
add(const global float* augend, const global float* addend, global float* sum, const uint count) {
    size_t id = get_global_id(0);

    if (id < count) {
        sum[id] = augend[id] + addend[id];
    }
}
