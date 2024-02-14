kernel void
add(const global float* augend,
    const global float* addend,
    global float* sum,
    const unsigned int count) {
    int id = get_global_id(0);

    if (id < count) {
        sum[id] = augend[id] + addend[id];
    }
}
