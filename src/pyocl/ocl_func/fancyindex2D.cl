__kernel void fancyindex2D(
    __global DTYPE * arr_in,
    __global int * arr_mask,
    __global DTYPE * arr_out,
    DTYPE value
)
{
    int id_x; id_x = get_global_id(1);
    int id_y; id_y = get_global_id(0);
    int w; w = get_global_size(1);
    if (arr_mask[id_y*w+id_x])
    {
        arr_out[id_y*w+id_x] = value;
        return;
    }
    arr_out[id_y*w+id_x] = arr_in[id_y*w+id_x];
}