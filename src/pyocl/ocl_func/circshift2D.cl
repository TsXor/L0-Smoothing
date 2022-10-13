__kernel void circshift2D(
    __global DTYPE * arr_in,
    __global DTYPE * arr_out,
    int shift_x,
    int shift_y
)
{
    int id_x; id_x = get_global_id(1);
    int id_y; id_y = get_global_id(0);
    int w; w = get_global_size(1);
    int h; h = get_global_size(0);
    int nid_x; nid_x = id_x + shift_x;
    int nid_y; nid_y = id_y + shift_y;
    nid_x = nid_x % w; nid_x = (nid_x < 0) ? nid_x + w : nid_x;
    nid_y = nid_y % h; nid_y = (nid_y < 0) ? nid_y + h : nid_y;
    arr_out[nid_y*w+nid_x] = arr_in[id_y*w+id_x];
}