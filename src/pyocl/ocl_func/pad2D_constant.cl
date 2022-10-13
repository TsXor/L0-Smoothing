__kernel void pad2D_constant(
	__global DTYPE * arr_in,
	__global DTYPE * arr_out,
	int l, int r, int t, int b,
	DTYPE constant_v
)
{
	int id_x; id_x = get_global_id(1);
	int id_y; id_y = get_global_id(0);
	int map_x; map_x = id_x - l;
	int map_y; map_y = id_y - t;
	int w; w = get_global_size(1);
	int h; h = get_global_size(0);
	int orig_w; orig_w = w - l - r;
	int orig_h; orig_h = h - t - b;
	if (map_x < 0 | map_x >= orig_w | map_y < 0 | map_y >= orig_h)
	{
		arr_out[id_y*w+id_x] = constant_v;
		return;
	}
	arr_out[id_y*w+id_x] = arr_in[map_y*orig_w+map_x];
}