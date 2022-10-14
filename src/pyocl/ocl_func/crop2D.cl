__kernel void crop2D(
	__global DTYPE * arr_in,
	__global DTYPE * arr_out,
	int xl, int xr, int yt, int yb
)
{
	int id_x; id_x = get_global_id(1);
	int id_y; id_y = get_global_id(0);
	int map_x; map_x = id_x - xl;
	int map_y; map_y = id_y - yt;
	int w; w = get_global_size(1);
	int h; h = get_global_size(0);
	int new_w; new_w = xr - xl;
	int new_h; new_h = yb - yt;
	if (map_x < 0 | map_x >= new_w | map_y < 0 | map_y >= new_h)
	{
		return;
	}
	arr_out[map_y*new_w+map_x] = arr_in[id_y*w+id_x];
}