std_msgs/Header header

# 识别到的物体列表
int32[] ids

# 局部坐标 (相对于你定义的 Origin)
float32[] x_local
float32[] y_local

# 角度
float32[] theta

# 系统状态
bool is_plane_locked