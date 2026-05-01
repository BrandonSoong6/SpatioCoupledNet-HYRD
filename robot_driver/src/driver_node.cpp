#include <rclcpp/rclcpp.hpp>
#include "robot_interfaces/msg/motor_command.hpp"
#include "robot_interfaces/msg/motor_state.hpp"

// 引入你的底层库
#include "SCServo/SCServo.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>
#include <string>
#include <algorithm>
#include <map>

// ==========================================
// 1. 全局配置与参数
// ==========================================
const int STS_COUNT = 6;  
const int HLS_COUNT = 4;  
const int TOTAL_SERVOS = STS_COUNT + HLS_COUNT;
const int BAUD_RATE = 500000; 
const int ACC_VAL   = 0;

// 物理参数
const float HLS_SPD_UNIT = 0.732f;
const float STS_SPD_UNIT = 0.0146f; 
const float UNIT_TO_MM = 0.012195f; 
const float UNIT_LOAD_PCT = 0.1f;
const int HLS_TORQUE_VAL = 500;

// 限制与控制参数
const float TARGET_RPM_LIMIT = 100.0f; // 安全限速
const float LIMIT_MIN_MM = 10.0f;   
const float LIMIT_MAX_MM = 160.0f;  
const float LIMIT_BUFFER_MM = 10.0f; 

const float MAX_OVERSHOOT_MM = 5.0f; 
const float OVERSHOOT_FADE_END = 20.0f; 

const float CPL_OFFSET = 10.0f;
const float CPL_COEFF  = 0.17f;
const float CPL_EXP    = 1.1f;
const float CPL_INV_EXP = 1.0f / 1.1f;

const int INIT_WAIT_MS = 2000;       
const float RECOVERY_KP = 3.0f;      
const float RECOVERY_MAX_RPM = 20.0f;

const float SNAPSHOT_KP = 1.5f; 
const float SNAPSHOT_MAX_RPM = 50.0f; 
const float STOP_THRESHOLD = 5.0f; 

// 辅助函数
float safe_pow(float base, float exp) { return (base < 0) ? 0.0f : std::pow(base, exp); }

// ==========================================
// 2. 舵机状态结构体
// ==========================================
struct ServoState {
    int id;
    bool is_hls;
    std::atomic<int> raw_pos{0};       
    std::atomic<long long> total_dist{0}; 
    int last_abs_pos{-1};              

    // 传感器反馈 (原子类型，线程安全)
    std::atomic<float> phys_spd_rpm{0.0f}; // 原始物理转速
    std::atomic<float> phys_load_pct{0.0f};
    std::atomic<float> phys_voltage{0.0f};
    
    // 指令速度
    std::atomic<float> cmd_rpm{0.0f};          
    
    // 内部计算状态
    float target_rpm_logic{0.0f}; 
    std::string limit_status{"OK"}; 
    float current_overshoot{0.0f};
    
    // 快照逻辑
    float snapshot_mm{LIMIT_MIN_MM}; 
    bool active_move_flag{false}; 

    ServoState(int _id, bool _hls) : id(_id), is_hls(_hls) {}

    void update_position(int p) {
        if (last_abs_pos == -1) {
            last_abs_pos = p;
            raw_pos.store(p);
            return;
        }
        int diff = p - last_abs_pos;
        if (diff < -2048)      diff += 4096;
        else if (diff > 2048)  diff -= 4096;

        // 偶数ID逻辑反向修正
        int logic_diff = (id % 2 == 0) ? (-diff) : diff;
        total_dist.fetch_add(logic_diff);
        last_abs_pos = p;
        raw_pos.store(p);
    }
    
    float get_mm() const { return total_dist.load() * UNIT_TO_MM; }
    
    // 获取逻辑修正后的物理速度 (RPM)
    float get_logic_speed_rpm() const {
        float raw_spd = phys_spd_rpm.load();
        return (id % 2 == 0) ? (-raw_spd) : raw_spd;
    }
};

// ==========================================
// 3. ROS2 Driver Node
// ==========================================
class RobotDriver : public rclcpp::Node {
public:
    RobotDriver() : Node("driver_node") {
        // --- A. 参数声明 ---
        this->declare_parameter("serial_port", "/dev/sts_servo");
        std::string port_name = this->get_parameter("serial_port").as_string();

        RCLCPP_INFO(this->get_logger(), "Connecting to serial port: %s", port_name.c_str());

        // --- B. 硬件初始化 ---
        if(!sms_sts.begin(BAUD_RATE, port_name.c_str()) || !hlscl.begin(BAUD_RATE, port_name.c_str())) {
            RCLCPP_FATAL(this->get_logger(), "Failed to open serial port!");
            rclcpp::shutdown();
            return;
        }

        // 初始化对象列表
        for(int i = 1; i <= 6; i++) servos_.push_back(new ServoState(i, false));
        for(int i = 7; i <= 10; i++) servos_.push_back(new ServoState(i, true));

        // 设置轮模式
        for(auto s : servos_) {
            std::lock_guard<std::mutex> lock(bus_mtx_);
            if(s->is_hls) hlscl.WheelMode(s->id);
            else sms_sts.WheelMode(s->id);
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }

        // --- C. ROS 通信初始化 ---
        sub_cmd_ = this->create_subscription<robot_interfaces::msg::MotorCommand>(
            "motor_cmd", 10, std::bind(&RobotDriver::command_callback, this, std::placeholders::_1));

        pub_state_ = this->create_publisher<robot_interfaces::msg::MotorState>("motor_state", 10);

        // --- D. 启动线程 ---
        running_ = true;
        start_time_ = std::chrono::steady_clock::now();

        feedback_thread_ = std::thread(&RobotDriver::feedback_loop, this);

        // [修改] 主循环频率提升至 100Hz (10ms)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), std::bind(&RobotDriver::control_loop, this));
            
        RCLCPP_INFO(this->get_logger(), "Driver Ready @ 100Hz. Waiting %d ms...", INIT_WAIT_MS);
    }

    ~RobotDriver() {
        running_ = false;
        if(feedback_thread_.joinable()) feedback_thread_.join();
        std::lock_guard<std::mutex> lock(bus_mtx_);
        for(auto s : servos_) {
            if(s->is_hls) hlscl.WriteSpe(s->id, 0);
            else sms_sts.WriteSpe(s->id, 0);
        }
        for(auto s : servos_) delete s;
    }

private:
    SMS_STS sms_sts;
    HLSCL hlscl;
    std::mutex bus_mtx_;
    std::vector<ServoState*> servos_;
    std::atomic<bool> running_;
    std::chrono::steady_clock::time_point start_time_;
    bool initialized_ = false;

    rclcpp::Subscription<robot_interfaces::msg::MotorCommand>::SharedPtr sub_cmd_;
    rclcpp::Publisher<robot_interfaces::msg::MotorState>::SharedPtr pub_state_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::thread feedback_thread_;

    void command_callback(const robot_interfaces::msg::MotorCommand::SharedPtr msg) {
        for (size_t i = 0; i < msg->ids.size(); ++i) {
            int target_id = msg->ids[i];
            float target_rpm = msg->target_rpms[i];

            for (auto s : servos_) {
                if (s->id == target_id) {
                    s->cmd_rpm.store(target_rpm); 
                    break;
                }
            }
        }
    }

    // 线程：硬件反馈 (极速读取)
    void feedback_loop() {
        uint8_t ids[TOTAL_SERVOS];
        for(int i=0; i<TOTAL_SERVOS; i++) ids[i] = servos_[i]->id;
        const int RX_LEN = 7;
        uint8_t rxPacket[RX_LEN];
        
        sms_sts.syncReadBegin(TOTAL_SERVOS, RX_LEN, 15);

        while(running_) {
            {
                std::lock_guard<std::mutex> lock(bus_mtx_);
                sms_sts.syncReadPacketTx(ids, TOTAL_SERVOS, 56, RX_LEN);
            }

            for(int i=0; i<TOTAL_SERVOS; i++) {
                ServoState* s = servos_[i];
                bool success = false;
                {
                    std::lock_guard<std::mutex> lock(bus_mtx_);
                    success = sms_sts.syncReadPacketRx(s->id, rxPacket);
                }
                if(success) {
                    // 1. 位置更新
                    int pos = sms_sts.syncReadRxPacketToWrod(15);
                    s->update_position(pos);
                    
                    // 2. 速度更新
                    int spd = sms_sts.syncReadRxPacketToWrod(15);
                    float unit = s->is_hls ? HLS_SPD_UNIT : STS_SPD_UNIT;
                    // 处理 SCServo 速度符号位 (如果是补码通常 int16 直接强转即可，这里假设库行为正常)
                    if(spd > 32767) spd -= 65536; 
                    s->phys_spd_rpm.store(spd * unit);
                    
                    // 3. 负载更新
                    int load = sms_sts.syncReadRxPacketToWrod(10);
                    if (load > 1000) load -= 1024;
                    s->phys_load_pct.store(load * UNIT_LOAD_PCT);
                    
                    // 4. 电压
                    int volt = rxPacket[6];
                    s->phys_voltage.store(volt * 0.1f);
                }
            }
            // [修改] 缩短休眠时间至 4ms，争取给总线更高吞吐量以匹配 100Hz
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
        sms_sts.syncReadEnd();
    }

    // 主控制循环
    void control_loop() {
        // [调试] 可开启耗时监控
        // auto start = std::chrono::steady_clock::now();
        
        auto now = std::chrono::steady_clock::now();
        
        if (!initialized_) {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count() > INIT_WAIT_MS) {
                initialized_ = true;
            } else {
                std::printf("\033[H\033[2J");
                std::printf(">>> SYSTEM INITIALIZING... <<<\n");
                return;
            }
        }

        // --- 核心算法 ---
        for (size_t i = 0; i < servos_.size(); ++i) {
            ServoState* s = servos_[i];
            s->limit_status = "OK";

            // [A] 读取指令
            float manual_rpm = s->cmd_rpm.load(); 

            // [B] 物理状态
            float self_mm = s->get_mm();
            float current_logic_rpm = s->get_logic_speed_rpm(); 

            // [C] 耦合限位
            int partner_idx = (s->id % 2 != 0) ? (i + 1) : (i - 1);
            if (partner_idx < 0 || partner_idx >= (int)servos_.size()) partner_idx = i;
            float partner_mm = servos_[partner_idx]->get_mm();
            
            float dyn_min = CPL_OFFSET + CPL_COEFF * safe_pow(partner_mm - CPL_OFFSET, CPL_EXP);
            float eff_min = std::max(LIMIT_MIN_MM, dyn_min);
            float dyn_max = CPL_OFFSET + safe_pow((partner_mm - CPL_OFFSET) / CPL_COEFF, CPL_INV_EXP);
            float eff_max = std::min(LIMIT_MAX_MM, dyn_max);

            // [D] 动态过冲
            float min_pos = std::min(self_mm, partner_mm);
            float overshoot = 0.0f;
            if (min_pos < OVERSHOOT_FADE_END) {
                float r = (OVERSHOOT_FADE_END - min_pos) / (OVERSHOOT_FADE_END - LIMIT_MIN_MM);
                overshoot = MAX_OVERSHOOT_MM * std::clamp(r, 0.0f, 1.0f);
            }
            s->current_overshoot = overshoot;

            // [E] Smart Snapshot
            bool is_active = (std::abs(manual_rpm) > 0.1f);
            bool is_fast = (std::abs(current_logic_rpm) > STOP_THRESHOLD);

            if (is_active) {
                s->active_move_flag = true;
            } else {
                if (s->active_move_flag && !is_fast) {
                    s->active_move_flag = false;
                }
            }

            if (s->active_move_flag) {
                float snap = std::clamp(self_mm, eff_min, eff_max);
                s->snapshot_mm = snap;
            }

            // [F] 力叠加
            float boundary_rpm = 0.0f;
            if (self_mm < eff_min) boundary_rpm = (eff_min - self_mm) * RECOVERY_KP;
            else if (self_mm > eff_max) boundary_rpm = (eff_max - self_mm) * RECOVERY_KP;
            boundary_rpm = std::clamp(boundary_rpm, -RECOVERY_MAX_RPM, RECOVERY_MAX_RPM);

            float snap_rpm = (s->snapshot_mm - self_mm) * SNAPSHOT_KP;
            snap_rpm = std::clamp(snap_rpm, -SNAPSHOT_MAX_RPM, SNAPSHOT_MAX_RPM);

            // 用户驱动力 (带阻尼)
            float user_factor = 1.0f;
            float full_range = LIMIT_BUFFER_MM + overshoot;
            
            if (manual_rpm < -0.1f) {
                float dist = (self_mm - eff_min) + overshoot;
                if (dist < 0.0f) { user_factor = 0.0f; s->limit_status = "MIN_CUT"; }
                else if (dist < full_range) { user_factor = dist / full_range; s->limit_status = "MIN_DAMP"; }
            } else if (manual_rpm > 0.1f) {
                float dist = (eff_max - self_mm) + overshoot;
                if (dist < 0.0f) { user_factor = 0.0f; s->limit_status = "MAX_CUT"; }
                else if (dist < full_range) { user_factor = dist / full_range; s->limit_status = "MAX_DAMP"; }
            }
            float user_rpm = manual_rpm * user_factor;

            // [G] 最终输出
            float total = user_rpm + boundary_rpm + snap_rpm;
            total = std::clamp(total, -TARGET_RPM_LIMIT, TARGET_RPM_LIMIT);

            if (!is_active) {
                if (std::abs(boundary_rpm) > 0.1f) s->limit_status = "RECOVER";
                else if (s->active_move_flag) s->limit_status = "COASTING";
                else if (std::abs(snap_rpm) > 0.1f) s->limit_status = "HOLDING";
            }
            
            s->target_rpm_logic = total;
        }

        // --- 执行 SyncWrite ---
        u8 s_ids[STS_COUNT]; u8 s_accs[STS_COUNT]; s16 s_speeds[STS_COUNT];
        u8 h_ids[HLS_COUNT]; u8 h_accs[HLS_COUNT]; s16 h_speeds[HLS_COUNT]; u16 h_torqs[HLS_COUNT];
        int s_idx = 0, h_idx = 0;

        for(auto s : servos_) {
            float final_rpm = (s->id % 2 == 0) ? (-s->target_rpm_logic) : (s->target_rpm_logic);
            s16 raw = s->is_hls ? (s16)(final_rpm / HLS_SPD_UNIT) : (s16)(final_rpm / STS_SPD_UNIT);

            if(!s->is_hls) {
                s_ids[s_idx] = s->id; s_accs[s_idx] = ACC_VAL; s_speeds[s_idx] = raw; s_idx++;
            } else {
                h_ids[h_idx] = s->id; h_accs[h_idx] = ACC_VAL; h_torqs[h_idx] = HLS_TORQUE_VAL;
                h_speeds[h_idx] = raw; h_idx++;
            }
        }

        {
            std::lock_guard<std::mutex> lock(bus_mtx_);
            sms_sts.SyncWriteSpe(s_ids, STS_COUNT, s_speeds, s_accs);
            hlscl.SyncWriteSpe(h_ids, HLS_COUNT, h_speeds, h_accs, h_torqs);
        }

        publish_and_print_state();
    }

    void publish_and_print_state() {
        auto msg = robot_interfaces::msg::MotorState();
        msg.header.stamp = this->get_clock()->now();

        // 终端清屏与表头
        std::printf("\033[H\033[2J");
        std::printf("======================================================================================================================================\n");
        std::printf(" ROS2 DRIVER (LATCHED) | 100Hz MODE | %d Servos Active\n", TOTAL_SERVOS);
        std::printf("======================================================================================================================================\n");
        std::printf(" ID | RawCmd | Tgt(RPM) | ActSpd(RPM) | Dist(mm)  | Volt(V) | Load(%%) |  Eff Min  |  Eff Max  | Over(mm) | Active | State    |\n");
        std::printf("----|--------|----------|-------------|-----------|---------|---------|-----------|-----------|----------|--------|----------|\n");

        for (size_t i = 0; i < servos_.size(); ++i) {
            auto s = servos_[i];
            
            // 辅助计算
            int partner_idx = (s->id % 2 != 0) ? (i + 1) : (i - 1);
            if (partner_idx < 0 || partner_idx >= (int)servos_.size()) partner_idx = i;
            float p_mm = servos_[partner_idx]->get_mm();
            float d_min = std::max(LIMIT_MIN_MM, CPL_OFFSET + CPL_COEFF * safe_pow(p_mm - CPL_OFFSET, CPL_EXP));
            float d_max = std::min(LIMIT_MAX_MM, CPL_OFFSET + safe_pow((p_mm - CPL_OFFSET) / CPL_COEFF, CPL_INV_EXP));

            float mm = s->get_mm();
            float raw_cmd_val = s->cmd_rpm.load();
            float logic_spd = s->get_logic_speed_rpm();

            // 打印
            std::printf(" %2d | %6.1f | %8.1f | %11.1f | %9.2f | %7.1f | %7.1f | %9.2f | %9.2f | %8.2f | %6s | %-8s |\n", 
                s->id, 
                raw_cmd_val,          
                s->target_rpm_logic,  
                logic_spd,            
                mm, 
                s->phys_voltage.load(), s->phys_load_pct.load(),
                d_min, d_max, s->current_overshoot,
                s->active_move_flag ? "YES" : "NO", s->limit_status.c_str());

            // 填充 ROS 消息 (确保 .msg 文件已包含 speeds)
            msg.ids.push_back(s->id);
            msg.positions.push_back(mm);     
            msg.speeds.push_back(logic_spd); // [新增] Speed
            msg.loads.push_back(s->phys_load_pct.load());
            msg.statuses.push_back(s->limit_status);
        }
        std::fflush(stdout);
        
        pub_state_->publish(msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotDriver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}