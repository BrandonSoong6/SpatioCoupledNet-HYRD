import csv
import re
import os

def parse_log_to_csv(input_path):
    # 自动在同级目录下生成同名的 csv 文件
    # 例如: raw_log.txt -> raw_log.csv
    base_name = os.path.splitext(input_path)[0]
    output_path = base_name + ".csv"
    
    # 定义CSV的表头
    headers = ['Epoch', 'Train_Loss', 'Val_Loss', 'Tip_X_mm', 'Tip_Y_mm', 'Alpha', 'LR', 'Status']
    
    data_rows = []
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到文件: {input_path}")
        return

    print(f"📂 正在读取: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 过滤规则：必须以数字开头，且包含 '|'
            if not re.match(r'^\d+\s+\|', line):
                continue
            
            parts = line.split('|')
            
            if len(parts) < 8:
                continue
                
            try:
                # 解析各列数据
                epoch = int(parts[0].strip())
                train_loss = float(parts[1].strip())
                val_loss = float(parts[2].strip())
                
                # 去除单位 'mm'
                tip_x = float(parts[3].strip().replace('mm', ''))
                tip_y = float(parts[4].strip().replace('mm', ''))
                
                alpha = float(parts[5].strip())
                lr = float(parts[6].strip())
                status = parts[7].strip()
                
                data_rows.append([epoch, train_loss, val_loss, tip_x, tip_y, alpha, lr, status])
            except ValueError:
                continue

    # 写入 CSV
    if data_rows:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data_rows)
        print(f"✅ 处理完成! 共提取 {len(data_rows)} 行数据。")
        print(f"💾 CSV已保存至: {output_path}")
    else:
        print("⚠️ 未提取到有效数据，请检查日志内容格式。")

if __name__ == "__main__":
    # 指定你的绝对路径
    target_file = '/home/brandon/brandon/hyrd_robot/src/robot_brain/scripts/raw_log.txt'
    
    parse_log_to_csv(target_file)