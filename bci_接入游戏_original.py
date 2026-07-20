# -*- coding: utf-8 -*-
"""
虚拟任务竞速赛 - LSL控制测试Demo
功能：创建LSL数据流，定时发送0/1/2/3控制赛车
对应游戏配置文件：虚拟任务竞速赛_Data/StreamingAssets/LSLInletConfig.txt
默认LSL名称: EEGback, 类型: EEG
"""

import time
import sys

# ============== 配置区域 ==============
# LSL流名称（必须和LSLInletConfig.txt中|左边一致）
STREAM_NAME = 'EEGback'
# LSL流类型（必须和LSLInletConfig.txt中|右边一致）
STREAM_TYPE = 'EEG'
# 采样率（Hz），控制发送频率
SAMPLE_RATE = 10  # 每秒10个样本
# 通道数（发送几个数值），这里只需要1个通道传分类结果
N_CHANNELS = 1


# ======================================


def test_lsl_installed():
    """检查pylsl是否安装"""
    try:
        from pylsl import StreamInfo, StreamOutlet
        return True
    except ImportError:
        print("❌ 未安装 pylsl 库")
        print("请先运行: pip install pylsl")
        return False


def create_lsl_outlet():
    """创建LSL输出流"""
    from pylsl import StreamInfo, StreamOutlet

    # 创建流信息：名称、类型、通道数、采样率、数据类型
    info = StreamInfo(
        name=STREAM_NAME,
        type=STREAM_TYPE,
        channel_count=N_CHANNELS,
        nominal_srate=SAMPLE_RATE,
        channel_format='float32',
        source_id='bci_test_demo'
    )

    # 创建输出端
    outlet = StreamOutlet(info)
    print(f"✅ LSL流已创建: 名称='{STREAM_NAME}', 类型='{STREAM_TYPE}'")
    print(f"   通道数: {N_CHANNELS}, 采样率: {SAMPLE_RATE}Hz")
    return outlet


def send_sample(outlet, value):
    """发送一个样本值"""
    # LSL要求传入列表，即使只有1个通道
    outlet.push_sample([float(value)])


def timed_sequence_test():
    """定时序列测试：自动循环发送0/1/2/3"""
    from pylsl import local_clock

    print("=" * 60)
    print("🎮 虚拟任务竞速赛 - LSL控制测试Demo")
    print("=" * 60)
    print()
    print("📋 控制映射:")
    print("   0 → 左移（左转区域加速）")
    print("   1 → 右移（右转区域加速）")
    print("   2 → 前进（直行区域加速）")
    print("   3 → 停止（加速区域触发buff）")
    print()
    print("⏱️ 时序安排（每2秒切换，循环往复）:")
    print("   0-2s → 2 (前进)")
    print("   2-4s → 0 (左移)")
    print("   4-6s → 2 (前进)")
    print("   6-8s → 1 (右移)")
    print("   8-10s → 3 (停止)")
    print("=" * 60)
    print()

    # 创建LSL流
    outlet = create_lsl_outlet()

    print("\n🚀 开始发送LSL数据（按 Ctrl+C 停止）...")
    print("💡 提示：现在启动游戏，进入比赛后应该会自动连接这个LSL流\n")

    try:
        start_time = time.time()
        interval = 1.0 / SAMPLE_RATE
        last_value = -1

        while True:
            elapsed = time.time() - start_time
            cycle_time = elapsed % 10  # 每10秒一个循环

            # 根据时间决定当前分类值
            if cycle_time < 2:
                current_value = 2  # 前进
            elif cycle_time < 4:
                current_value = 0  # 左移
            elif cycle_time < 6:
                current_value = 2  # 前进
            elif cycle_time < 8:
                current_value = 1  # 右移
            else:
                current_value = 3  # 停止

            # 发送LSL样本
            send_sample(outlet, current_value)

            # 只在变化时打印，避免刷屏
            if current_value != last_value:
                action_map = {0: '左移', 1: '右移', 2: '前进', 3: '停止'}
                print(f"📤 发送值: {current_value} → {action_map[current_value]}")
                last_value = current_value

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户停止测试")
    except Exception as e:
        print(f"\n❌ 出错: {e}")


def manual_test():
    """手动控制模式：按键盘0/1/2/3发送"""
    import msvcrt  # Windows专用

    outlet = create_lsl_outlet()

    print("\n🎮 手动控制模式")
    print("按 0=左移  1=右移  2=前进  3=停止  q=退出\n")

    try:
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore')
                if key.lower() == 'q':
                    break
                if key in ['0', '1', '2', '3']:
                    value = int(key)
                    send_sample(outlet, value)
                    action_map = {0: '左移', 1: '右移', 2: '前进', 3: '停止'}
                    print(f"按键 {key} → 发送值 {value} ({action_map[value]})")
            time.sleep(0.01)
    finally:
        print("\n测试结束")


if __name__ == '__main__':
    if not test_lsl_installed():
        sys.exit(1)

    # 命令行参数：python lsl_test_demo.py manual 进入手动模式
    if len(sys.argv) > 1 and sys.argv[1] == 'manual':
        manual_test()
    else:
        timed_sequence_test()