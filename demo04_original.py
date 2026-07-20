import time
import threading
import socket
import struct
import numpy as np
import mne


class Ringbuffer:
    """先进先出环形缓冲区，利用 Numpy 切片实现高速滚动"""

    def __init__(self, n_chan, n_points):
        self.buffer = np.zeros((n_chan, n_points))
        self.n_chan = n_chan
        self.n_points = n_points
        self.currentPtr = 0
        self.nUpdate = 0

    def appendBuffer(self, data):
        pts = data.shape[1]
        if pts >= self.n_points:
            self.buffer = data[:, -self.n_points:]
            self.currentPtr = 0
        else:
            end_ptr = (self.currentPtr + pts) % self.n_points
            if end_ptr > self.currentPtr:
                self.buffer[:, self.currentPtr:end_ptr] = data
            else:
                rem = self.n_points - self.currentPtr
                self.buffer[:, self.currentPtr:] = data[:, :rem]
                self.buffer[:, :end_ptr] = data[:, rem:]
            self.currentPtr = end_ptr
        self.nUpdate += pts

    def getData(self):
        return np.roll(self.buffer, -self.currentPtr, axis=1)


class DataServerThread(threading.Thread):
    def __init__(self, device_name, n_chan, srate, t_buffer):
        super().__init__()
        self.device_name = device_name
        self.n_chan = n_chan
        self.srate = srate
        self.buffer_size = int(t_buffer * srate)
        self.ring_buf = Ringbuffer(self.n_chan, self.buffer_size)

        self.notconnect = True
        self._is_running = False
        self.sock = None

        # 二进制解析规则：(通道数-1) 个 float + 1 个 int (Trigger)
        self.bytes_per_pkg = self.n_chan * 4
        self.struct_format = '<' + 'f' * (self.n_chan - 1) + 'i'

    def Connect(self, hostname='127.0.0.1', port=8712):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((hostname, port))
            self.notconnect = False
        except Exception as e:
            print(f"连接失败: {e}")
            self.notconnect = True
        return self.notconnect

    def GetDataLenCounter(self):
        return self.ring_buf.nUpdate

    def ResetDataLenCount(self, count=0):
        self.ring_buf.nUpdate = count

    def GetBufferData(self):
        return self.ring_buf.getData()

    def run(self):
        if self.notconnect or self.sock is None:
            return

        self._is_running = True
        residual_bytes = b''

        while self._is_running:
            try:
                recv_data = self.sock.recv(4096)
                if not recv_data:
                    break

                residual_bytes += recv_data
                num_packages = len(residual_bytes) // self.bytes_per_pkg

                if num_packages > 0:
                    valid_bytes_length = num_packages * self.bytes_per_pkg
                    valid_bytes = residual_bytes[:valid_bytes_length]
                    residual_bytes = residual_bytes[valid_bytes_length:]

                    parsed_data = np.array(list(struct.iter_unpack(self.struct_format, valid_bytes)))
                    self.ring_buf.appendBuffer(parsed_data.T)

            except socket.timeout:
                continue
            except Exception as e:
                print(f"数据接收异常: {e}")
                break

    def stop(self):
        self._is_running = False
        if self.sock:
            self.sock.close()


def main():
    # 1. 设备基础配置
    neuracle = dict(device_name='Neuracle', hostname='127.0.0.1', port=8712,
                    srate=1000, chanlocs=['Pz', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'Oz', 'O1', 'O2', 'TRG'], n_chan=10)
    target_device = neuracle

    # 2. 启动后台数据接收线程 (设定 3 秒滑动窗 = 3000个点)
    time_buffer = 3
    thread_data_server = DataServerThread(device_name=target_device['device_name'], n_chan=target_device['n_chan'],
                                          srate=target_device['srate'], t_buffer=time_buffer)

    print(f"正在尝试连接端口 {target_device['port']}...")
    notconnect = thread_data_server.Connect(hostname=target_device['hostname'], port=target_device['port'])

    if notconnect:
        raise ConnectionError("连接失败。请确保官方软件已开启 Data Sending。")

    thread_data_server.daemon = True
    thread_data_server.start()
    print('TCP 连接建立成功，正在接收原生数据...\n')

    # ---------------------------------------------------------
    # 3. Electrode selection for the realtime extraction example.
    # ---------------------------------------------------------
    # Electrode names must exist in target_device['chanlocs'].
    target_electrodes = ['POz', 'Oz', 'O1', 'O2']

    channel_list = target_device['chanlocs']
    target_indices = [channel_list.index(elec) for elec in target_electrodes if elec in channel_list]
    valid_electrodes = [elec for elec in target_electrodes if elec in channel_list]

    # Prebuild MNE metadata for converting each window into a RawArray.
    mne_info = mne.create_info(ch_names=valid_electrodes, sfreq=target_device['srate'],
                               ch_types=['eeg'] * len(valid_electrodes))

    N, flagstop = 0, False
    try:
        while not flagstop:
            nUpdate = thread_data_server.GetDataLenCounter()

            # 每累积 1 秒的新数据，就提取一次矩阵
            if nUpdate >= target_device['srate']:
                N += 1

                # full_data 是包含所有通道、长达 3 秒的完整矩阵 (10 x 3000)
                full_data = thread_data_server.GetBufferData()
                thread_data_server.ResetDataLenCount()

                # ---------------------------------------------------------
                # 4. Select the target channel matrix from the full ring buffer.
                # ---------------------------------------------------------
                # Shape: (selected_channels, buffered_samples).
                selected_raw_data = full_data[target_indices, :]

                print(f"--- 第 {N} 次实时抓取 ---")
                print(f"目标通道: {valid_electrodes} | 矩阵 selected_raw_data 维度: {selected_raw_data.shape}")

                # Convert the selected window to an MNE RawArray for downstream filtering.
                raw_chunk = mne.io.RawArray(selected_raw_data, mne_info, verbose=False)

            # Brief sleep to reduce CPU usage.
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n手动中止接收。")
    finally:
        # 安全断开连接，释放端口
        thread_data_server.stop()
        thread_data_server.join()
        print("数据接收端口已安全关闭。")


if __name__ == '__main__':
    main()
