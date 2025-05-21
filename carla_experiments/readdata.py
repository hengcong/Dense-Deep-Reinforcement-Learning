import numpy as np

# 设置你的文件路径
file_path = "/home/hguo/Dense-Deep-Reinforcement-Learning/carla_experiments/weight0.npy"

# 加载数据
try:
    data = np.load(file_path, allow_pickle=True)
    print("📦 成功读取数据:")
    print(data)

    # 如果是长度为3的统计信息
    if isinstance(data, (list, np.ndarray)) and len(data) == 3:
        crash, success, total = data
        print(f"\n💥 Crash 数量: {crash}\n✅ 成功数: {success}\n🎯 总实验数: {total}")

    # 如果是每个 episode 的权重
    elif isinstance(data, (list, np.ndarray)):
        print(f"\n📊 共 {len(data)} 个 episode 结果:")
        for i, w in enumerate(data):
            print(f"Episode {i}: weight = {w}")
except Exception as e:
    print(f"❌ 读取文件出错: {e}")
