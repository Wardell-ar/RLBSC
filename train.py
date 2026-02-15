import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# --- 修正导入 ---
# 改回 simglucose.envs，这才是支持 patient_name 参数的类
from simglucose.envs import T1DSimEnv 
from custom_env import RLMealWrapper

def make_env():
    """
    创建并封装环境
    """
    # 使用 Gym 包装器版本，它会自动加载 patient_name
    base_env = T1DSimEnv(patient_name='adolescent#001')

    # 应用我们的兼容性 Wrapper
    env = RLMealWrapper(base_env)
    
    # allow_early_resets 很有用，防止环境未重置时出错
    env = Monitor(env, filename="./logs/", allow_early_resets=True)
    return env

def main():
    # --- 1. 初始化环境 ---
    env = make_env()
    print("环境创建成功！(Gym Wrapper Mode)")
    print(f"状态空间: {env.observation_space.shape}")
    print(f"动作空间: {env.action_space}")

    # --- 2. 定义模型 (SAC) ---
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        learning_starts=100,  # 【新增】前1000步纯随机探索，不更新网络
        buffer_size=50_000, 
        batch_size=2048,

        # 【优化 2】调整训练频率 (train_freq) 和 梯度更新步数 (gradient_steps)
        # 原理：不要“走1步、练1次”，这样 CPU 和 GPU 频繁切换很浪费时间。
        # 改为：让 CPU 专心跑 100 步仿真，然后让 GPU 专心突击训练 100 次。
        # 这样能显著减少切换开销，提升 GPU 占用率。
        train_freq=(100, "step"), 
        gradient_steps=100,

        gamma=0.99,
        tau=0.005,
        ent_coef='auto', 
        verbose=1,
        tensorboard_log="./sac_glucose_tensorboard/",
        device='cuda'
    )

    # --- 3. 设置检查点 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./logs/checkpoints/',
        name_prefix='rl_meal_model'
    )

    # --- 4. 开始训练 ---
    print("开始训练...")
    # 第一次跑可以设置小一点步数，确认无报错，例如 2000
    model.learn(total_timesteps=10000, callback=checkpoint_callback)
    print("训练结束！")

    # --- 5. 保存 ---
    save_path = "final_rl_meal_model"
    model.save(save_path)
    print(f"模型已保存至: {save_path}.zip")

if __name__ == "__main__":
    os.makedirs("./logs/checkpoints/", exist_ok=True)
    main()