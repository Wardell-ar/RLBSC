import gymnasium as gym
import numpy as np
import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from simglucose.envs import T1DSimEnv 
from custom_env import RLMealWrapper

# --- 1. 定义环境工厂函数 ---
# 多进程模式下，每个进程都需要独立调用这个函数来创建一个新环境
def make_env_factory():
    # 这里可以指定病人，建议固定一个或者随机
    patient_name = 'adolescent#006' 
    base_env = T1DSimEnv(patient_name=patient_name)
    base_env.sample_time = 5
    env = RLMealWrapper(base_env,patient_name=patient_name)
    return env

def main():
    # --- 2. 配置并行数量 ---
    # n_envs: 并行环境数量。
    # 如果你的电脑是 8核，建议设为 8 或 12；如果是 4核，设为 4。
    # 越高，收集数据越快，但内存消耗越大。
    n_envs = 8  
    
    print(f"正在启动 {n_envs} 个并行环境进程...")

    # --- 3. 创建向量化环境 (Vectorized Env) ---
    # vec_env_cls=SubprocVecEnv 是关键，它让每个环境在独立的进程中运行 (True Multiprocessing)
    env = make_vec_env(
        make_env_factory, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv
    )
    
    # VecMonitor 用于记录多进程环境的奖励曲线，替代单环境的 Monitor
    env = VecMonitor(env, filename="./logs/")

    print("环境创建成功！")
    
    # --- 4. 定义网络结构 ---
    # 使用更大的网络来逼近论文中 GRU 的效果
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )

    # --- 5. 定义模型 ---
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=int(1e6),   
        batch_size=256,         
        ent_coef='auto',
        gamma=0.99,            
        tau=0.005,

        train_freq=(100, "step"), 
        # 这样：每收集 800 (100*8) 个数据，就训练 800 次
        gradient_steps=100 * n_envs,       
        
        learning_starts=5000, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./sac_rl_meal_tensorboard/",
        device='cuda' 
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // n_envs, # 注意：save_freq 是按主进程步数算的，所以要除以并行数
        save_path='./logs/checkpoints/',
        name_prefix='rl_meal_sac'
    )

    print("开始多进程并行训练...")
    # 20万步 / 8进程 = 主循环只需跑 2.5万次，速度会快很多
    model.learn(
        total_timesteps=100000, 
        callback=checkpoint_callback,
        progress_bar=True,  # <--- 加上这一行
        log_interval=1      # <--- 配合这个，强制每次更新都打印日志
    )
    
    model.save("final_rl_meal_model")
    print("训练完成。")
    
    # 关闭进程池
    env.close()

if __name__ == "__main__":
    # Windows 下使用多进程必须放在这里面，否则会无限递归报错
    os.makedirs("./logs/checkpoints/", exist_ok=True)
    main()