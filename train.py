import warnings
# 放在最前面，屏蔽烦人的警告
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import random
import gymnasium as gym
import numpy as np
import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# 确保安装了 simglucose: pip install simglucose
from simglucose.envs import T1DSimEnv 
from custom_env import RLMealWrapper
from gymnasium.wrappers import TimeLimit

# 1. 定义包含所有 30 位患者的列表
PATIENT_LIST = [f'adolescent#{str(i).zfill(3)}' for i in range(1, 6)] + \
               [f'adult#{str(i).zfill(3)}' for i in range(1, 6)] + \
               [f'child#{str(i).zfill(3)}' for i in range(1, 6)]

# 2. 定义一个支持动态切换病人的“通用环境”
class UniversalT1DSimEnv(gym.Env):
    def __init__(self, patient_list):
        self.patient_list = patient_list
        self.env = None
        # 初始化时随机选一个，以便建立 action_space 等
        self._switch_patient()
        
    def _switch_patient(self):
        # 随机选择一个病人
        patient_name = random.choice(self.patient_list)
        # 关闭旧环境（如果有）
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
        
        # 创建新环境
        # print(f"Switching to patient: {patient_name}") # 调试用
        self.env = T1DSimEnv(patient_name=patient_name)
        
        # ⚠️ 必须强制重设 sample_time，否则新环境会变回默认值
        self.env.sample_time = 5
        if hasattr(self.env, 'sensor'):
            self.env.sensor.sample_time = 5

    def reset(self, seed=None, options=None):
        # 1. 切换病人
        self._switch_patient()
        
        # 2. 安全地调用底层环境的 reset
        try:
            # 尝试传入 seed (针对支持新版 API 的环境)
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            # 如果报错 "unexpected keyword argument 'seed'"
            # 说明底层是旧版 gym，只能不带参数调用
            return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
        
    def close(self):
        if self.env:
            self.env.close()

    # --- 关键魔法方法 ---
    # 这一步至关重要！它让 RLMealWrapper 能直接访问 self.env.CHO_hist 等属性
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

# 3. 修改工厂函数使用 UniversalEnv
def make_env_factory(universal=True):
    def _init():
        if universal:
            # 使用通用环境
            base_env = UniversalT1DSimEnv(PATIENT_LIST)
        else:
            # 使用固定单一环境
            base_env = T1DSimEnv(patient_name='adolescent#006')
            base_env.sample_time = 5
            
        # 套上你的特征提取包装器
        env = RLMealWrapper(base_env)
        
        # ⚠️ 强烈建议加上 TimeLimit，确保每 21 天(6048步)强制重置一次
        # 这样才能触发换人，也能让 log 正常输出
        env = TimeLimit(env, max_episode_steps=6048)
        
        return env
    return _init

def main():
    # --- 2. 配置并行数量 ---
    # 建议设置为你的 CPU 核心数 (例如 4, 8, 12)
    n_envs = 16  
    print(f"正在启动 {n_envs} 个并行环境进程...")

    # --- 3. 创建向量化环境 ---
    # 使用列表推导式创建多个工厂函数实例
    env_fns = [make_env_factory(universal=True) for _ in range(n_envs)]
    
    # 启动多进程环境
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env, filename="./logs/")

    print("环境创建成功！")
    
    # --- 4. 定义网络结构 ---
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )

    # --- 5. 定义模型 ---
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,     # 论文参数
        buffer_size=int(1e6),   # 论文参数
        batch_size=256,         # 论文参数
        ent_coef='auto',        
        gamma=0.99,             # 论文参数
        tau=0.005,
        
        # 这里的逻辑是：每收集 8000 步数据（所有进程总和），就进行 8000 次梯度更新
        # 这既保证了训练频率，又避免了多进程回合对齐的问题
        train_freq=(500, "step"), 
        gradient_steps=500*n_envs,         
        
        learning_starts=200000, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./sac_rl_meal_tensorboard/",
        device='cuda' 
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=200000 // n_envs, 
        save_path='./logs/checkpoints/',
        name_prefix='rl_meal_sac'
    )

    print("开始多进程并行训练...")
    # 论文进行了 300 epochs，总计约 1.8M 步。
    # 这里先跑 200,000 步测试效果
    model.learn(
        total_timesteps=2000000, 
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=1
    )
    
    model.save("final_rl_meal_model")
    print("训练完成。")
    env.close()

if __name__ == "__main__":
    # 这一步是为了避免某些环境下路径不存在报错
    os.makedirs("./logs/checkpoints/", exist_ok=True)
    main()