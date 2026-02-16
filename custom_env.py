import numpy as np
import gymnasium as gym
from collections import deque

class RLMealWrapper(gym.Wrapper):
    metadata = {"render_modes": []}
    render_mode = None  

    def __init__(self, env):
        self.env = env
        
        # 论文参数：BG和胰岛素看过去4小时(48个点)，饮食看过去2小时(24个点)
        self.history_length_bg = 48 
        self.history_length_insulin = 48  
        self.history_length_meal = 24  
        
        self.bg_history = deque(maxlen=self.history_length_bg)
        self.insulin_history = deque(maxlen=self.history_length_insulin)
        self.meal_history = deque(maxlen=self.history_length_meal)
        
        # 展平后的状态维度: 48 + 48 + 24 = 120
        # 状态空间范围建议设为 [-inf, inf] 或 [0, 1] (如果归一化)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.history_length_bg + self.history_length_insulin + self.history_length_meal,), 
            dtype=np.float32
        )
        # 动作空间：基础率 (0 ~ 0.05 U/min 左右，视具体 pump 参数而定)
        self.action_space = gym.spaces.Box(low=0, high=0.1, shape=(1,), dtype=np.float32)

    def _get_scalar(self, val):
        if hasattr(val, 'CGM'):
            return float(val.CGM)
        if isinstance(val, (np.ndarray, np.generic)):
            try:
                return float(val.item())
            except ValueError:
                return float(val.ravel()[0])
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    def reset(self, seed=None, options=None):
        try:
            ret = self.env.reset(seed=seed, options=options)
        except TypeError:
            ret = self.env.reset()
        
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
        else:
            obs = ret
            info = {}
        
        current_bg = self._get_scalar(obs)
        
        self.bg_history.clear()
        self.insulin_history.clear()
        self.meal_history.clear()

        # 初始化历史记录，填满初始值
        for _ in range(self.history_length_bg):
            self.bg_history.append(current_bg)
        for _ in range(self.history_length_insulin):
            self.insulin_history.append(0.0)
        for _ in range(self.history_length_meal):
            self.meal_history.append(0.0)
            
        return self._get_state(), info

    def step(self, action):
        action_val = self._get_scalar(action)
        # 执行动作
        step_result = self.env.step(action_val)
        
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        elif isinstance(step_result, tuple) and len(step_result) == 4:
            obs, reward, done, info = step_result
            truncated = False
        else:
            obs = step_result
            reward = 0
            done = False
            truncated = False
            info = {}

        current_bg = self._get_scalar(obs)
        # 尝试从 info 或 env 中获取真实的 meal 数据
        # simglucose 的 CHO_hist 通常记录了碳水摄入
        if hasattr(self.env, 'CHO_hist') and len(self.env.CHO_hist) > 0:
             current_meal = self.env.CHO_hist[-1]
        else:
             current_meal = info.get('meal', 0) * 1.0
        
        self.bg_history.append(current_bg)
        self.insulin_history.append(action_val)
        self.meal_history.append(current_meal)
        
        # 使用复现论文的 Reward 计算逻辑
        custom_reward = self._calculate_paper_reward(current_bg)
        
        # 如果 simglucose 判定结束 (通常是低血糖/高血糖严重越界)，则保持 done
        return self._get_state(), custom_reward, done, truncated, info

    def _get_state(self):
        # --- 关键修改：归一化 ---
        # 参考 simglucose_gym_env.py 的处理方式
        # 血糖归一化：除以 400 (常见上限)
        norm_bg = np.array(self.bg_history) / 400.0
        # 胰岛素归一化：乘以 10 (将 0.05 变成 0.5，使其数值范围更适合神经网络)
        norm_insulin = np.array(self.insulin_history) * 10.0
        # 饮食归一化：除以 20 (假设一餐 60g 变成 3.0，或者除以更大数值如 100)
        # 参考代码中是 / 20.
        norm_meal = np.array(self.meal_history) / 20.0
        
        return np.concatenate([
            norm_bg,
            norm_insulin,
            norm_meal
        ]).astype(np.float32)

    def _magni_risk(self, bg):
        # Magni et al. 风险函数
        if bg <= 1: bg = 1
        f_bg = 3.5506 * ((np.log(bg)**0.8353) - 3.7932)
        risk = 10 * (f_bg**2)
        return risk

    def _calculate_paper_reward(self, current_bg):
        # 实现论文公式 (3) 和参考代码逻辑
        # R = -risk - duration_penalty
        
        current_risk = self._magni_risk(current_bg)
        
        bg_list = list(self.bg_history)
        
        # 持续时间窗口：过去1小时 (12个点)
        duration_time = min(len(bg_list), 12)
        duration_risk = 0.0
        
        # 参考 reward_functions.py 中的 magni_reward_duration 实现
        # 只有当血糖连续处于异常状态时，才累加风险
        if current_bg > 180:
            # 高血糖情况：回溯过去，直到血糖不高于 180 停止
            for i in range(2, duration_time + 1):
                idx = -i
                if abs(idx) > len(bg_list): break
                val = bg_list[idx]
                if val < 180: # 中断
                    break
                duration_risk += self._magni_risk(val)
                
        elif current_bg < 70:
            # 低血糖情况：回溯过去，直到血糖不低于 70 停止
            for i in range(2, duration_time + 1):
                idx = -i
                if abs(idx) > len(bg_list): break
                val = bg_list[idx]
                if val > 70: # 中断
                    break
                duration_risk += self._magni_risk(val)
        
        # 论文公式：duration penalty 是平均值
        duration_penalty = duration_risk / 12.0
            
        total_reward = - current_risk - duration_penalty
        return total_reward