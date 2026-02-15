import numpy as np
import gymnasium as gym
from collections import deque

class RLMealWrapper(gym.Wrapper):
    # --- 核心修复 ---
    # 在这里直接定义 render_mode，覆盖 gym.Wrapper 的 property
    # 这样 SB3 读取时会直接拿到 None，而不会去查底层的 simglucose
    metadata = {"render_modes": []}
    render_mode = None  

    def __init__(self, env):
        self.env = env
        
        # 注意：不要在这里写 self.render_mode = None，会报错
        
        self.history_length_bg = 48 
        self.history_length_insulin = 48  
        self.history_length_meal = 24  
        
        self.bg_history = deque(maxlen=self.history_length_bg)
        self.insulin_history = deque(maxlen=self.history_length_insulin)
        self.meal_history = deque(maxlen=self.history_length_meal)
        
        # 展平后的状态维度: 48 + 48 + 24 = 120
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.history_length_bg + self.history_length_insulin + self.history_length_meal,), 
            dtype=np.float32
        )
        # 动作空间：基础率
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

        for _ in range(self.history_length_bg):
            self.bg_history.append(current_bg)
        for _ in range(self.history_length_insulin):
            self.insulin_history.append(0.0)
        for _ in range(self.history_length_meal):
            self.meal_history.append(0.0)
            
        return self._get_state(), info

    def step(self, action):
        action_val = self._get_scalar(action)
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
        current_meal = info.get('meal', 0) * 1.0 
        
        self.bg_history.append(current_bg)
        self.insulin_history.append(action_val)
        self.meal_history.append(current_meal)
        
        custom_reward = self._calculate_paper_reward(current_bg)
        
        return self._get_state(), custom_reward, done, truncated, info

    def _get_state(self):
        return np.concatenate([
            np.array(self.bg_history),
            np.array(self.insulin_history),
            np.array(self.meal_history)
        ]).astype(np.float32)

    def _magni_risk(self, bg):
        if bg <= 1: bg = 1
        f_bg = 3.5506 * (np.log(bg)**0.8353) - 3.7932
        risk = 10 * (f_bg**2)
        return risk

    def _calculate_paper_reward(self, current_bg):
        current_risk = self._magni_risk(current_bg)
        
        bg_list = list(self.bg_history)
        last_hour_bgs = bg_list[-12:] 
        
        duration_penalty = 0.0
        
        if current_bg > 180: 
            sum_risk = sum([self._magni_risk(b) for b in last_hour_bgs if b > 180])
            duration_penalty = sum_risk / 12.0
            
        elif current_bg < 70: 
            sum_risk = sum([self._magni_risk(b) for b in last_hour_bgs if b < 70])
            duration_penalty = sum_risk / 12.0
            
        total_reward = - current_risk - duration_penalty
        return total_reward