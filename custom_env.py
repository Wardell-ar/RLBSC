import numpy as np
import gymnasium as gym
from collections import deque
# 注意：我们不再需要导入 Action，因为我们只传 float 给底层环境
# from simglucose.controller.base import Action 

class RLMealWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        
        self.history_length_4h = 48 
        self.history_length_2h = 24  
        
        self.bg_history = deque(maxlen=self.history_length_4h)
        self.insulin_history = deque(maxlen=self.history_length_4h)
        self.meal_history = deque(maxlen=self.history_length_2h)
        
        # 展平后的状态维度: 48 + 48 + 24 = 120
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, 
            shape=(self.history_length_4h + self.history_length_4h + self.history_length_2h,), 
            dtype=np.float32
        )
        # 动作空间：基础率 (0 ~ 30 U/h)
        self.action_space = gym.spaces.Box(low=0, high=0.5, shape=(1,), dtype=np.float32)

    def _get_scalar(self, val):
        """
        稳健的数值转换函数
        """
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
        # 这里的 self.env 是 simglucose 的 GymEnv，它可能只接受 reset() 不带参数
        # 为了兼容性，我们尝试不带参数调用
        ret = self.env.reset()
        
        # 兼容 simglucose 可能返回 (obs, info) 或仅 obs
        if isinstance(ret, tuple) and len(ret) == 2:
            obs = ret[0]
            info = ret[1]
        else:
            obs = ret
            info = {}
        
        current_bg = self._get_scalar(obs)
        
        # 清空并初始化 Buffer
        self.bg_history.clear()
        self.insulin_history.clear()
        self.meal_history.clear()

        for _ in range(self.history_length_4h):
            self.bg_history.append(current_bg)
        for _ in range(self.history_length_4h):
            self.insulin_history.append(0.0)
        for _ in range(self.history_length_2h):
            self.meal_history.append(0.0)
            
        return self._get_state(), info

    def step(self, action):
        # --- 核心修改 ---
        # 1. 确保 action 是纯 float
        action_val = self._get_scalar(action)
        
        # 2. 【关键】直接传 float 给 self.env
        # 解释: self.env 是 simglucose.envs.T1DSimEnv。
        # 它内部的代码会自动做: act = Action(basal=action_val, bolus=0)
        # 如果我们在外面这里再包一层 Action(...)，它就会变成 Action(basal=Action(...))，导致报错。
        step_result = self.env.step(action_val)
        
        # 兼容返回值
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            truncated = False
        elif len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        else:
            obs = step_result
            reward = 0
            done = False
            truncated = False
            info = {}

        current_bg = self._get_scalar(obs)
        current_meal = info.get('meal', 0) 
        
        self.bg_history.append(current_bg)
        self.insulin_history.append(action_val)
        self.meal_history.append(current_meal)
        
        custom_reward = self._calculate_reward()
        
        return self._get_state(), custom_reward, done, truncated, info

    def _get_state(self):
        return np.concatenate([
            np.array(self.bg_history),
            np.array(self.insulin_history),
            np.array(self.meal_history)
        ]).astype(np.float32)

    def _calculate_reward(self):
        bg_last_hour = list(self.bg_history)[-12:]
        current_bg = bg_last_hour[-1]
        
        # 1. 基础 Risk 计算 (Magni Risk Function)
        # 这是 simglucose 的标准计算方式
        if current_bg <= 1: current_bg = 1 
        try:
            log_bg = np.log(current_bg)
            term = 3.5506 * (max(0, log_bg)**0.8353) - 3.7932
            risk_val = 10 * (term**2)
        except:
            risk_val = 100

        # 2. 严重的低血糖惩罚 (Hypo Panic)
        # 只要血糖低于 100，就开始扣分；低于 70，扣大分！
        hypo_penalty = 0
        if current_bg < 100:
            # 这是一个指数级惩罚，血糖越低，罚得越狠
            # 例如: bg=90 -> 罚款 ~100
            #       bg=60 -> 罚款 ~1000
            hypo_penalty = 100 * ((100 - current_bg) ** 1.5)

        # 3. 胰岛素平滑惩罚 (Action Smoothness)
        # 鼓励动作平滑，不要突然猛打
        last_action = self.insulin_history[-1]
        action_penalty = last_action * 5.0

        # 总奖励
        # 我们希望 Risk 越小越好，所以取负
        total_reward = -risk_val - hypo_penalty - action_penalty
        return total_reward