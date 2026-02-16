import warnings
# 放在最前面，屏蔽烦人的警告
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from simglucose.envs import T1DSimEnv
from custom_env import RLMealWrapper
import matplotlib.pyplot as plt

def magni_risk(bg):
  
    if bg <= 1: bg = 1

    f_bg = 3.5506 * ((np.log(bg)**0.8353) - 3.7932)
    
    risk = 10 * (f_bg**2)
    return risk

def calculate_lbgi_hbgi(bg_array):
    """
    计算 LBGI (低血糖风险指数) 和 HBGI (高血糖风险指数)
    这是糖尿病领域通用的评估标准
    """
    f_bg = 1.509 * ((np.log(bg_array)**1.084) - 5.381)
    rl = np.where(f_bg < 0, 10 * (f_bg**2), 0)
    rh = np.where(f_bg > 0, 10 * (f_bg**2), 0)
    return np.mean(rl), np.mean(rh)

def main():
    # --- 1. 设置参数 ---
    model_path = "logs/checkpoints/rl_meal_sac_2000000_steps" 
    # 论文中使用了 adult#006, adolescent#006, child#006 作为典型案例 
    patient = 'adult#003'  
    
    # 论文设定：测试时长为 14 天 
    TEST_HORIZON_DAYS = 14 
    
    print(f"正在加载模型: {model_path} ...")
    print(f"创建仿真环境，病人: {patient}")
    
    # --- 环境初始化 ---
    base_env = T1DSimEnv(patient_name=patient)
    
    # --- 关键修改：强制设置采样时间为 5 分钟 ---
    # 这样一天就是 288 步 (24 * 12)
    base_env.sample_time = 5
    if hasattr(base_env, 'sensor'):
        base_env.sensor.sample_time = 5
        
    env = RLMealWrapper(base_env)

    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}.zip")
        return

    # --- 2. 开始测试 ---
    obs, info = env.reset()
    
    # 有些旧版本的 simglucose reset 返回的是 tuple，有些是单个 obs
    if isinstance(obs, tuple):
        obs = obs[0]
        
    done = False
    truncated = False
    
    bg_history = []
    risk_history = []
    
    # 计算目标总步数：14天 * 24小时 * 12步/小时 (60/5)
    total_steps_target = TEST_HORIZON_DAYS * 288
    
    print(f"开始 {TEST_HORIZON_DAYS} 天仿真测试 (目标步数: {total_steps_target})...")

    while not (done or truncated):
        # 使用 deterministic=True 进行评估，这是测试 RL 模型的标准做法
        action, _ = model.predict(obs, deterministic=True)
        
        step_result = env.step(action)
        
        # 兼容不同 gym 版本的返回值解包
        if len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            truncated = False
        
        # 记录数据
        # 优先从 info 获取真实 BG，如果没有则尝试从 observation 反推 (不推荐)
        if 'bg' in info:
            bg_val = info['bg']
        elif hasattr(base_env, 'CGM_hist'):
             bg_val = base_env.CGM_hist[-1]
        else:
             bg_val = 0 # Should not happen
             
        bg_history.append(bg_val)
        risk_history.append(magni_risk(bg_val))
        
        # 检查是否达到目标天数
        if len(bg_history) >= total_steps_target:
             print(f"达到 {TEST_HORIZON_DAYS} 天测试时长，停止测试。")
             break

    print("-" * 30)
    if len(bg_history) == 0:
        print("错误：未收集到数据，可能环境直接终止了。")
        return

    # --- 3. 计算论文核心指标 ---
    bg_array = np.array(bg_history)
    
    # A. Risk (论文核心指标)
    mean_risk = np.mean(risk_history)
    
    # B. TIR / Hypo / Hyper (论文核心指标)
    # 正常范围: [70, 180]
    tir = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100
    # 低血糖: < 70
    hypo = np.sum(bg_array < 70) / len(bg_array) * 100
    # 高血糖: > 180
    hyper = np.sum(bg_array > 180) / len(bg_array) * 100
    
    # C. LBGI / HBGI (辅助分析)
    lbgi, hbgi = calculate_lbgi_hbgi(bg_array)

    print(f"测试结果 (Patient: {patient}, Days: {TEST_HORIZON_DAYS})")
    print(f"总步数: {len(bg_array)}")
    print(f"平均血糖: {np.mean(bg_array):.2f} mg/dL")
    print("-" * 20)
    print(f"Risk (Magni): {mean_risk:.2f}   (越低越好)")
    print(f"TIR (70-180): {tir:.2f}%     (越高越好)")
    print(f"Hypo (<70):   {hypo:.2f}%    ")
    print(f"Hyper (>180): {hyper:.2f}%   ")
    print("-" * 20)
    print(f"LBGI (低血糖风险): {lbgi:.2f}")
    print(f"HBGI (高血糖风险): {hbgi:.2f}")

    # --- 4. 绘图 ---
    plt.figure(figsize=(12, 6))
    plt.plot(bg_history, label='Blood Glucose', linewidth=1)
    # 画出安全范围
    plt.axhline(y=180, color='r', linestyle='--', alpha=0.5, label='Hyper Limit (180)')
    plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Hypo Limit (70)')
    # 画出目标值参考线 (140)
    plt.axhline(y=140, color='g', linestyle=':', alpha=0.3, label='Target (140)')
    
    plt.title(f'14-Day Simulation: {patient}')
    plt.ylabel('BG (mg/dL)')
    plt.xlabel('Steps (5 min/step)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()