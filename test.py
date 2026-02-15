import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from simglucose.envs import T1DSimEnv
from custom_env import RLMealWrapper
import matplotlib.pyplot as plt

def magni_risk(bg):
    """
    计算 Magni Risk (论文公式 1) [cite: 126]
    """
    if bg <= 1: bg = 1
    f_bg = 3.5506 * (np.log(bg)**0.8353) - 3.7932
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
    model_path = "final_rl_meal_model" 
    # 论文中使用了 adult#006, adolescent#006, child#006 作为典型案例 
    patient = 'adolescent#006'  
    
    # 论文设定：测试时长为 14 天 
    # simglucose 默认采样通常为 3分钟(1天480步) 或 5分钟(1天288步)
    # 无论哪种，我们通过模拟时长来控制最稳妥
    TEST_HORIZON_DAYS = 14 
    
    print(f"正在加载模型: {model_path} ...")
    print(f"创建仿真环境，病人: {patient}")
    
    # 注意：这里需要重新实例化环境，确保和训练时一致
    base_env = T1DSimEnv(patient_name=patient)
    env = RLMealWrapper(base_env)

    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}.zip")
        return

    # --- 2. 开始测试 ---
    obs, info = env.reset()
    done = False
    truncated = False
    
    bg_history = []
    risk_history = []
    
    # 假设每步 3 分钟 (simglucose 默认)，14天 = 6720步
    # 我们用 while 循环配合天数检查
    current_day = 0
    steps = 0
    max_steps = 14 * 480 # 预估最大步数，用于进度条，实际由 env 时间决定

    print(f"开始 14 天仿真测试...")

    while not (done or truncated):
        # 使用 deterministic=True 进行评估
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # 记录数据
        if 'bg' in info:
            bg_val = info['bg']
            bg_history.append(bg_val)
            risk_history.append(magni_risk(bg_val))
        
        steps += 1
        
        # 检查是否达到 14 天 (20160 分钟)
        # simglucose 的 info['time'] 是 datetime 对象，或者我们简单地用步数截断
        # 这里为了简单直接用步数强制截断 (假设 3分钟一步)
        if steps >= 14 * 288 * (5/3): # 粗略估算，或者直接跑满
             # 更稳妥的方式是看 bg_history 的长度
             # 实际上 simglucose 默认跑很久，我们手动 break 即可
             if len(bg_history) >= 14 * 288: # 假设 5分钟间隔，14天数据量
                 print("达到 14 天测试时长，停止测试。")
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
    tir = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100
    hypo = np.sum(bg_array < 70) / len(bg_array) * 100
    hyper = np.sum(bg_array > 180) / len(bg_array) * 100
    
    # C. LBGI / HBGI (辅助分析)
    lbgi, hbgi = calculate_lbgi_hbgi(bg_array)

    print(f"测试结果 (Patient: {patient}, Days: 14)")
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
    plt.plot(bg_history, label='Blood Glucose')
    # 画出安全范围
    plt.axhline(y=180, color='r', linestyle='--', alpha=0.5, label='Hyper Limit (180)')
    plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Hypo Limit (70)')
    plt.title(f'14-Day Simulation: {patient}')
    plt.ylabel('BG (mg/dL)')
    plt.xlabel('Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()