import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from simglucose.envs import T1DSimEnv
from custom_env import RLMealWrapper
import matplotlib.pyplot as plt

def main():
    # --- 1. 设置模型路径 ---
    # 如果你想测试中间的 checkpoint，可以修改这里，比如 "./logs/checkpoints/rl_meal_model_10000_steps"
    model_path = "final_rl_meal_model" 
    
    print(f"正在加载模型: {model_path} ...")

    # --- 2. 创建测试环境 ---
    # 注意：测试环境必须和训练环境的结构保持一致 (Wrapper 顺序要一样)
    # 我们可以换一个病人试试泛化能力，或者用同一个病人测试效果
    # patient_name 可选: adolescent#001 - #010, adult#001 - #010, child#001 - #010
    patient = 'adolescent#001' 
    print(f"创建仿真环境，病人: {patient}")
    
    base_env = T1DSimEnv(patient_name=patient)
    env = RLMealWrapper(base_env)

    # --- 3. 加载模型 ---
    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}.zip，请检查文件名或路径。")
        return

    # --- 4. 开始测试循环 ---
    obs, info = env.reset()
    done = False
    truncated = False
    
    total_reward = 0
    steps = 0
    
    print("开始仿真控制...")
    
    # 记录数据用于简单的统计
    bg_history = [] 

    while not (done or truncated):
        # deterministic=True 很重要！
        # 在训练时我们需要随机探索，但在测试/部署时，我们需要模型输出它认为“最好”的动作。
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # 获取当前血糖 (从 info 中获取更准确，或者从 obs 解包)
        # simglucose 的 info 包含 'bg'
        if 'bg' in info:
            bg_history.append(info['bg'])
            
        # 可选：打印每一步的信息
        # if steps % 100 == 0:
        #     print(f"Step {steps}: BG={info.get('bg', 'N/A'):.2f}, Action={action}, Reward={reward:.2f}")

    print("仿真结束！")
    print("-" * 30)
    print(f"总步数: {steps}")
    print(f"总奖励: {total_reward:.2f}")
    
    # --- 5. 简单的统计分析 ---
    if bg_history:
        bg_array = np.array(bg_history)
        mean_bg = np.mean(bg_array)
        min_bg = np.min(bg_array)
        max_bg = np.max(bg_array)
        
        # 计算 Time in Range (TIR): 70-180 mg/dL
        tir = np.sum((bg_array >= 70) & (bg_array <= 180)) / len(bg_array) * 100
        # 计算低血糖率 (<70)
        hypo = np.sum(bg_array < 70) / len(bg_array) * 100
        # 计算高血糖率 (>180)
        hyper = np.sum(bg_array > 180) / len(bg_array) * 100

        print(f"平均血糖: {mean_bg:.2f} mg/dL")
        print(f"最低/最高: {min_bg:.2f} / {max_bg:.2f}")
        print(f"TIR (70-180): {tir:.2f}%  <-- 核心指标 (目标 >70%)")
        print(f"低血糖 (<70):  {hypo:.2f}%")
        print(f"高血糖 (>180): {hyper:.2f}%")

    # --- 6. 绘图 ---
    print("正在生成图表...")
    # simglucose 自带的渲染器，会生成 CVGA 图和时序图
    env.render() 
    
    # 注意：env.render() 在 simglucose 中通常会阻塞程序直到你关闭窗口
    # 如果没有弹出窗口，可能是 matplotlib backend 的问题，可以在代码开头加 import matplotlib; matplotlib.use('TkAgg')
    # 【新增】加上这两行，防止窗口直接关闭
    import matplotlib.pyplot as plt
    print("图表已生成。请关闭图表窗口以结束程序...")
    plt.show(block=True)
    
if __name__ == "__main__":
    main()