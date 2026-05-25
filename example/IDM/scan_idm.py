import numpy as np
import matplotlib.pyplot as plt
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


plt.rc('text', usetex=True)
plt.rc('font', family='serif') 


CPP_EXECUTABLE = "/home/dayun/PhaseTracerForMe/bin/run_IDM" 

def run_single_simulation(args: tuple):
    """
    单个点的模拟函数，供多进程调用
    params: (i, j, dict[lam2, lamL, mA, mH, mHpm, paramNumber, Resum])
    """
    index_i, index_j, params = args
    try:    
        cmd = [
            CPP_EXECUTABLE,
            "--lam2", str(params['lam2']),
            "--lamL", str(params['lamL']),
            "--mA", str(params['mA']),
            "--mH", str(params['mH']),
            "--mHpm", str(params['mHpm']), 
            "--paramNumber", str(params['paramNumber']),
            "--Resum", str(params['Resum'])
        ]
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 检查返回码
        if result.returncode != 0:
            #print(f"Warning: C++ process failed for mA={mA}, mH={mH}. Stderr: {result.stderr}", flush=True)
            return index_i, index_j, np.nan, np.nan, np.nan  # 修复：始终返回三个值
            
        # 解析输出
        # 获取所有非空行，并取最后一行作为结果
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return index_i, index_j, np.nan, np.nan, np.nan  # 修复：始终返回三个值
            
        output_str = lines[-1]
         
        try:
            parts = output_str.split()
            v_over_T = np.abs(float(parts[0]))
            alpha = np.abs(float(parts[1]))
            beta_H = np.abs(float(parts[2]))
        except (ValueError, IndexError):
            #print(f"Warning: Could not parse output '{output_str}' for mA={mA}, mH={mH}", flush=True)
            return index_i, index_j, np.nan, np.nan, np.nan
            
        if np.isnan(v_over_T):
            return index_i, index_j, np.nan, np.nan, np.nan
        return index_i, index_j, v_over_T, alpha, beta_H
                
    except Exception as e:
        # 记录错误以便调试，包含参数信息
        #import traceback
        #error_msg = f"Error at mA={mA}, mH={mH}: {str(e)}\n{traceback.format_exc()}"
        #print(error_msg, flush=True)
        return index_i, index_j, np.nan, np.nan, np.nan

def plot_PT(mH_values: np.ndarray, mA_values: np.ndarray, PT_param_values: np.ndarray, filename: str, title: str):
    # 5. 绘图
    plt.figure()
    X, Y = np.meshgrid(mH_values, mA_values)
    CF = plt.contourf(X, Y, PT_param_values, cmap='viridis')
    cbar = plt.colorbar(CF, label=title)
    cbar.ax.yaxis.label.set_size(20)
    cbar.ax.tick_params(labelsize=17)
    #plt.title('IDM Phase Transition Strength Scan')
    plt.xlabel('$m_H$ [GeV]', fontsize = 20)
    plt.ylabel('$m_A$ [GeV]', fontsize = 20)
    
    # 显式设置坐标刻度，确保一致性
    plt.tick_params(labelsize=17)
    
    # X 轴：固定间隔 25 GeV
    plt.xlim(mH_values.min(), mH_values.max())
    plt.xticks(np.arange(60, 201, 25))  # 60, 85, 110, 135, 160, 185, 200
    
    # Y 轴：固定间隔 25 GeV
    plt.ylim(mA_values.min(), mA_values.max())
    plt.yticks(np.arange(200, 401, 25))  # 200, 225, 250, 275, 300, 325, 350, 375, 400
    
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"/home/dayun/fig/{filename}.png", dpi=300)
    print(f"Plot saved to /home/dayun/fig/{filename}.png")

def main():
    """
    主函数，用于执行模拟任务
    """
    mA_values = np.linspace(200, 400, 20) # 示例范围
    mH_values = np.linspace(60, 200, 20) # 示例范围
    
    n_mA = len(mA_values)
    n_mH = len(mH_values)
    
    # 预分配结果数组
    v_over_T_grid = np.full((n_mA, n_mH), np.nan)
    alpha_grid = np.full((n_mA, n_mH), np.nan)
    beta_H_grid = np.full((n_mA, n_mH), np.nan)
    
    # 2. 构建任务列表，携带索引 (i, j)
    tasks = []
    for i, mA in enumerate(mA_values):
        for j, mH in enumerate(mH_values):   
            params = {
                'lam2': 0.2,      # 示例值
                'lamL': 0.0015,      # 示例值
                'mA': format(mA, '.3f'),
                'mH': format(mH, '.3f'),
                'mHpm': format(mA, '.3f'),
                'paramNumber': "1",
                'Resum': "Parwani"
            }
            # 使用包含独立参数副本的元组
            tasks.append((i, j, params))
            
    print(f"Starting parallel scan with {len(tasks)} points...")
    
    # 3. 并行执行
    # 用于统计错误类型
    error_count = 0
    nan_count = 0
    valid_count = 0
    
    # 尝试导入 tqdm 以显示进度条，如果不可用则使用简单计数
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        completed_count = 0
        
    # 使用 ProcessPoolExecutor 进行并行计算
    with ProcessPoolExecutor() as executor:
        # 提交所有任务
        future_to_task = {executor.submit(run_single_simulation, task): task for task in tasks}
        
        # 处理完成的任务
        iterator = as_completed(future_to_task)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(tasks), desc="Scanning Parameters")
            
        for future in iterator:
            try:
                idx_i, idx_j, vot, alpha, beta_H = future.result()      
                if np.isnan(vot):
                    nan_count += 1
                    # if nan_count <= 5:
                    #     print(f"Debug: NaN result for mA={mA:.2f}, mH={mH:.2f}. Check C++ stderr for details.")
                else:
                    valid_count += 1
                    
                # 直接填入预分配的数组中，保证位置正确
                v_over_T_grid[idx_i, idx_j] = vot
                alpha_grid[idx_i, idx_j] = alpha
                beta_H_grid[idx_i, idx_j] = beta_H  
                
            except Exception as e:
                # 捕获执行过程中的异常
                error_count += 1
                print(f"Task FAILED for mA={mA_values[idx_i]:.6f}, mH={mH_values[idx_j]:.6f}: {type(e).__name__}: {e}", flush=True)
                # import traceback
                # traceback.print_exc()
                # 保持默认为 nan，无需额外操作，因为预分配时已设为 nan
            
            # 如果不使用 tqdm，每完成一定数量任务打印一次进度
            if not use_tqdm:
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(tasks):
                    print(f"Progress: {completed_count}/{len(tasks)} tasks completed.", flush=True)
            
    
    # 统计成功与失败的比例，用于诊断
    total_valid = np.sum(~np.isnan(v_over_T_grid))
    total_nan = np.sum(np.isnan(v_over_T_grid))
    print(f"Scan completed.")
    print(f"Valid points: {total_valid}/{v_over_T_grid.size}")
    print(f"NaN points: {total_nan}/{v_over_T_grid.size}")
    print(f"Crashed/Timeout tasks: {error_count}")
    
    if total_valid == 0:
        print("WARNING: No valid points found. The parameter space might be invalid or constraints are too tight.")
        print("Check the 'Debug: NaN result...' messages above and corresponding C++ stderr output.")

    # 4. 绘图
    plot_PT(mH_values, mA_values, v_over_T_grid, "Parwani_vot", "$v_n/T_n$")
    plot_PT(mH_values, mA_values, alpha_grid, "Parwani_alpha", "$\\alpha$")
    plot_PT(mH_values, mA_values, beta_H_grid, "Parwani_beta_H", "$\\beta/H$")
    

if __name__ == "__main__":
    main()