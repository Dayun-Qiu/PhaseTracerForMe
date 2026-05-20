import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


plt.rc('text', usetex=True)
plt.rc('font', family='serif') 


CPP_EXECUTABLE = "/home/dayun/PhaseTracerForMe/bin/run_IDM" 

def run_single_simulation(args):
    """
    单个点的模拟函数，供多进程调用
    params: (i, j, dict[lam2, lamL, mA, mH, mHpm, paramNumber, Resum])
    """
    _, _, params = args
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
            return np.nan
            
        # 解析输出
        # 获取所有非空行，并取最后一行作为结果
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return np.nan
            
        output_str = lines[-1]
        
        if not output_str:
            return np.nan 
        try:
            v_over_T = float(output_str)
        except ValueError:
            #print(f"Warning: Could not parse output '{output_str}' for mA={mA}, mH={mH}", flush=True)
            return np.nan
            
        if np.isnan(v_over_T):
            return np.nan      
        return v_over_T  
                
    except Exception as e:
        # 记录错误以便调试，包含参数信息
        #import traceback
        #error_msg = f"Error at mA={mA}, mH={mH}: {str(e)}\n{traceback.format_exc()}"
        #print(error_msg, flush=True)
        return np.nan

def main():
    """
    主函数，用于执行模拟任务
    """
    mA_values = np.linspace(200, 400, 20) # 示例范围
    mH_values = np.linspace(60, 200, 20) # 示例范围
    
    
    params = {
        'lam2': 0.2,      # 示例值
        'lamL': 0.0015,      # 示例值
        'mA': 0.0,
        'mH': 0.0,
        'mHpm': 0.0,
        'paramNumber': "1",
        'Resum': "DJ"
    }
    
    n_mA = len(mA_values)
    n_mH = len(mH_values)
    
    # 预分配结果数组
    v_over_T_grid = np.full((n_mA, n_mH), np.nan)
    
    # 2. 构建任务列表，携带索引 (i, j)
    tasks = []
    for i, mA in enumerate(mA_values):
        for j, mH in enumerate(mH_values):
            # 修复: 创建 params 的副本，避免所有任务共享同一个字典引用
            task_params = params.copy()
            
            # 传递索引 i, j 以便后续填充数组
            task_params['mA'] = mA
            task_params['mH'] = mH
            task_params['mHpm'] = mA
            
            # 使用包含独立参数副本的元组
            tasks.append((i, j, task_params))
            
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
            task = future_to_task[future]
            idx_i, idx_j, _ = task
            try:
                result = future.result()
                vot = result
                
                if np.isnan(vot):
                    nan_count += 1
                    # if nan_count <= 5:
                    #     print(f"Debug: NaN result for mA={mA:.2f}, mH={mH:.2f}. Check C++ stderr for details.")
                else:
                    valid_count += 1
                    
                # 直接填入预分配的数组中，保证位置正确
                v_over_T_grid[idx_i, idx_j] = vot
                
            except Exception as e:
                # 捕获执行过程中的异常
                error_count += 1
                # print(f"Task FAILED for mA={mA:.4f}, mH={mH:.4f}: {type(e).__name__}: {e}", flush=True)
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

    # 5. 绘图
    plt.figure(figsize=(10, 8))
    
    X, Y = np.meshgrid(mH_values, mA_values)
    CF = plt.contourf(X, Y, v_over_T_grid, cmap='viridis')
    cbar = plt.colorbar(CF, label='$v_n/T_n$')
    cbar.ax.yaxis.label.set_size(20)
    cbar.ax.tick_params(labelsize=17)
    plt.title('IDM Phase Transition Strength Scan')
    plt.xlabel('$m_H$ [GeV]', fontsize = 20)
    plt.ylabel('$m_A$ [GeV]', fontsize = 20)
    plt.tight_layout()
    plt.tick_params(labelsize=17)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("/home/dayun/fig/DJ.png", dpi=300)
    print("Plot saved to /home/dayun/fig/DJ.png")

if __name__ == "__main__":
    main()