import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile

# 绘图字体设置
# 使用内置 mathtext 渲染数学公式，避免依赖系统 LaTeX
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix' # 使用 STIX 字体以获得更好的数学符号显示效果

# 配置部分
# CPP_EXECUTABLE 指向编译好的可执行文件路径
CPP_EXECUTABLE = "/home/dayun/PhaseTracerForMe/bin/run_IDM" 

def run_single_simulation(args):
    """
    单个点的模拟函数，供多进程调用
    args: (mA, mH, other_fixed_params_dict)
    """
    mA, mH, fixed_params = args
    
    try:
        # 【修正】明确设置 mHpm。如果物理模型要求 mHpm 独立，请将其加入 fixed_params
        # 假设原意是 mHpm 随 mA 变化 (简并情况)
        current_params = fixed_params.copy()
        current_params['mHpm'] = mA 
        
        # 【修改】增加参数合法性预检查，避免传入明显非法的值给 C++
        if mA <= 0 or mH <= 0 or current_params['mHpm'] <= 0:
            return mA, mH, np.nan
            
        # 【修改】生成临时输入文件供 C++ 可执行文件读取
        # C++ 代码中硬编码了读取 "/home/dayun/IDM_input.txt"
        # 为了支持并行扫描，最好每个进程使用唯一的文件名，或者确保串行执行且文件写入原子性
        # 这里为了简单且配合原 C++ 代码硬编码路径，我们写入固定路径。
        # 注意：如果并行执行，这会引发竞争条件。建议修改 C++ 代码接受文件名参数，或使用线程锁。
        # 鉴于原代码是串行循环，这里暂时保持写入固定路径。
        input_file_path = "/home/dayun/IDM_input.txt"
        
        with open(input_file_path, 'w') as f:
            f.write(f"lam2 = {current_params['lam2']}\n")
            f.write(f"lamL = {current_params['lamL']}\n")
            f.write(f"mA = {mA}\n")
            f.write(f"mH = {mH}\n")
            f.write(f"mHpm = {current_params['mHpm']}\n")
            f.write(f"paramNumber = {current_params['paramNumber']}\n")
            f.write(f"Resum = {current_params['Resum']}\n")
            
        # 【修改】在调用 C++ 函数前打印参数，并强制刷新缓冲区
        print(f"Processing: mA={mA:.4f}, mH={mH:.4f}, mHpm={current_params['mHpm']:.4f}", flush=True)
        
        # 调用 C++ 可执行文件
        # 捕获 stdout 和 stderr
        result = subprocess.run(
            [CPP_EXECUTABLE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60 # 设置超时时间，防止单个点卡死
        )
        
        # 检查返回码
        if result.returncode != 0:
            print(f"Warning: C++ process failed for mA={mA}, mH={mH}. Stderr: {result.stderr}", flush=True)
            return mA, mH, np.nan
            
        # 解析输出
        output_str = result.stdout.strip()
        if not output_str:
            return mA, mH, np.nan
            
        try:
            v_over_T = float(output_str)
        except ValueError:
            print(f"Warning: Could not parse output '{output_str}' for mA={mA}, mH={mH}", flush=True)
            return mA, mH, np.nan
            
        # pybind11 返回的 nan 在 Python 中也是 nan
        if np.isnan(v_over_T):
            return mA, mH, np.nan
            
        return mA, mH, v_over_T
            
    except subprocess.TimeoutExpired:
        print(f"Error: Timeout for mA={mA}, mH={mH}", flush=True)
        return mA, mH, np.nan
    except Exception as e:
        # 记录错误以便调试，包含参数信息
        import traceback
        error_msg = f"Error at mA={mA}, mH={mH}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        return mA, mH, np.nan

def main():
    # 1. 定义扫描范围
    # 建议缩小范围或使用对数空间，如果线性空间大部分无效的话
    # 这里保持原样，但增加诊断
    mA_values = np.linspace(200, 400, 20) # 示例范围
    mH_values = np.linspace(60, 200, 20) # 示例范围
    
    # 固定参数 (移除 mHpm，因为它将动态设置)
    fixed_params = {
        'lam2': 0.2,      # 示例值
        'lamL': 0.0015,      # 示例值
        'paramNumber': "1",
        'Resum': "DJ"
    }
    
    # 2. 构建任务列表
    tasks = []
    for mA in mA_values:
        for mH in mH_values:
            tasks.append((mA, mH, fixed_params))
            
    print(f"Starting scan with {len(tasks)} points...")
    
    # 3. 串行执行 (取消多线程/多进程)
    results = []
    
    # 用于统计错误类型
    error_count = 0
    nan_count = 0
    valid_count = 0
    
    # 使用简单的 for 循环替代 ProcessPoolExecutor
    for task in tasks:
        mA, mH, _ = task
        try:
            # 直接调用函数，不再通过 executor.submit
            result = run_single_simulation(task)
            mA_res, mH_res, vot = result
            
            if np.isnan(vot):
                nan_count += 1
                # 打印前几个 NaN 点的参数，方便调试
                if nan_count <= 5:
                    print(f"Debug: NaN result for mA={mA_res:.2f}, mH={mH_res:.2f}. Check C++ stderr for details.")
            else:
                valid_count += 1
                
            results.append(result)
            
        except Exception as e:
            # 捕获执行过程中的异常
            error_count += 1
            # 【修改】增加更详细的错误日志，包括参数值
            print(f"Task FAILED for mA={mA:.4f}, mH={mH:.4f}: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # 添加一个无效结果以保持数据对齐
            results.append((mA, mH, np.nan))
            
    # 4. 数据处理
    mA_res = []
    mH_res = []
    v_over_T_res = []
    
    for mA, mH, vot in results:
        mA_res.append(mA)
        mH_res.append(mH)
        v_over_T_res.append(vot)
        
    mA_res = np.array(mA_res)
    mH_res = np.array(mH_res)
    v_over_T_res = np.array(v_over_T_res)
    
    # 统计成功与失败的比例，用于诊断
    total_valid = np.sum(~np.isnan(v_over_T_res))
    total_nan = np.sum(np.isnan(v_over_T_res))
    print(f"Scan completed.")
    print(f"Valid points: {total_valid}/{len(v_over_T_res)}")
    print(f"NaN points: {total_nan}/{len(v_over_T_res)}")
    print(f"Crashed/Timeout tasks: {error_count}")
    
    if total_valid == 0:
        print("WARNING: No valid points found. The parameter space might be invalid or constraints are too tight.")
        print("Check the 'Debug: NaN result...' messages above and corresponding C++ stderr output.")

    # 5. 绘图
    plt.figure(figsize=(10, 8))
    
    # 【修改】将一维结果重塑为二维网格以用于 contourf
    # 获取网格维度
    n_mA = len(mA_values)
    n_mH = len(mH_values)
    
    # 确保数据长度匹配网格大小
    if len(mA_res) == n_mA * n_mH:
        # 重塑数组为 (n_mA, n_mH)
        # 注意：scan_idm.py 中的循环顺序是 outer mA, inner mH
        # 所以 reshape 时默认顺序 (C-style) 对应 mA 为行，mH 为列
        mA_grid = mA_res.reshape((n_mA, n_mH))
        mH_grid = mH_res.reshape((n_mA, n_mH))
        vot_grid = v_over_T_res.reshape((n_mA, n_mH))
        
        # 创建绘图用的 Meshgrid
        # contourf 期望 X 和 Y 是二维坐标矩阵
        # np.meshgrid 默认 indexing='xy'，即 X 对应列变化，Y 对应行变化
        # 我们的数据中，mH 是内层循环（列方向），mA 是外层循环（行方向）
        # 因此 X 轴对应 mH，Y 轴对应 mA
        X, Y = np.meshgrid(mH_values, mA_values)
        
        # 绘制等高线图
        # 注意：如果 vot_grid 中有 nan，contourf 可能会在这些区域留白
        scatter = plt.contourf(X, Y, vot_grid, cmap='viridis')
        plt.colorbar(scatter, label='$v_n/T_n$')
        plt.title('IDM Phase Transition Strength Scan')
        plt.xlabel('$m_H$ [GeV]')
        plt.ylabel('$m_A$ [GeV]')
    else:
        print(f"Warning: Data size mismatch for reshaping. Expected {n_mA * n_mH}, got {len(mA_res)}. Falling back to scatter plot if possible.")
        #  fallback 逻辑可选，这里保持原意主要修复 contourf 报错
        mask = ~np.isnan(v_over_T_res)
        if np.any(mask):
             plt.scatter(mH_res[mask], mA_res[mask], c=v_over_T_res[mask], cmap='viridis', s=10)
             plt.colorbar(label='$v_n/T_n$')
             plt.title('IDM Phase Transition Strength Scan (Scatter Fallback)')
             plt.xlabel('$m_H$ [GeV]')
             plt.ylabel('$m_A$ [GeV]')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("/home/dayun/fig/DJ.png", dpi=300)
    print("Plot saved to /home/dayun/fig/DJ.png")

if __name__ == "__main__":
    main()