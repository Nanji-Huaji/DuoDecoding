import subprocess
import os
import sys
from datetime import datetime


def run_command(command, log_file, description):
    """
    运行shell命令并将输出重定向到日志文件

    Args:
        command: 要执行的命令
        log_file: 输出日志文件名
        description: 脚本描述
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {description}...")

    try:
        # 使用subprocess.run确保命令完成后再继续
        with open(log_file, "w", encoding="utf-8") as f:
            result = subprocess.run(
                command,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                cwd="/home/tiantianyi/code/DuoDecoding",  # 设置工作目录
                check=False,  # 不在非零退出码时抛出异常
            )

        if result.returncode == 0:
            print(f"   {description} 完成，输出已保存到 {log_file}")
        else:
            print(f"   {description} 执行失败 (退出码: {result.returncode})，输出已保存到 {log_file}")

        return result.returncode

    except Exception as e:
        print(f"   执行 {description} 时发生错误: {str(e)}")
        return -1


def main():
    print("开始运行基线实验脚本...")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 确保工作目录存在
    work_dir = "/home/tiantianyi/code/DuoDecoding"
    if not os.path.exists(work_dir):
        print(f"错误: 工作目录 {work_dir} 不存在")
        sys.exit(1)

    # 要执行的脚本列表 (命令, 日志文件, 描述)
    scripts = [
        (
            "zsh /home/tiantianyi/code/DuoDecoding/cmds/precise/precise_speeculative_decoding.sh",
            "prcise_spec.log",
            "运行 precise_speeculative_decoding.sh",
        ),
        (
            "zsh /home/tiantianyi/code/DuoDecoding/cmds/precise/precise_uncertainty_decoding.sh",
            "prcise_uncertainty_decoding_log.log",
            "运行 precise_uncertainty_decoding.sh",
        ),
        (
            "zsh /home/tiantianyi/code/DuoDecoding/cmds/precise/precise_tridecoding.sh",
            "precise_tri_qual_exp.log",
            "运行 precise_tridecoding.sh",
        ),
        (
            "zsh /home/tiantianyi/code/DuoDecoding/cmds/precise/precise_full_prob_speculative_decoding.sh",
            "precise_full_prob_speculative_decoding.log",
            "运行 precise_full_prob_speculative_decoding.sh",
        ),
    ]

    failed_scripts = []

    # 顺序执行每个脚本
    for i, (command, log_file, description) in enumerate(scripts, 1):
        print(f"\n{i}. {description}...")

        # 检查脚本文件是否存在
        script_path = command.split()[-1]  # 获取脚本路径
        if not os.path.exists(script_path):
            print(f"   警告: 脚本文件 {script_path} 不存在")
            failed_scripts.append(description)
            continue

        # 执行脚本
        return_code = run_command(command, log_file, description)

        if return_code != 0:
            failed_scripts.append(description)

    # 输出总结
    print(f"\n{'='*50}")
    print(f"所有基线实验脚本执行完成！")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if failed_scripts:
        print(f"\n执行失败的脚本:")
        for script in failed_scripts:
            print(f"  - {script}")
        sys.exit(1)
    else:
        print("\n所有脚本都成功执行！")


if __name__ == "__main__":
    main()
