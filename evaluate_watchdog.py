import sys
import os
import time
import subprocess
import threading
from datetime import datetime
import yaml

# 60초 동안 터미널 출력이 없으면 프로세스가 죽은 것으로 간주
TIMEOUT_SECONDS = 60 

def log_event(log_path, message):
    """터미널에 출력함과 동시에, Hydra 아웃풋 폴더 내의 로그 파일에 기록을 남깁니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(formatted_msg + "\n")

def monitor_output(process, last_output_time):
    """자식 프로세스의 출력을 가로채서 화면에 뿌려주고, 마지막 활동 시간을 갱신합니다."""
    for line in iter(process.stdout.readline, ''):
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
            last_output_time[0] = time.time()
    process.stdout.close()

def main():
    print("=" * 60)
    print("[Watchdog] Guardian System Activated")
    print("=" * 60)

    model_name = None 
    resume_dir = None
    filtered_args = []
    
    for arg in sys.argv[1:]:
        if arg.startswith("model="):
            model_name = arg.split("=")[1]
            filtered_args.append(arg)
        elif arg.startswith("--resume="):
            resume_dir = arg.split("=")[1]
        else:
            filtered_args.append(arg)

    if model_name is None:
        try:
            config_path = os.path.join(os.getcwd(), "conf", "eval_config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    for item in config.get("defaults", []):
                        if isinstance(item, dict) and "model" in item:
                            model_name = item["model"]
                            break
                    if not model_name and isinstance(config.get("model"), str):
                        model_name = config.get("model")
        except Exception as e:
            print(f"Failed to read yaml config: {e}")

    if not model_name:
        model_name = "default_model"

    # Hydra Output Directory 고정
    if resume_dir and os.path.exists(resume_dir):
        fixed_output_dir = resume_dir
        print(f"Manual Resume Mode Activated! Resuming from: {resume_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{model_name}_eval_{timestamp}"
        fixed_output_dir = os.path.join(os.getcwd(), "outputs", folder_name)
        os.makedirs(fixed_output_dir, exist_ok=True)
        print(f"Fresh Start Mode: {fixed_output_dir}")

    # Watchdog log file path
    watchdog_log_path = os.path.join(fixed_output_dir, "watchdog_guardian.log")
    
    log_event(watchdog_log_path, "Watchdog System Initialized.")
    log_event(watchdog_log_path, f"Target output directory locked: {fixed_output_dir}")

    cmd = [sys.executable, "-u", "evaluate.py"] + filtered_args + [f"hydra.run.dir={fixed_output_dir}"]
    
    restart_count = 0
    
    while True:
        log_event(watchdog_log_path, f"Starting evaluation process... (Restart Count: {restart_count})")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )

        last_output_time = [time.time()]
        
        monitor_thread = threading.Thread(target=monitor_output, args=(process, last_output_time))
        monitor_thread.daemon = True
        monitor_thread.start()

        while process.poll() is None:
            time.sleep(1)
            idle_time = time.time() - last_output_time[0]
            
            if idle_time > TIMEOUT_SECONDS:
                log_event(watchdog_log_path, f"Freeze detected! No output for {idle_time:.1f} seconds.")
                log_event(watchdog_log_path, "Killing the hanging process...")
                process.kill()
                restart_count += 1
                time.sleep(2) 
                break 

        if process.poll() == 0:
            log_event(watchdog_log_path, "Evaluation completed successfully!")
            
            temp_file = os.path.join(fixed_output_dir, "temp_checkpoint.json")
            if os.path.exists(temp_file):
                os.remove(temp_file)
                log_event(watchdog_log_path, f"Cleaned up temporary checkpoint file: {temp_file}")
            break
        elif process.poll() is not None:
            exit_code = process.poll()
            log_event(watchdog_log_path, f"Process crashed with return code {exit_code}.")
            log_event(watchdog_log_path, "Restarting immediately...")
            restart_count += 1
            time.sleep(2)

if __name__ == "__main__":
    main()