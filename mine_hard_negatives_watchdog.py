import os
import time
import signal
import subprocess
from datetime import datetime # 시간 모듈 추가

def run_universal_watchdog(target_script="mine_hard_negatives.py", timeout_sec=120): # 120초로 세팅
    print(f"👀 [Watchdog] 범용 출력 감시 및 고아 프로세스 박멸 모드로 가동합니다.")
    log_file = "pipeline_heartbeat.log"
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n🚀 [Watchdog] 프로세스 가동 (진행 상황: '{log_file}')")
        
        # 재실행 시 로그 파일에 명확한 시간 구분선 삽입 (Append 모드)
        with open(log_file, "a") as f:
            f.write(f"\n\n================ [RESTART at {current_time}] ================\n\n")
            
        # 서브프로세스 실행 (동일하게 Append 모드 적용)
        with open(log_file, "a") as f:
            process = subprocess.Popen(
                ["python", target_script],
                stdout=f,
                stderr=f,
                env=env,
                preexec_fn=os.setsid 
            )
        
        time.sleep(5)
        
        while process.poll() is None:
            time.sleep(10)
            
            if os.path.exists(log_file):
                last_modified = os.path.getmtime(log_file)
                idle_time = time.time() - last_modified
                
                if idle_time > timeout_sec:
                    print(f"\n🚨 [Watchdog] {timeout_sec}초간 응답 없음 (데드락 감지).")
                    print(f"🔪 프로세스 그룹(PGID: {os.getpgid(process.pid)}) 전체를 몰살합니다.")
                    
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                    
                    print("🔄 3초 후 안전하게 이어달리기를 재개합니다.")
                    time.sleep(3)
                    break 
                    
        if process.poll() == 0:
            print("\n🎉 [Watchdog] 파이프라인 100% 완료. 감시 종료.")
            break
        elif process.poll() is not None and process.poll() != 0:
            print(f"\n⚠️ 프로세스 비정상 종료 (에러 코드: {process.poll()}). 즉시 재실행.")
            time.sleep(3)

if __name__ == "__main__":
    run_universal_watchdog()