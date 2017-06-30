from a3c_master import A3cMaster
from environment import Environment

def main():
    env = Environment("Breakout-v0")
    a3c_master = A3cMaster(gpu_id=0, environment=env, worker_count=2, max_elapsed_time=1e-7)
    a3c_master.start_training()

if __name__ == "__main__":
    main()