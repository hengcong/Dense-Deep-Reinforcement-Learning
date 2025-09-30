import argparse
import numpy as np
import traceback  # ✅ 用于输出详细的错误信息
from conf import conf
from carlaenv import CarlaEnv

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='NDE',
                    help='simulation running mode: TM / NDE / D2RL / behavior_policy')
parser.add_argument('--experiment_name', type=str, default='debug',
                    help='specify experiment name')
parser.add_argument('--worker_id', type=int, default=0,
                    help='worker id (for multiprocess if needed)')
args = parser.parse_args()

# === Apply experiment config ===
conf.experiment_config["mode"] = args.mode
conf.experiment_config["experiment_name"] = args.experiment_name
if args.mode == "NDE":
    conf.simulation_config["epsilon_setting"] = "fixed"
elif args.mode == "TM":
    conf.simulation_config["epsilon_setting"] = "tm"
elif args.mode == "D2RL":
    conf.simulation_config["epsilon_setting"] = "drl"
elif args.mode == "behavior_policy":
    conf.simulation_config["epsilon_setting"] = "fixed"
else:
    raise ValueError(f"Unknown mode: {args.mode}")

# === Create Carla environment ===
env = CarlaEnv(num_veh=500, num_ped=10)
env.worker_id = args.worker_id

# === (Optional) Load D2RL agent ===
if args.mode == "D2RL":
    try:
        conf.discriminator_agent = conf.load_discriminator_agent(
            checkpoint_path="./checkpoints/2lane_400m/model.pt")
        print("✅ Loaded D2RL discriminator agent.")
    except Exception as e:
        print(f"❌ Failed to load D2RL agent: {e}")
        traceback.print_exc()
        exit(1)

# === Run episodes ===
EPISODE_NUM = 500

try:
    for ep in range(EPISODE_NUM):
        print(f"\n🚦 Starting episode {ep} | Mode: {args.mode}")
        try:
            env.reset()
            print(f"📍 Start location: {env.start_location} | ⏱ Start time: {env.start_time:.2f}s")
        except Exception as e:
            print(f"❌ Reset failed in episode {ep}: {e}")
            traceback.print_exc()
            break

        while True:
            try:
                env.step()
            except Exception as e:
                print(f"❌ [step()] failed in episode {ep}: {e}")
                traceback.print_exc()
                break

            try:
                done, reason, _ = env.check_done()
            except Exception as e:
                print(f"❌ [check_done()] failed in episode {ep}: {e}")
                traceback.print_exc()
                break

            if done:
                print(f"💥 Episode {ep} ended: {reason}")
                print(f"📏 Distance traveled: {env.distance_travelled:.2f} m")
                print(f"⏱ Duration: {env.get_simulation_time() - env.start_time:.2f} s")
                # env.log_episode()
                break

except KeyboardInterrupt:
    print("\n🚪 Interrupted by user.")

finally:
    print("✅ Simulation finished.")
