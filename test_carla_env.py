import time
from carlaenv import CarlaEnv

def test_env():
    print("🧪 [TEST] Initialize CarlaEnv")
    env = CarlaEnv(num_veh=100, num_ped=5)
    time.sleep(1.0)

    num_episodes = 15
    for ep in range(num_episodes):
        print(f"\n🚦 Starting Episode {ep + 1}")
        env.reset()
        time.sleep(0.5)

        max_frames = 50
        for frame in range(max_frames):
            print(f"\n🌀 Frame {frame + 1}/{max_frames}")

            try:
                env.step()
            except Exception as e:
                print(f"❌ Error in step(): {e}")
                break

            time.sleep(0.03)

            try:
                env.render_image()
            except Exception as e:
                print(f"❌ Render error: {e}")

            try:
                obs = env.get_state()
                ego_obs = obs["Ego"]
                print("📷 Ego observation:", ego_obs)

                ego_wp = env.get_waypoint(env.ego_vehicle)
                print(f"📍 Ego s: {ego_wp.s:.2f}")

                for role in ["Lead", "Foll", "LeftLead", "LeftFoll", "RightLead", "RightFoll"]:
                    veh_obs = obs.get(role)
                    print(f"🚗 {role} observation:", veh_obs)
                    if veh_obs:
                        try:
                            wp = env.get_waypoint(env.get_vehicle_by_id(veh_obs["veh_id"]))
                            print(f"📍 {role} s: {wp.s:.2f}")
                        except:
                            print(f"⚠️ Failed to get {role} waypoint")

                        dx = veh_obs["position"][0] - ego_obs["position"][0]
                        if "Foll" in role:
                            dx = -dx
                        print(f"📏 Δx to {role}: {dx:.2f} m | Ego ↔ {role} ID: {veh_obs['veh_id']}")
                        if dx < 0 and "Lead" in role:
                            print(f"⚠️ {role} vehicle is BEHIND ego? Check logic.")
                        if dx < 0 and "Foll" in role:
                            print(f"⚠️ {role} vehicle is AHEAD of ego? Check logic.")
            except Exception as e:
                print(f"❌ Observation error: {e}")

            try:
                dist, ttc = env.get_av_ttc()
                print(f"🕒 TTC distance: {dist:.2f}, TTC: {ttc:.2f}")
            except Exception as e:
                print(f"❌ TTC computation error: {e}")

            if env.check_collision():
                print("💥 Collision detected!")
                break

            done, reason, _ = env.check_done()
            if done:
                print(f"🛑 Episode ended early. Reason: {reason}")
                break

        try:
            print(f"\n✅ Final ego position: {env.ego_vehicle.get_location()}")
        except:
            print("❓ Ego vehicle might have been destroyed")

        print(f"✅ Number of background vehicles: {len(env.vehicles)}")
        print(f"✅ Number of pedestrians: {len(env.pedestrians)}")
        print(f"✅ Total steps recorded: {len(env.episode_data)}")

        print("📝 Logging episode...")
        env.log_episode()
        print(f"📦 Total episodes logged: {len(env.episode_logs)}")

    print("\n✅ All episodes complete.")

if __name__ == "__main__":
    test_env()
