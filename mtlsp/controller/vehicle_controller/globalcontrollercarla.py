from abc import ABC, abstractmethod
import carla

class GlobalController(ABC):
    def __init__(self, env, veh_type):
        self.env = env
        self.veh_type = veh_type
        self.control_log = {}

    @property
    def controllable_veh_id_list(self):
        return self._get_controllable_veh_id_list()

    def _get_controllable_veh_id_list(self):
        controllable_veh_id_list = []
        for veh_id, wrapper in self.env.vehicle_wrapper_list.items():
            if wrapper.role == self.veh_type:
                controllable_veh_id_list.append(veh_id)
        return controllable_veh_id_list

    @abstractmethod
    def step(self):
        pass

    def apply_control_permission(self):
        return True

class DummyGlobalController(GlobalController):
    def reset_control_and_action_state(self):
        if self.veh_type == "CAV":
            vehicle = self.env.ego_vehicle_wrapper
            #print(f"[DummyGlobalController] Reset control for ego vehicle ID: {vehicle.vehicle.id}")
            vehicle.reset_control_state()

        elif self.veh_type == "BV":
            for veh_id in self.controllable_veh_id_list:
                vehicle = self.env.vehicle_wrapper_list[veh_id]
                vehicle.reset_control_state()

    def step(self):
        if self.apply_control_permission():
            self.reset_control_and_action_state()

            if self.veh_type == "CAV":
                vehicle = self.env.ego_vehicle_wrapper
                #print(f"[DummyGlobalController] Stepping ego controller: {type(vehicle.controller)}")
                vehicle.controller.step()
                vehicle.update(self.env)

                cmds = []
                if vehicle.cached_transform is not None:
                    cmds.append(carla.command.ApplyTransform(vehicle.vehicle, vehicle.cached_transform))
                if vehicle.cached_velocity is not None:
                    cmds.append(carla.command.ApplyTargetVelocity(vehicle.vehicle, vehicle.cached_velocity))
                self.env.client.apply_batch(cmds)


            elif self.veh_type == "BV":
                for veh_id in self.controllable_veh_id_list:
                    vehicle = self.env.vehicle_wrapper_list[veh_id]
                    vehicle.controller.step()
                    vehicle.update()
class PedestrianGlobalController(GlobalController):
    """Global controller for pedestrians (wrappers stored in env.pedestrian_wrapper_list).
       Assumes each wrapper.update(env, dt) performs manual integration and calls set_transform().
    """

    # override to use pedestrian wrapper list instead of vehicle list
    def _get_controllable_veh_id_list(self):
        ped_ids = []
        ped_map = getattr(self.env, "pedestrian_wrapper_list", {})
        for ped_id, wrapper in ped_map.items():
            if wrapper.role == self.veh_type:  # e.g., "PED" (set wrapper.role accordingly)
                ped_ids.append(ped_id)
        return ped_ids

    def reset_control_and_action_state(self):
        ped_map = getattr(self.env, "pedestrian_wrapper_list", {})
        for ped_id in self.controllable_veh_id_list:
            w = ped_map.get(ped_id)
            if w:
                w.reset_control_state()

    def step(self):
        if not self.apply_control_permission():
            return

        self.reset_control_and_action_state()

        ped_map = getattr(self.env, "pedestrian_wrapper_list", {})
        dt = getattr(self.env, "frame_dt", getattr(self.env, "step_dt", 0.05))

        for ped_id in self.controllable_veh_id_list:
            w = ped_map.get(ped_id)
            if not w or not getattr(w, "controller", None) or not hasattr(w.controller, "step"):
                continue

            # 1) make controller produce action (WalkerControl stored at w.controller.action)
            w.controller.step()

            # 2) manual integrate and push pose via set_transform()
            w.update(self.env, dt=dt)