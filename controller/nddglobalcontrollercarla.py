from mtlsp.controller.vehicle_controller.globalcontrollercarla import DummyGlobalController
from .nddcontrollercarla import NDDController
from mtlsp.controller.vehicle_controller.controller import Controller

class NDDBVGlobalController(DummyGlobalController):
    def __init__(self, env, veh_type="BV"):
        super().__init__(env, veh_type)
        self.control_vehicle_set = set()

    def step(self):
        if self.apply_control_permission():
            self.reset_control_and_action_state()
            self.update_controlled_vehicles(controller=NDDController)
            for veh_id in self.controllable_veh_id_list:
                vehicle = self.env.vehicle_list[veh_id]
                vehicle.controller.step()
                vehicle.update()

        else:
            self.control_vehicle_set = set()

    def update_controlled_vehicles(self, controller=NDDController):

        CAV = self.env.ego_vehicle_wrapper

        context_vehicle_set = set(CAV.observation.context.keys())

        if self.control_vehicle_set != context_vehicle_set:
            for veh_id in context_vehicle_set - self.control_vehicle_set:
                self.env.get_surrounding_vehicles(self.env.vehicle_wrapper_list[veh_id].vehicle)
                ctrl = controller(env=self.env)
                ctrl.attach_to_vehicle(self.env.vehicle_wrapper_list[veh_id])
                self.env.vehicle_wrapper_list[veh_id].install_controller(ctrl)

            for veh_id in self.control_vehicle_set - context_vehicle_set:
                if veh_id in self.env.vehicle_wrapper_list:
                    self.env.vehicle_wrapper_list[veh_id].install_controller(None)

            self.control_vehicle_set = context_vehicle_set