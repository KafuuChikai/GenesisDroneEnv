import torch
from typing import TYPE_CHECKING
from genesis.utils.geom import quat_to_R

if TYPE_CHECKING:
    from genesis.engine.entities.drone_entity import DroneEntity

class DroneController():
    def __init__(self):
        self.err_v_integral = None
        self.max_integral = 2.0 
        self.kp = 8.0 
        self.kv = 4.0
        self.ki = 0.0
        self.kw = 8.0
        self.z_vec = torch.tensor([0.0, 0.0, 1.0])
        self.g = 9.81

    def vee(self, mat):
        return torch.stack([mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]], dim=-1)

    def run(self, pos, vel, quat, pos_d, vel_d, yaw_d):
        device = pos.device
        if not isinstance(pos, torch.Tensor): pos = torch.tensor(pos, device=device)
        if not isinstance(vel, torch.Tensor): vel = torch.tensor(vel, device=device)
        if not isinstance(quat, torch.Tensor): quat = torch.tensor(quat, device=device)
        
        # Ensure target vectors match batch shape if necessary
        if pos.ndim > 1 and pos_d.ndim == 1:
            pos_d = pos_d.expand_as(pos)
        if vel.ndim > 1 and vel_d.ndim == 1:
            vel_d = vel_d.expand_as(vel)
        if pos.ndim > 1 and yaw_d.ndim == 0:
            yaw_d = yaw_d.expand(pos.shape[0])

        err_p = pos - pos_d
        err_v = vel - vel_d
        
        # Initialize integral term with correct shape
        if self.err_v_integral is None or self.err_v_integral.shape != err_v.shape:
            self.err_v_integral = torch.zeros_like(err_v)
            
        self.err_v_integral += err_v
        self.err_v_integral = torch.clamp(self.err_v_integral, -self.max_integral, self.max_integral)

        # Broadcasting z_vec
        z_vec = self.z_vec.to(device)
        
        target_acc = -self.kp * err_p - self.kv * err_v + self.g * z_vec - self.ki * self.err_v_integral

        z_d = target_acc / torch.linalg.norm(target_acc, dim=-1, keepdim=True)
        
        x_d_temp = torch.stack([torch.cos(yaw_d), torch.sin(yaw_d), torch.zeros_like(yaw_d)], dim=-1)
        
        y_d = torch.cross(z_d, x_d_temp, dim=-1)
        y_d = y_d / torch.linalg.norm(y_d, dim=-1, keepdim=True)
        
        x_d = torch.cross(y_d, z_d, dim=-1)
        
        R_d = torch.stack([x_d, y_d, z_d], dim=-1)

        R = quat_to_R(quat)
        
        # Matrix multiplication support for (3,3) and (B,3,3)
        R_d_T = R_d.transpose(-1, -2)
        R_T = R.transpose(-1, -2)
        
        err_R_mat = 0.5 * (R_d_T @ R - R_T @ R_d)
        err_R_vec = self.vee(err_R_mat)

        w_d = -self.kw * err_R_vec

        # Z-body projection
        # If batch: R is (B,3,3), z_vec is (3). Result (B,3) via broadcasting
        z_body = R @ z_vec 
        
        # Dot product: (3)@(3)->scalar OR (B,3)*(B,3)->sum(-1)->(B)
        u1 = (target_acc * z_body).sum(dim=-1)

        return torch.cat([u1.unsqueeze(-1), w_d], dim=-1)

class DroneLLC:
    def __init__(self):
        self.mass = 0.027
        self.arm = 0.0397
        
        self.kf = 3.16e-10
        self.km = 7.94e-12 
        
        self.intertia = torch.diag(torch.tensor([1.4e-5, 1.4e-5, 2.17e-5]))
        
        self.max_rpm = 25000.0
        self.min_rpm = 0.0

        d = self.arm / (2 ** 0.5) 
        kf = self.kf
        km = self.km
        

        # 3(1)  x   0(-1)  <- Front
        #    \  ^  /
        #     \ | /
        #      \|/
        # y<----|
        #      / \
        #     /   \
        #    /     \
        # 2(-1)     1(1)   <- Back

        self.allocation_matrix = torch.tensor([
            [kf,      kf,      kf,      kf],      # Thrust
            [-d*kf,   -d*kf,   d*kf,    d*kf],    # Roll
            [-d*kf,    d*kf,   d*kf,   -d*kf],    # Pitch
            [-km,      km,     -km,      km]      # Yaw
        ])

        try:
            self.allocation_matrix_inv = torch.linalg.pinv(self.allocation_matrix)
        except:
            self.allocation_matrix_inv = torch.inverse(self.allocation_matrix)
            
        self._kp = torch.tensor([20.0, 20.0, 40.0])

    def run(self, ct_sp, ang_vel, ang_vel_sp):
        device = ang_vel.device
        if not isinstance(ang_vel, torch.Tensor): ang_vel = torch.tensor(ang_vel, device=device)
        if not isinstance(ang_vel_sp, torch.Tensor): ang_vel_sp = torch.tensor(ang_vel_sp, device=device)
        
        if self.intertia.device != device:
            self.intertia = self.intertia.to(device)
            self.allocation_matrix_inv = self.allocation_matrix_inv.to(device)
            self._kp = self._kp.to(device)

        total_thrust = ct_sp * self.mass

        total_thrust = torch.clamp(total_thrust, min=0.0)

        torque_cmd = self._angular_controller(ang_vel, ang_vel_sp)

        prop_rpm = self._mixer(total_thrust, torque_cmd)
        return prop_rpm

    def _angular_controller(self, ang_vel, ang_vel_sp):
        err_w = ang_vel_sp - ang_vel
        
        # Robust matrix multiplication for both (3) and (B,3)
        # J @ err_w -> (..., 3)
        # Using unsqueeze to treat vectors as (..., 3, 1) to support batch matmul
        J_err = (self.intertia @ (self._kp * err_w).unsqueeze(-1)).squeeze(-1)
        
        J_w = (self.intertia @ ang_vel.unsqueeze(-1)).squeeze(-1)
        gyroscopic = torch.cross(ang_vel, J_w, dim=-1)
        
        torque_cmd = J_err + gyroscopic
        return torque_cmd
    
    def _mixer(self, thrust, torque):
        if thrust.ndim == 0:
            thrust = thrust.unsqueeze(0)
        if thrust.ndim == 1 and torque.ndim == 2:
            thrust = thrust.unsqueeze(-1)

        # desired_wrench: (..., 4)
        desired_wrench = torch.cat([thrust, torque], dim=-1)
        
        # alloc_inv: (4,4) @ wrench: (..., 4, 1) -> (..., 4)
        # Or wrench @ alloc_inv.T
        prop_rpm_squared = (self.allocation_matrix_inv @ desired_wrench.unsqueeze(-1)).squeeze(-1)
        
        prop_rpm_squared = torch.clamp(prop_rpm_squared, min=0.0)
        prop_rpm = torch.sqrt(prop_rpm_squared)
        
        prop_rpm = torch.clamp(prop_rpm, self.min_rpm, self.max_rpm)
        return prop_rpm

if __name__ == "__main__":
    import genesis as gs
    gs.init()
    n_envs = 5
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())
    drone : "DroneEntity" = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0, 0.5),
        ),
    )
    scene.build(n_envs=n_envs)
    pos_controller = DroneController()
    llc_controller = DroneLLC()
    for i in range(1000):
        t = torch.tensor(i * scene.sim.dt)
        if n_envs > 0:
            pos_d = torch.zeros(n_envs, 3,)
            pos_d[:, 2] = 0.5
            vel_d = torch.zeros(n_envs, 3,)
            yaw_d = torch.zeros(n_envs,) + t * torch.pi
            for j in range(n_envs):
                pos_d[j, 0] =  j*0.2
        else:
            pos_d = torch.tensor([0.0, 0.0, 0.5],)
            vel_d = torch.tensor([0.0, 0.0, 0.0],)
            yaw_d = torch.tensor(0.0,)

        pos = drone.get_pos()
        vel = drone.get_vel()
        quat = drone.get_quat()
        
        ctbr_sp = pos_controller.run(pos, vel, quat, pos_d, vel_d, yaw_d)
        
        R = quat_to_R(quat) # (3,3) or (B,3,3)
        ang_vel_world = drone.get_ang() # (3,) or (B,3)
        ang_vel_body = (R.transpose(-1, -2) @ ang_vel_world.unsqueeze(-1)).squeeze(-1)
        prop_rpm = llc_controller.run(
            ct_sp=ctbr_sp[..., 0],
            ang_vel=ang_vel_body,
            ang_vel_sp=ctbr_sp[..., 1:4],
        )
        
        drone.set_propellels_rpm(prop_rpm)
        scene.step()
