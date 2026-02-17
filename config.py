DDIM_config = {
  "_class_name": "DDIMScheduler", # Switched to DDIM for more stable log-prob paths
  "_diffusers_version": "0.24.0.dev0",
  
  # 1. Lower the end-of-process "crush"
  "beta_end": 0.01,           
  
  # 2. Use a smoother schedule to avoid the Step 24 explosion
  "beta_schedule": "scaled_linear", 
  
  # 3. Lower start to allow finer atomic resolution at t=0
  "beta_start": 0.0001,       
  
  "clip_sample": False,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  
  # 4. Crucial for SDPO: ensure alpha starts at 1.0 for the final step
  "set_alpha_to_one": True,    
  
  "steps_offset": 0,
  "timestep_spacing": "leading", # "leading" often aligns better with RL trajectory indexing
  "timestep_type": "discrete"
}