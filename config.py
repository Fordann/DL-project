CONFIG = {
    'img_size': 64,    
    'batch_size': 32, 
    'epochs': 60,  
    'T': 1000,  # number of steps from image x0 -> to complete noise
    'beta_start': 1e-4, # start noise 
    'beta_end':  0.02, # final noise   
    'best_model_name': 'diffusion_model_complete.keras' 
}