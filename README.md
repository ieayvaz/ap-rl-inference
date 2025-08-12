# ap-rl-inference
Inference code for a trained model to control an Ardupilot driven uav. Models are trained in flyrl simulation.

# How it works
Model takes 36d state vector and output reference roll, pitch and throttle values. Ardupilot is set to FBWA mode. Reference values are sent to Ardupilot via RC channel override. FBWA mode takes in reference values and try to 
hold reference value with it's own inner PID. 
# Handling inner PID coordination
The model is trained with a custom PID(similar coefficients). During training model observes PID inner working history as this is curicial for convergence and robust control.
In inference, inner workings of Ardupilot's PID estimated via a ghost PID, that works similar to Ardupilot's PID but does not output any action.
