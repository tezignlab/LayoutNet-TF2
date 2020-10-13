# configuration for building the network
y_dim = 6
tr_dim = 7
ir_dim = 10
latent_dim = 128
z_dim = 128
batch_size = 128
lr = 0.0002
beta1 = 0.5

# configuration for the supervisor
logdir = "./log"
sampledir = "./example"
max_steps = 30000
sample_every_n_steps = 100
summary_every_n_steps = 1
save_model_secs = 120
checkpoint_basename = "layout"
checkpoint_dir = "./checkpoints"
filenamequeue = "./dataset/layout_1205.tfrecords"
min_after_dequeue = 5000
num_threads = 4
