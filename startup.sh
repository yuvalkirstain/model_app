# Install packages
echo "Installing packages"
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install fastapi[all] uvicorn[standard] gunicorn flax transformers ftfy diffusers[flask]

# Get code
echo "Getting code"
if [ ! -d "model_app" ]; then
  git clone https://github.com/yuvalkirstain/model_app.git
fi

cd model_app
git pull

# Start service
echo "Starting service"
fuser -k 5000/tcp
tmux kill-server

export TPU_CHIPS_PER_HOST_BOUNDS="1,2,1"
export TPU_HOST_BOUNDS="1,1,1"

export TPU_VISIBLE_DEVICES="0,1"
export TPU_MESH_CONTROLLER_ADDRESS="localhost:8476"
export TPU_MESH_CONTROLLER_PORT="8476"
tmux new-session -d -s model_app_start 'python3 -m gunicorn -t 300 -b :5000 -w 1 -k uvicorn.workers.UvicornWorker api:app'

export TPU_VISIBLE_DEVICES="2,3"
export TPU_MESH_CONTROLLER_ADDRESS="localhost:8477"
export TPU_MESH_CONTROLLER_PORT="8477"
tmux new-session -d -s model_app_end 'python3 -m gunicorn -t 300 -b :5001 -w 1 -k uvicorn.workers.UvicornWorker api:app'

tmux ls