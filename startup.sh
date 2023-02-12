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
git stash
git pull

# Start service
echo "Starting service"
fuser -k 5000/tcp
fuser -k 5001/tcp
fuser -k 5002/tcp
fuser -k 5003/tcp
tmux kill-server


tmux new-session -d -s model_app1 'export MODEL_ID="yuvalkirstain/dreamlike-photoreal-2-flax"; export TPU_HOST_BOUNDS="1,1,1"; export TPU_CHIPS_PER_HOST_BOUNDS="1,2,1"; export TPU_VISIBLE_DEVICES="0"; export TPU_MESH_CONTROLLER_ADDRESS="localhost:8476"; export TPU_MESH_CONTROLLER_PORT="8476"; python3 -m gunicorn -t 300 -b :5000 -w 1 -k uvicorn.workers.UvicornWorker api:app'
tmux new-session -d -s model_app2 'export MODEL_ID="yuvalkirstain/stable-diffusion-2-1"; export TPU_HOST_BOUNDS="1,1,1"; export TPU_CHIPS_PER_HOST_BOUNDS="1,2,1"; export TPU_VISIBLE_DEVICES="1"; export TPU_MESH_CONTROLLER_ADDRESS="localhost:8477"; export TPU_MESH_CONTROLLER_PORT="8477"; python3 -m gunicorn -t 300 -b :5001 -w 1 -k uvicorn.workers.UvicornWorker api:app'

sleep 30

tmux new-session -d -s model_app3 'export MODEL_ID="stabilityai/dreamlike-photoreal-2-flax"; export TPU_HOST_BOUNDS="1,1,1"; export TPU_CHIPS_PER_HOST_BOUNDS="1,2,1"; export TPU_VISIBLE_DEVICES="2"; export TPU_MESH_CONTROLLER_ADDRESS="localhost:8478"; export TPU_MESH_CONTROLLER_PORT="8478"; python3 -m gunicorn -t 300 -b :5002 -w 1 -k uvicorn.workers.UvicornWorker api:app'
tmux new-session -d -s model_app4 'export MODEL_ID="stabilityai/stable-diffusion-2-1"; export TPU_HOST_BOUNDS="1,1,1"; export TPU_CHIPS_PER_HOST_BOUNDS="1,2,1"; export TPU_VISIBLE_DEVICES="3"; export TPU_MESH_CONTROLLER_ADDRESS="localhost:8479"; export TPU_MESH_CONTROLLER_PORT="8479"; python3 -m gunicorn -t 300 -b :5003 -w 1 -k uvicorn.workers.UvicornWorker api:app'

tmux ls