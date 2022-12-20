# Install packages
echo "Installing packages"
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install fastapi[all] uvicorn[standard] gunicorn flax transformers ftfy diffusers

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
tmux new-session -d -s model_app 'python3 -m gunicorn -t 300 -b :5000 -w 1 -k uvicorn.workers.UvicornWorker api:app'
tmux ls