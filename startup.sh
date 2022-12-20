# Install packages
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install fastapi[all] uvicorn[standard] gunicorn flax transformers ftfy diffusers

# Get code
git clone https://github.com/yuvalkirstain/model_app.git
cd model_app

# Start service
tmux new-session -d -s model_app 'python3 -m gunicorn -t 300 -b :5000 -w 1 -k uvicorn.workers.UvicornWorker api:app'
