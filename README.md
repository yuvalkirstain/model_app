# Image Generation Backend

## Create TPUs that run the code

To create a single TPU:
```bash
bash request_vm.sh <idx> <type>
```
for example:
```bash
bash request_vm.sh 2 v3-8
```
Will create a TPU VM with name tpu-2 of type v3-8.

## Set up the VM
To set up a single VM ssh into the VM:
```bash
gcloud compute tpus tpu-vm ssh <tpu-name> --zone <zone>
```
install dependencies and start tmux session:
```bash
git clone https://github.com/yuvalkirstain/model_app.git
cd model_app
bash startup.sh
```
Run the backend:
```bash
python3 -m gunicorn -t 300 -b :5000 -w 1 -k uvicorn.workers.UvicornWorker api:app
```


## Add address to the demo.
Login to Heroku and add the external IP address of the VM to the BACKEND_URLS env var and deploy.

