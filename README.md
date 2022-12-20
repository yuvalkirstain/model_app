# Image Generation Backend

## Installation
Create a virtual environment and install the requirements from the `requirements.txt` file (not all are necessary).

## Single CMD
After you have installed the dependencies, you can run:
```bash
python cli.py fill-quota
```

At any time, you can stop, and run:
```bash
python cli.py prepare-backend-urls-env-vars-str
```
to get the backend urls env var before you update it in heroku.

## Step-by-Step
### Create TPUs that run the code

To create a single TPU:
```bash
bash request_vm.sh <idx> <type>
```
for example:
```bash
bash request_vm.sh 2 v3-8
```
Will create a TPU VM with name tpu-2 of type v3-8.

### Set up the VM
To set up a single VM ssh into the VM:
```bash
gcloud compute tpus tpu-vm ssh <tpu-name> --zone <zone>
```
install dependencies and start tmux session with the app:
```bash
git clone https://github.com/yuvalkirstain/model_app.git
cd model_app
bash startup.sh
```

### Add address to the demo.
Login to Heroku and add the external IP address of the VM to the BACKEND_URLS env var and deploy.

# To run with GPU
Bunch of stuff need to be one. One of them is start a ray cluster and serve with it.
```bash
ray start --head
serve run model:model --port 8503
```

 