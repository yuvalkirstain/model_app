import json
import os
import subprocess
import typer
from tqdm import tqdm

app = typer.Typer()

ZONES = ["europe-west4-a", "us-central1-f"]
ZONE2TPU_TYPE = {
    "europe-west4-a": "v3-8",
    "us-central1-f": "v2-8",
}
ZONE2QUOTA = {
    "europe-west4-a": 5,
    "us-central1-f": 5,
}

STARTUP_SCRIPT_PATH = "startup.sh"


def os_call(cmd: str) -> str:
    return subprocess.check_output(cmd.split()).rstrip().decode()


def get_tpu_names(zone: str):
    tpus = json.loads(os_call(f"gcloud compute tpus list --zone {zone} --format json"))
    tpu_names = [tpu['name'].split('/')[-1] for tpu in tpus]
    return tpu_names


def get_valid_tpu_names(zone: str):
    tpus = json.loads(os_call(f"gcloud compute tpus list --zone {zone} --format json"))
    tpu_names = [tpu['name'].split('/')[-1] for tpu in tpus if tpu['state'] == 'READY']
    return tpu_names


def describe_resources_of_zone(zone: str):
    cmd = f"gcloud compute tpus list --zone {zone}"
    print(f"Running {cmd}")
    os.system(cmd)
    print(f"Done {cmd}")


@app.command()
def describe_resources():
    for zone in ZONES:
        describe_resources_of_zone(zone)


@app.command()
def print_all_tpu_names():
    for zone in ZONES:
        tpu_names = get_tpu_names(zone)
        for tpu_name in tpu_names:
            print(tpu_name)


@app.command()
def print_quota():
    for zone in ZONES:
        allocated = len(get_tpu_names(zone))
        quota = ZONE2QUOTA[zone]
        print(f"{zone}: {allocated}/{quota}")


@app.command()
def fill_quota_for_zone(zone: str):
    tpu_type = ZONE2TPU_TYPE[zone]
    quota = ZONE2QUOTA[zone]
    for i in range(1, quota + 1):
        tpu_names = get_tpu_names(zone)
        num_allocated = len(tpu_names)
        tpu_name = f"tpu-{i}"
        if num_allocated >= quota:
            print(f"Zone {zone} is full")
            break
        if tpu_name in tpu_names:
            print(f"TPU {tpu_name} in zone {zone} already exists")
            continue
        cmd = f"CMD='gcloud compute tpus tpu-vm create {tpu_name} --zone {zone} --accelerator-type {tpu_type} --version tpu-vm-base'; until $CMD; do sleep 3; done"
        print(f"Running {cmd}")
        os.system(cmd)
        print(f"Done {cmd}")
        print(f"Created TPU {tpu_name} in zone {zone} with type {tpu_type}")
        run_app(tpu_name, zone)


@app.command()
def fill_quota():
    for zone in ZONES:
        fill_quota_for_zone(zone)
    prepare_backend_urls_env_vars_str()


def send_file_to_tpu(tpu_name: str, zone: str, local_file_path: str, remote_file_path: str, username="yuvalkirstain"):
    assert os.path.exists(local_file_path), f"Local file {local_file_path} does not exist"
    cmd = f"gcloud compute tpus tpu-vm scp {local_file_path} {username}@{tpu_name}:{remote_file_path} --zone {zone}"
    print(f"Running {cmd}")
    os.system(cmd)
    print(f"Done {cmd}")


def exec_on_tpu(tpu_name: str, zone: str, command: str):
    cmd = f"gcloud compute tpus tpu-vm ssh {tpu_name} --zone {zone} --command '{command}'"
    print(f"Running {cmd}")
    os.system(cmd)
    print(f"Done {cmd}")


@app.command()
def run_app(tpu_name: str, zone: str, username="yuvalkirstain"):
    send_file_to_tpu(
        tpu_name=tpu_name,
        zone=zone,
        local_file_path=STARTUP_SCRIPT_PATH,
        remote_file_path=STARTUP_SCRIPT_PATH,
        username=username
    )
    exec_on_tpu(
        tpu_name=tpu_name,
        zone=zone,
        command=f"bash {STARTUP_SCRIPT_PATH}",
    )


@app.command()
def run_app_on_all_machines():
    for zone in ZONES:
        print(f"Zone {zone}:")
        tpu_names = get_valid_tpu_names(zone)
        for tpu_name in tpu_names:
            print(f"Running on {tpu_name}")
            run_app(tpu_name, zone)
            print(f"Done running on {tpu_name}")


def get_external_ip_of_tpu(tpu_name: str, zone: str):
    cmd = f"gcloud compute tpus tpu-vm describe {tpu_name} --zone {zone} --format json"
    tpu_info = json.loads(os_call(cmd))
    return tpu_info["networkEndpoints"][0]["accessConfig"]["externalIp"]


@app.command()
def get_external_ips():
    for zone in ZONES:
        print(f"Zone {zone}:")
        tpu_names = get_valid_tpu_names(zone)
        for tpu_name in tpu_names:
            external_ip = get_external_ip_of_tpu(tpu_name, zone)
            print(f"{tpu_name}: {external_ip}")


@app.command()
def prepare_backend_urls_env_vars_str():
    env_vars_str = []
    for zone in ZONES:
        tpu_names = get_valid_tpu_names(zone)
        for tpu_name in tqdm(tpu_names):
            external_ip = get_external_ip_of_tpu(tpu_name, zone)
            env_vars_str.append(f"http://{external_ip}:5000/generate")
    print(f"BACKEND_URLS={json.dumps(env_vars_str)}")


if __name__ == '__main__':
    app()
