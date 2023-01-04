import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict

import requests
import typer
from tqdm import tqdm

app = typer.Typer()


@dataclass
class Resource:
    tpu_type: str
    quota: int
    zone: str
    is_preemptible: bool

    def __repr__(self):
        return f"{self.tpu_type} in {self.zone} (preemptible: {self.is_preemptible})"


RESOURCES = [
    Resource(tpu_type="v2-8", quota=5, zone="us-central1-f", is_preemptible=False),
    Resource(tpu_type="v3-8", quota=5, zone="europe-west4-a", is_preemptible=False),
    # Resource(tpu_type="v2-8", quota=30, zone="us-central1-f", is_preemptible=True),
]

STARTUP_SCRIPT_PATH = "startup.sh"


def os_call(cmd: str) -> str:
    return subprocess.check_output(cmd.split()).rstrip().decode()


def is_preemptible(tpu_json: Dict):
    return tpu_json["schedulingConfig"].get("preemptible", False)


def get_tpu_names_by_zone(zone: str):
    tpus = json.loads(os_call(f"gcloud compute tpus list --zone {zone} --format json"))
    tpu_names = [tpu['name'].split('/')[-1] for tpu in tpus]
    return tpu_names


def get_tpu_names(resource: Resource):
    zone = resource.zone
    tpu_names = get_tpu_names_by_zone(zone)
    return tpu_names


def get_valid_tpu_names(zone: str):
    tpus = json.loads(os_call(f"gcloud compute tpus list --zone {zone} --format json"))
    working_tpus = []
    tpu_names = [tpu['name'].split('/')[-1] for tpu in tpus if tpu['state'] == 'READY']
    for tpu_name in tpu_names:
        external_ip = get_external_ip_of_tpu(tpu_name, zone)
        try:
            response = requests.get(f"http://{external_ip}:5000/")
            if response.status_code == 200:
                working_tpus.append(tpu_name)
        except requests.exceptions.ConnectionError:
            print(f"Connection error for {tpu_name}")
    return working_tpus


def get_invalid_tpu_names(zone: str):
    tpus = json.loads(os_call(f"gcloud compute tpus list --zone {zone} --format json"))
    tpu_names = [tpu['name'].split('/')[-1] for tpu in tpus if tpu['state'] != 'READY']
    return tpu_names


def delete_tpu(tpu_name: str, zone: str):
    print(f"Deleting {tpu_name} in {zone}")
    os.system(f"gcloud compute tpus tpu-vm delete {tpu_name} --zone {zone}")


@app.command()
def clean_up():
    for zone in get_zones():
        tpu_names = get_invalid_tpu_names(zone)
        for tpu_name in tpu_names:
            delete_tpu(tpu_name, zone)


def describe_resources_of_zone(zone: str):
    cmd = f"gcloud compute tpus list --zone {zone}"
    print(f"Running {cmd}")
    os.system(cmd)
    print(f"Done {cmd}")


def get_zones():
    return set([resource.zone for resource in RESOURCES])


@app.command()
def describe_resources():
    for zone in get_zones():
        describe_resources_of_zone(zone)


@app.command()
def print_all_tpu_names():
    for resource in RESOURCES:
        tpu_names = get_tpu_names(resource)
        for tpu_name in tpu_names:
            print(tpu_name)


@app.command()
def print_quota():
    for resource in RESOURCES:
        allocated = len(get_tpu_names(resource))
        quota = resource.quota
        print(f"{resource} | {allocated}/{quota}")


def create_machine(tpu_name: str, resource: Resource):
    cmd = f"CMD='gcloud compute tpus tpu-vm create {tpu_name} --zone {resource.zone} --accelerator-type {resource.tpu_type} --version tpu-vm-base --preemptible'; until $CMD; do sleep 3; done"
    if not resource.is_preemptible:
        cmd = cmd.replace("--preemptible", "")
    print(f"Running {cmd}")
    os.system(cmd)
    print(f"Done {cmd}")
    print(f"Created TPU {tpu_name} from resource: {resource}")


def create_and_run(tpu_name: str, resource: Resource):
    create_machine(tpu_name, resource)
    run_app(tpu_name, resource.zone)


def fill_quota_for_resource(resource: Resource):
    quota = resource.quota
    for i in range(1, quota + 1):
        tpu_names = get_tpu_names(resource)
        valid_tpu_names = get_valid_tpu_names(resource.zone)
        num_allocated = len(valid_tpu_names)
        tpu_name = f"tpu-{i}"
        if num_allocated == quota:
            print(f"resource: {resource} is full")
            break
        if num_allocated > quota:
            print(f"resource: {resource} EXCEEDED quota!!!!! HANDLE THIS!!!!")
            break
        if resource.is_preemptible:
            tpu_name += "-p"
        if tpu_name in tpu_names and tpu_name not in valid_tpu_names:
            delete_tpu(tpu_name, resource.zone)
        if tpu_name in valid_tpu_names:
            print(f"TPU {tpu_name} in resource {resource} already exists")
            continue
        create_and_run(tpu_name, resource)


@app.command()
def fill_quota():
    for resource in RESOURCES:
        fill_quota_for_resource(resource)
    prepare_backend_urls_env_vars_str()


@app.command()
def fill_quota_for_zone(zone: str):
    for resource in RESOURCES:
        if resource.zone == zone:
            fill_quota_for_resource(resource)
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
    for zone in get_zones():
        print(f"Zone {zone}:")
        tpu_names = get_tpu_names_by_zone(zone)
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
    for zone in get_zones():
        print(f"Zone {zone}:")
        tpu_names = get_valid_tpu_names(zone)
        for tpu_name in tpu_names:
            external_ip = get_external_ip_of_tpu(tpu_name, zone)
            print(f"{tpu_name}: {external_ip}")


@app.command()
def prepare_backend_urls_env_vars_str():
    env_vars_str = []
    for zone in get_zones():
        print(f"Zone {zone}:")
        tpu_names = get_valid_tpu_names(zone)
        for tpu_name in tpu_names:
            print(f"Adding {tpu_name=}")
            external_ip = get_external_ip_of_tpu(tpu_name, zone)
            env_vars_str.append(f"http://{external_ip}:5000/generate")
            env_vars_str.append(f"http://{external_ip}:5001/generate")
    print(f"BACKEND_URLS={json.dumps(env_vars_str)}")


if __name__ == '__main__':
    app()
