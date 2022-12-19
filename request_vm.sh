tpu=$2

if [[ $tpu == "v2-8" ]]
then
  zone="us-central1-f"
elif [[ $tpu == "v3-8" ]]
then
  zone="europe-west4-a"
else
  echo "Unknown TPU type: $tpu"
  exit 1
fi

CMD="gcloud compute tpus tpu-vm create tpu-$1 \
--zone $zone \
--accelerator-type $tpu \
--version tpu-vm-base \
--metadata=startup-script=startup.sh
"
echo "$CMD"

until $CMD
do
    sleep 5
done
