CMD="gcloud compute tpus tpu-vm create tpu-$1 \
--zone europe-west4-a \
--accelerator-type v3-8 \
--version tpu-vm-base \
--metadata=startup-script=startup.sh
"

until $CMD
do
    sleep 5
done
