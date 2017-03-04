Training commands:
    python galaxian2.py --train --num_workers=6 --ui_tasks=1 --eval_tasks=1 --port=5000
    tail /tmp/galaxian-*.stderr -n1; nvidia-smi
    tensorboard --logdir=logs/2.30

If the training expodes, set lower learning rate, e.g. --learning_rate=1e-5.
