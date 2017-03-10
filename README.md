Training commands:
    python galaxian2.py --train --train_pnn --num_workers=6 --ui_tasks=1 --eval_tasks=1 --port=5000
    tail /tmp/galaxian-*.stderr -n1; nvidia-smi
    tensorboard --logdir=logs/2.32
If the training expodes, set lower learning rate, e.g. --learning_rate=1e-5.

Playing commands:
    python galaxian2.py --train=False --ui_tasks=0 --eval_tasks=0 --search
