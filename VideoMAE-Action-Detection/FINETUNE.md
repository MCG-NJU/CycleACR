# Fine-tuning Instruction

The implementation supports **multi-node distributed training**. We provide the **off-the-shelf** scripts in the [scripts folder](scripts/ava).

-  For example, to fine-tune **CycleACR** with VideoMAE ViT-L backbone (**pre-trained & fine-tuned on k700**) on **AVA v2.2** with 32 GPUs (4 nodes x 8 GPUs), you can run

  ```bash
# Set the path to save checkpoints and logs
OUTPUT_DIR='YOUR_PATH/ava_videomae_vit_large_k700_pretrain+finetune_cycleacr'
# path to pretrain model
MODEL_PATH='YOUR_PATH_TO_PRETRAINED_MODEL/videomae-vitl-k700-16x4.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12320 --nnodes=4 \
      --node_rank=$1 --master_addr=$2 run_class_finetuning.py \
      --model vit_large_patch16_224 \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 4 \
      --input_size 224 \
      --save_ckpt_freq 1 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 2.5e-4 \
      --layer_decay 0.8 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 20 \
      --data_set "ava" \
      --drop_path 0.2 \
      --val_freq 10 \
      --lr_scale 1.0
  ```

  on the first node. On other nodes, run the same command with `--node_rank 1`, ..., `--node_rank 3` respectively.  `--master_addr` is set as the ip of the node 0.

  The results will be stored into `'YOUR_PATH/ava_videomae_vit_large_k700_pretrain+finetune_cycleacr/inference/result.log'`

  ```
{'PascalBoxes_PerformanceByCategory/AP@0.5IOU/answer phone': 0.8774243265637183,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/bend/bow (at the waist)': 0.5454838371001356,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/carry/hold (an object)': 0.6995938609118674,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/climb (e.g., a mountain)': 0.22614184076703142,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/close (e.g., a door, a box)': 0.34330539682628153,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/crouch/kneel': 0.4695719376769968,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/cut': 0.3688286713657408,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/dance': 0.7142979740182793,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/dress/put on clothing': 0.16288616190813235,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/drink': 0.4040026688769943,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/drive (e.g., a car, a truck)': 0.7032899854697596,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/eat': 0.5301125016663247,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/enter': 0.07599282324793781,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/fall down': 0.20370413977480936,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/fight/hit (a person)': 0.5955737883746683,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/get up': 0.4201350528997425,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/give/serve (an object) to (a person)': 0.2648374849314989,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/grab (a person)': 0.1420868276766687,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hand clap': 0.4929991185675148,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hand shake': 0.34235074158041623,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hand wave': 0.216025745601165,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hit (an object)': 0.19809996314764083,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/hug (a person)': 0.27632736346175946,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/jump/leap': 0.1987808387246536,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/kiss (a person)': 0.4223519899079323,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/lie/sleep': 0.5118203234922136,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/lift (a person)': 0.32053462908186514,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/lift/pick up': 0.06713370329577158,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/listen (e.g., to music)': 0.0582944191999971,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/listen to (a person)': 0.7542762030735352,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/martial art': 0.5550745444299658,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/open (e.g., a window, a car door)': 0.3974531420505743,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/play musical instrument': 0.7395103188172645,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/point to (an object)': 0.010966867959431052,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/pull (an object)': 0.09130896997970543,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/push (an object)': 0.223065758852855,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/push (another person)': 0.07937341395233959,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/put down': 0.08878331306326105,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/read': 0.5226027136621851,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/ride (e.g., a bike, a car, a horse)': 0.6342589173913136,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/run/jog': 0.6592833216387155,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/sail boat': 0.2505577512522059,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/shoot': 0.3444299326878697,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/sing to (e.g., self, a person, a group)': 0.47512819786946303,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/sit': 0.8699283641920188,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/smoke': 0.4711997610529014,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/stand': 0.8781074219708637,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/swim': 0.7142772636300567,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/take (an object) from (a person)': 0.1922323377140398,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/take a photo': 0.0687274935916688,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/talk to (e.g., self, a person, a group)': 0.8591373209555752,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/text on/look at a cellphone': 0.2862496408536627,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/throw': 0.13449507384377335,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/touch (an object)': 0.4205804746214338,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/turn (e.g., a screwdriver)': 0.11051991342676351,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/walk': 0.8293338784920672,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (a person)': 0.7798729865444604,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (e.g., TV)': 0.24616428331480153,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/work on a computer': 0.3900785730029905,
 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/write': 0.4073027482465523,
 'PascalBoxes_Precision/mAP@0.5IOU': 0.40560448363753054}
  ```
### Note:

- Here total batch size = (`batch_size` per gpu) x `nodes` x (gpus per node).
- `lr` here is the base learning rate. The ` actual lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `` actual lr`` = `lr` * total batch size / 256.
