# Deep Learning Final Project

This work is built based on the [minillm paper](https://arxiv.org/abs/2306.08543), here's the repo for the original paper: https://github.com/microsoft/LMOps/tree/main/minillm


Pretrain the teacher model (OPT-6.7B):

```
bash script/pretrain_teacher.sh
```

Distillation:
```
bash script/distill_peft.sh
```

Requirement of usage: 
```
torch                     2.4.1 
transformers              4.45.1
peft                      0.11.2.dev0
accelerate                0.34.2 
```


