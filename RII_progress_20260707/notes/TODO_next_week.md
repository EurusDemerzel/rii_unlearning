# RII 项目下周计划（7.14 起）

[ ] 1. FineTune + SISA on CIFAR-100
    python pipeline.py --dataset cifar100 --model cnn --unlearn_method all --forget_ratio 0.05 0.10 --epochs 30

[ ] 2. RII-Regularized Unlearning
    修改 unlearn.py：loss = CE(retain) + λ * ||μ_f - μ_r||²
    对比 λ=[0.01, 0.1, 1.0, 10.0] 的效果

[ ] 3. 论文图表
    Fig 1: RII vs Dataset Complexity (MNIST/CIFAR-10/CIFAR-100)
    Fig 2: MIA vs RII scatter (orthogonal leakage axes)
    Fig 3: RII pullback curve (forget ratio 5%→50%)

[ ] 4. 写论文 Section 3 (Methodology) + Section 4 (Experiments)

[ ] 5. (可选) ResNet-18 on CIFAR-100
    python pipeline.py --dataset cifar100 --model resnet18 --epochs 30
