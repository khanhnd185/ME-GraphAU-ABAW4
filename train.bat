d:\Python\Python310\python.exe train_au_stage1.py --dataset SAW2 --arc resnet18 --exp-name resnet18_first_stage -b 64 -lr 0.0001 -j 0 -c 12 -n 4
d:\Python\Python310\python.exe train_au_stage2.py --dataset SAW2 --arc resnet18 --exp-name resnet18_second_stage -b 32 --resume results/resnet18_first_stage/bs_64_seed_0_lr_0.0001/cur_model_fold1.pth --lam 0.1
d:\Python\Python310\python.exe train_au_stage1.py --dataset SAW2 --arc resnet34 --exp-name resnet34_first_stage -b 64 -lr 0.0001 -j 0 -c 12 -n 4
d:\Python\Python310\python.exe train_au_stage2.py --dataset SAW2 --arc resnet34 --exp-name resnet34_second_stage -b 32 --resume results/resnet34_first_stage/bs_64_seed_0_lr_0.0001/cur_model_fold1.pth --lam 0.1
d:\Python\Python310\python.exe train_au_stage1.py --dataset SAW2 --arc resnet50 --exp-name resnet50_first_stage -b 64 -lr 0.0001 -j 0 -c 12 -n 4
d:\Python\Python310\python.exe train_au_stage2.py --dataset SAW2 --arc resnet50 --exp-name resnet50_second_stage -b 32 --resume results/resnet50_first_stage/bs_64_seed_0_lr_0.0001/cur_model_fold1.pth --lam 0.1
d:\Python\Python310\python.exe train_au_stage1.py --dataset SAW2 --arc swin_transformer_base --exp-name swin_base_first_stage -b 32 -lr 0.0001 -j 0 -c 12 -n 4
d:\Python\Python310\python.exe train_au_stage2.py --dataset SAW2 --arc swin_transformer_base --exp-name swin_base_second_stage -b 16 --resume results/swin_base_first_stage/bs_32_seed_0_lr_0.0001/cur_model_fold1.pth --lam 0.1
d:\Python\Python310\python.exe train_au_stage1.py --dataset SAW2 --arc swin_transformer_tiny --exp-name swin_tiny_first_stage -b 64 -lr 0.0001 -j 0 -c 12 -n 4
d:\Python\Python310\python.exe train_au_stage2.py --dataset SAW2 --arc swin_transformer_tiny --exp-name swin_tiny_second_stage -b 32 --resume results/swin_tiny_first_stage/bs_64_seed_0_lr_0.0001/cur_model_fold1.pth --lam 0.1
