GPU=1

# +

CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --dataset_order=Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
    --warmup_length=0 \
    --ls 0.2 \
    --lr 0.008 0.008 0.008 0.008 0.008 0.008 0.008 0.008 0.02 0.02 0.02 \
    --iterations 1000 \
    --eval-interval 500 \
    --method ours \
    --save 'ckpts/ours_order1' \
    --data-location="data/"

CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --dataset_order=StanfordCars,Food,MNIST,OxfordPet,Flowers,SUN397,Aircraft,Caltech101,DTD,EuroSAT,CIFAR100 \
    --warmup_length=0 \
    --ls 0.2 \
    --lr 0.02 0.008 0.008 0.02 0.008 0.02 0.008 0.008 0.008 0.008 0.008 \
    --iterations 1000 \
    --eval-interval 500 \
    --method ours \
    --save 'ckpts/ours_order2' \
    --data-location="data/"


CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --dataset_order=Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
    --ls 0.0 \
    --lr 0.03 0.02 0.02 0.01 0.01 0.01 0.001 0.02 0.01 0.03 0.02 \
    --few-shot 5 \
    --batch-size 32 \
    --iterations 300 \
    --eval-interval 150 \
    --method ours_semi \
    --save "ckpts/ours_order1_5shot" \
    --data-location="data/"

CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --dataset_order=StanfordCars,Food,MNIST,OxfordPet,Flowers,SUN397,Aircraft,Caltech101,DTD,EuroSAT,CIFAR100 \
    --ls 0.0 \
    --lr 0.03 0.001 0.02 0.01 0.01 0.02 0.03 0.02 0.01 0.01 0.02 \
    --few-shot 5 \
    --batch-size 32 \
    --iterations 300 \
    --eval-interval 150 \
    --method ours_semi \
    --save "ckpts/ours_order2_5shot" \
    --data-location="data/"

