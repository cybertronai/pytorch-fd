Implementation of https://arxiv.org/abs/1810.00004 for automatic learning rate annealing.

Bonus: TensorboardX logging (example below).

## Try the sample
```
git clone git@github.com:cybertronai/pytorch-fd.git
cd pytorch-fd
pip install -e .
python test_fd.py
tensorboard --logdir=runs
```

## Sample results
Green: `python test_fd.py --batch-size=512 --log-interval=30 --scheduler=fd`

Blue: `python test_fd.py --batch-size=512 --log-interval=30 --optimizer=cosine`
![](images/loss.png)


![](images/histogram.png)