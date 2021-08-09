# Corrective neural networks (cornet)
## How to train minimal 3 times faster. Train a big network on your laptop!


## The basic idea behind cornet

We tend to make a complex neural net and then train it all together. In nature that is not how things work. 
Often during evolution extra complexity is added without to add new or improved functionality. Not all layers for at once.

This idea was taken earlier by Nvidea to make the progressive Gan. But although they add layers, everything underneath keeps trainable and therefor changing.
It helps with training time, but can we not do (much) better...

Introducing the cornet. Corrective neural net. We train a very simple model till it doesn't get better then stick on a new layer of complexity to correct what it could not get right.
We only train the last block (can be one or multiple layers) and the new block. Everything before we see as set in stone.

According to common understanding of neural networks this should not give an accurate result, because always only the last blocks/layers are trained.

But is the Intuitive conception correct?

Lets test is out with a standard, 14 blocks, resnet for image categorisation and use cifar10.
I know this is probably not the best network for cifar10, but it is an easy way to show the principles. This method works very well on resnet type of architecture, networks
where we add results from an earlier block, but it seems to work well on all deep networks (try it with one of these deep NLP networks!)


first, the original code as a reference:


Lets first train a standard resnet architecture. See base_ori (model) and train_ori. (you can just run train_ori)


All three networks where trained on cifar 10 over 10 epochs. Results are from test dataset.

The Cornet and the pronet start with 1 resblock. A new one in added every 500 batches to end with 14. 
Same a the basic resnet that is used as a reference.

|Network|Time for 10 epochs | Results(loss) |result(acc)|
|----------|-------------------|---------------|-----------|
|resnet 14 | 1118 seconds      | 1.0386        | 0.7131    | 
|pronet(*) | 700 seconds       | 1.4383        | 0.6486    |
|cornet    | 367 seconds       | 0.7579        | 0.7475    |

(*) To try the progressive net set trainold=False to trainold=True in line 68 or train_cornet.
In this cornet the number of steps before adding a new block are linear (fixed). This is not the best way to do it, but still working on that.


This idea/work in mine, code and ideas are mine. Basic keras implementation of resnet and the residual blocks come from the internet (stackexchange). I don't know the original author.

Should this cornet already exists under a different name and/or have libs etc. Let me know.

It helps me train huge NLP networks on my laptop in a fraction of the original time. So more improvements or tips very welcome.


Remko Weingarten

Remko66@gmail.com


Here the loss graph for cornet. Look at its distinctive pattern:

![loss](./graph/cornet_loss.png)

Compared to training of original:

![loss](./graph/ori_loss.png)