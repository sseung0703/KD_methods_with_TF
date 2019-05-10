# Knowledge Distillation Methods with Tensorflow
(Sadly, my English skill is terribly low!! so please get patient to read my mess of words. :) )

Knowledge distillation is the method to enhance student network by teacher knowledge.
So annually knowledge distillation methods have been proposed but each paper's do experiments with different networks and compare with different methods.
And each method is implemented by each author, so if a new researcher wants to study knowledge distillation, they have to find or implement all of the methods. Surely it is very hard work.
To reduce this burden, I publish some code that is modified from my research codes.
I'll update the code and knowledge distillation algorithm, and all of the things will be implemented by Tensorflow.
Now this repository is not so perfect but I will be improved, so please give some advice to be. (especially English.. :D)

# Implemented Knowledge Distillation Methods
below methods are implemented and base on insight with [TAKD](https://arxiv.org/abs/1902.03393), I make each category. I think they are meaningful categories, but if you think it has problems please notice for me :)

## Response-based Knowledge
Defined knowledge by the neural response of the hidden layer or the output layer of the network
- Soft-logit : [Geoffrey Hinton, et al. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.](https://arxiv.org/abs/1503.02531)

## Multi-connection Knowledge
Increases knowledge by sensing several points of the teacher network
- FitNet : [Adriana Romero, et al. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.](https://arxiv.org/abs/1412.6550)
- Attention transfer (AT) : [Zagoruyko, Sergey et. al. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016.](https://arxiv.org/pdf/1612.03928.pdf)
- Activation boundary (AB) : [Byeongho Heo, et. al. Knowledge transfer via distillation of activation boundaries formed by hidden neurons. AAAI2019](https://arxiv.org/abs/1811.03233)

## Shared-representation Knowledge
Defined knowledge by the relation between two feature maps
- Flow of Procedure (FSP) : [Junho Yim, et. al. A gift from knowledge distillation:
Fast optimization, network minimization, and transfer learning. CVPR 2017.](http://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html)
- KD using Singular value decomposition(KD-SVD) : [Seung Hyun Lee, et. al. Self-supervised knowledge distillation using singular value decomposition. ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/html/SEUNG_HYUN_LEE_Self-supervised_Knowledge_Distillation_ECCV_2018_paper.html)

## Relational Knowledge
Defined knowledge by intra-data relation
- Relational Knowledge Distillation (In process) [Wonpyo Park, et. al. Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068?context=cs.LG)

# Experimental Results
The below table and plot are sample results using [ResNet](http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

I use the same hyper-parameter for training each network, and only tune hyper-parameter of each distillation algorithm. But the results may be not optimal. All of the numerical values and plots are averages of five trials.



## Network architecture
The teacher network is ResNet32 and Student is ResNet8, and the student network is well-converged (not over and under-fit) for evaluating each distillation algorithm performance precisely.

## Training/Validation plots

Methods | Last Accuracy | Best Accuracy
------------ | ------------- | -------------
Student     | 71.76 | 71.92 
Teacher     | 78.96 | 79.08 
Soft-logits | 71.79 | 72.08 
FitNet        | 72.74 | 72.96
AT           | 72.31 | 72.60
FSP          | 71.56 | 71.70
KD-SVD    | 73.68 | 73.78
AB            | 72.80 |73.10

<img src="plots.png" width="600">

