# Knowledge Distillation Methods with Tensorflow
Knowledge distillation is the method to enhance student network by teacher knowledge.
So annually knowledge distillation methods have been proposed but each paper's do experiments with different networks and compare with different methods.
And each method is implemented by each author, so if a new researcher wants to study knowledge distillation, they have to find or implement all of the methods. Surely it is very hard work.
To reduce this burden, I publish some code that is modified from my research codes.
I'll update the code and knowledge distillation algorithm, and all of the things will be implemented by Tensorflow.

If you want something a new method, please notice to me :)


# Implemented Knowledge Distillation Methods
below methods are implemented and base on insight with [TAKD](https://arxiv.org/abs/1902.03393), I make each category. I think they are meaningful categories, but if you think it has problems please notice for me :)

## Response-based Knowledge
Defined knowledge by the neural response of the hidden layer or the output layer of the network
- Soft-logit : The first knowledge distillation method for deep neural network. Knowledge is defined by softened logits. Because it is easy to handle it, many applied methods were proposed using it such as semi-supervised learning, defencing adversarial attack and so on.
  - [Geoffrey Hinton, et al. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.](https://arxiv.org/abs/1503.02531)

## Multi-connection Knowledge
Increases knowledge by sensing several points of the teacher network
- FitNet : To increase amounts of information, knowledge is defined by multi-connected networks and compared feature maps by L2-distance.
  - [Adriana Romero, et al. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.](https://arxiv.org/abs/1412.6550)
  
- Attention transfer (AT) : Knowledge is defined by attention map which is L2-norm of each feature point.
  - [Zagoruyko, Sergey et. al. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016.](https://arxiv.org/pdf/1612.03928.pdf)
  
- Jacobian Matching (JB) (In process): They combine attention matching loss and Jacobian matching loss.
  -[Suraj Srinivas, Francois Fleuret. Knowledge Transfer with Jacobian Matching. arXiv preprint arXiv:1803.00443, 2018.](https://arxiv.org/pdf/1803.00443.pdf)
- Activation boundary (AB) : To soften teacher network's constraint, they propose the new metric function inspired by hinge loss which usually used for SVM.
  - [Byeongho Heo, et. al. Knowledge transfer via distillation of activation boundaries formed by hidden neurons. AAAI2019](https://arxiv.org/abs/1811.03233)

## Shared-representation Knowledge
Defined knowledge by the relation between two feature maps
- Flow of Procedure (FSP) : To soften teacher network's constraint, they define knowledge as relation of two feature maps.
  - [Junho Yim, et. al. A gift from knowledge distillation:
Fast optimization, network minimization, and transfer learning. CVPR 2017.](http://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html)
- KD using Singular value decomposition(KD-SVD) : To extract major information in feature map, they use singular value decomposition.
  - [Seung Hyun Lee, et. al. Self-supervised knowledge distillation using singular value decomposition. ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/html/SEUNG_HYUN_LEE_Self-supervised_Knowledge_Distillation_ECCV_2018_paper.html)

## Relational Knowledge
Defined knowledge by intra-data relation
- Relational Knowledge Distillation : they propose knowledge which contains not only feature information but also intra-data relation information.
  - [Wonpyo Park, et. al. Relational Knowledge Distillation. CVPR2019](https://arxiv.org/abs/1904.05068?context=cs.LG)

# Experimental Results
The below table and plot are sample results using [ResNet](http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

I use the same hyper-parameter for training each network, and only tune hyper-parameter of each distillation algorithm. But the results may be not optimal. All of the numerical values and plots are averages of five trials.



## Network architecture
The teacher network is ResNet32 and Student is ResNet8, and the student network is well-converged (not over and under-fit) for evaluating each distillation algorithm performance precisely.

## Training/Validation plots

Methods | Last Accuracy | Best Accuracy
------------| ------------- | -------------
Student     | 71.76 | 71.92 
Teacher     | 78.96 | 79.08 
Soft-logits | 71.79 | 72.08 
FitNet      | 72.74 | 72.96
AT          | 72.31 | 72.60
FSP         | 72.65 | 72.91
KD-SVD      | 73.68 | 73.78
AB          | 72.80 | 73.10
RKD         | 73.40 | 73.48

<img src="plots.png" width="600">

# Plan to do
- Implement the Jacobian matching
