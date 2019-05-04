# Knowledge Distillation Methods with Tensorflow
(Sadly, my English skill is terribly low!! so please get patient to read my mess of words. :) )

Knowledge distillation is the method to enhance student network by teacher knowledge.
So annually knowledge distillation methods have been proposed but each paper's do experiments with different networks and compare with different methods.
And each method is implemented by each author, so if new researcher want to study about knowledge distillation, they have to find or implement all of methods. Surely it is very hard work.
To reduce this burden, I publish some code which are modified from my research codes.
I'll update code and knowledge distillation algorithm, and all of things will be implemented by Tensorflow.
Now this repository is not so perfect but i will be improved, so please give some advise to be. (especially English.. :D)

# Implemented Knowledge Distillation Methods
below methods are implemented and base on insight with [TAKD](https://arxiv.org/abs/1902.03393), I make each category. I think they are meaningful category, but if you think it has problem please notice for me :)

## Response-based Knowledge
Defined knowledge by the neural response of the hidden layer or the output layer of the network
- Soft-logit : [Geoffrey Hinton, et al. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.](https://arxiv.org/abs/1503.02531)

## Multi-connection Knowledge
Increases knowledge by sensing several points of the teacher network
- FitNet : [Adriana Romero, et al. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.](https://arxiv.org/abs/1412.6550)
- Activation boundary (AB) : [Byeongho Heo, et. al. Knowledge transfer via distillation of activation boundaries formed by hidden neurons. AAAI2019](https://arxiv.org/abs/1811.03233)

## Shared-representation Knowledge
Defined knowledge by the relation between two feature maps
- Flow of Procedure (FSP) : [Junho Yim, et. al. A gift from knowledge distillation:
Fast optimization, network minimization and transfer learning. CVPR 2017.](http://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html)
- KD using Singular value decomposition(KD-SVD) : [Seung Hyun Lee, et. al. Self-supervised knowledge distillation using singular value decomposition. ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/html/SEUNG_HYUN_LEE_Self-supervised_Knowledge_Distillation_ECCV_2018_paper.html)

## Relational Knowledge
Defined knoweldge by intra-data relation
- Relational Knowledge Distillation (In process) [Wonpyo Park, et. al. Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068?context=cs.LG)

# Expriment Results
직접 [ResNet](http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)을 통해 Teacher-Student networks를 구현 및 학습한 결과입니다.
network를 학습하기 위한 hyper-parameter는 모두 통일하여 학습했으며, 각 기법에서 사용하는 hyper-parameter만 tuning하였습니다. 많은 수의 실험을 하지 않았기 때문에 hyper-parameter는 optimal이 아닐 수 있으며 아래 table과 표의 수치는 참고로만 봐주시면 될 것 같습니다. Table의 있는 수치와 plot은 5번의 결과를 평균낸 것입니다.



## Network architecture
Teacher는 ResNet-32, Student는 ResNet-8을 사용한 것으로 각 network는 tranining accuracy 기준으로 충분히 converged시킨 후 Knowledge distillation을 적용하였습니다.

## Training/Validation plots

Methods | Last Accuracy | Best Accuracy
------------ | ------------- | -------------
Student     | 71.76 | 71.92 
Teacher     | 78.96 | 79.08 
Soft-logits | 71.79 | 72.08 
FitNet        | 72.74 | 72.96
FSP          | 71.56 | 71.70
KD-SVD    | 73.68 | 73.78
AB            | 72.80 |73.10

<img src="plots.png" width="600">

