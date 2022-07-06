# 2D Pose Basic

`2022.7.6`

## 0. Null

对于单人姿态估计，输入是一个 crop 出来的行人，然后在行人区域位置内找出需要的关键点，比如头部，左手，右膝等。

对于**多人姿态估计**，目前主要有 2 种主流思路，分别是 **top-down** 以及 **bottom-up** 方法。对于 top-down 方法，往往先找到图片中所有行人，然后对每个行人做姿态估计，寻找每个人的关键点。单人姿态估计往往可以被直接用于这个场景。对于 bottom-up，思路正好相反，先是找图片中所有 parts （关键点），比如所有头部，左手，膝盖等，然后把这些 parts（关键点）组装成一个个行人。



 ## 1. Dataset

- [LSP（Leeds Sports Pose Dataset）](https://sam.johnson.io/research/lsp.html)：单人人体关键点检测数据集，关键点个数为14，样本数2K，在目前的研究中作为第二数据集使用。
- [FLIC（Frames Labeled In Cinema）](https://bensapp.github.io/flic-dataset.html)：单人人体关键点检测数据集，关键点个数为9，样本数2W，在目前的研究中作为第二数据集使用。
- [MPII（MPII Human Pose Dataset）](http://human-pose.mpi-inf.mpg.de/)：单人/多人人体关键点检测数据集，关键点个数为16，样本数25K，是单人人体关键点检测的主要数据集。它是 2014 年由马普所创建的，目前可以认为是单人姿态估计中最常用的 benchmark， 使用的是 [PCKh](http://human-pose.mpi-inf.mpg.de/#results) 的指标。
- [MSCOCO](https://cocodataset.org/#home)：多人人体关键点检测数据集，关键点个数为17，样本数量多于30W。目前是多人关键点检测的主要数据集，使用的是 [AP 和 OKS](https://cocodataset.org/#keypoints-eval) 指标。
- [human3.6M](http://vision.imar.ro/human3.6m/description.php)：是 3D 人体姿势估计的最大数据集，由 360 万个姿势和相应的视频帧组成，这些视频帧包含11 位演员从4个摄像机视角执行 15 项日常活动的过程。数据集庞大将近100G。
- [PoseTrack](https://posetrack.net/)：最新的关于人体骨骼关键点的数据集，多人人体关键点跟踪数据集，包含单帧关键点检测、多帧关键点检测、多人关键点跟踪三个人物，多于500个视频序列，帧数超过20K，关键点个数为15。



## 2. Evaluation

### 2.1 pck

(percentage of correct keypoints)
$$
P C K=\frac{\sum_{i} \delta\left(\frac{d_{i}}{d} \leq T\right)}{\sum_{i} 1}
$$
其中：

* `di` 表示第`i`个关节点的预测值和 `groundtruth` 的欧氏距离
* `d` 是一个人体的尺度因子
* 认为当小于阈值`T`时分类正确

### 2.2 oks

(object keypoint similarrity)
$$
O K S_{p}=\frac{\sum_{i} \exp \left\{-d_{p i}^{2} / 2 S_{p}^{2} \sigma_{i}^{2}\right\} \delta\left(v_{p i}=1\right)}{\sum_{i} \delta\left(v_{p i}=1\right)}
$$
其中:

* p  表示`groudtruth`中，人的`id`
* i  表示`keypoint`的`id`
* $d_{p i} $ 表示`groudtruth`中每个人和预测的每个人的关键点的欧氏距离
* $S_{p} $  表示当前人的尺度因子，这个值等于此人在 `groundtruth`中所占面积的平方根  $\mathrm{Q}$  ，即  $\sqrt{\left(x_{2}-x_{1}\right)\left(y_{2}-y_{1}\right)} $
* $ \sigma_{i}$   表示第i个骨骼点的归一化因子，这个因此是通过对数据集中所有 `groundtruth`计算的标准差  $\mathrm{Q} $ 而得到的，反映出当前骨骼点标注时候的标准差， $\sigma $ 越大表示这个点越难标注。
* $v_{p i}$  代表第 $ \mathrm{p}  $个人的第 $\mathrm{i} $ 个关键点是否可见
*  $\delta $ 用于将可见点选出来进行计算的函数



### 2.3 AP

(Average Precision)



这个和目标检测里的 AP 概念是一样的，只不过度量方式 iou 芙换成了 oks 。如果 oks 大于阈值  T  ，则认为该关键点被成功检侧到。单人姿态估计和多人姿态估计的计算方式不同。对于单人姿态估计的AP， 目标图片中只有一个人体，所以计算方式为:
$$
A P=\frac{\sum_{p} \delta\left(o k s_{p}>T\right)}{\sum_{p} 1}
$$




对于多人姿态估计而言，由于一张图像中有  \mathrm{M}  个目标，假设总共预测出  \mathrm{N}  个人体，那么groundtruth和预 测值之间能构成一个  M x N  的矩阵，然后将每一行的最大值作为该目标的 oks，则：
$$
A P=\frac{\sum_{m} \sum_{p} \delta\left(o k s_{p}>T\right)}{\sum_{m} \sum_{p} 1}
$$






