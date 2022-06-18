# Hung-Yee-Li-ML2021
李宏毅2021ML课程作业<br/>
HWN 任务类型 数据集 训练技巧<br/>
HW1 Regression 新冠感染人数预测 Ensemble Model<br/>
HW2 Classification TIMIT11帧Phoneme分类<br/>
HW3 CNN Classification food11数据集 Data Augmentation<br/>
HW4 Self Attention 语者辨识 voxceleb数据集<br/>
HW5 Transformer Seq2Seq机器翻译 TED演讲中英文字幕 大模型+Semi Supervise Learning<br/>
HW6 GAN Anime头像生成 Crypko二次元头像 DCGAN->WGAN<br/>
HW7 BERT for QA Chinese Reading Comprehension 把输入的窗口调大,改stride让输入有overlap,限制输出长度,把dev集并入训练集训练<br/>
HW8 Anomaly Detection 不知名的人脸图像,女人为normal,男人为abnormal 小型CNN+BN+lr_scheduler<br/>
HW9 Explainable AI food11 答题回<br/>
HW10 Adversarial Attack 用i-fgsm算法攻击CIFAR-10的pretrain model<br/>
HW11 Domain Adversarial 在真实图片上训练,在涂鸦图片上测试 算法为DaNN,原理类似GAN,用一个feature extractor把两个domain的特征抽的越近越好(骗过Discriminator)<br/>
HW12 RL月球登录,我的绝活,直接DDQN爆杀,注意调网络大小就好<br/>
HW13 Model Compression 压缩一个food11上的pretrain的resnet18,参数<100,000 用到知识蒸馏,卷积分离,网络剪枝<br/>
HW14 Life Long Learning 在5个不同旋转角度的mnist上做LLL,整体思想就是在训练当前task时,在loss_fn上加上一项来限制当前网络参数与之前网络的差别
