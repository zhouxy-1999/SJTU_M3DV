# SJTU_M3DV

---------项目说明---------

本项目用于SJTU倪冰冰老师的机器学习课程大作业。

---------编译环境---------

系统配置：Win10系统，GeForce GTX 1050
框架：pytorch
编译工具：VS2017+python3.6

---------程序文件说明-------------

emersion.py:用来复现出最好结果的程序
test.py:主程序
read_data.py: 读取数据程序
mylenet3d.py:模型程序，用以生成训练和测试所用的模型
my_train.py:训练程序，用以训练模型，并返回loss，生成图像
my_test.py:测试程序，用来测试数据

---------复现步骤---------------
1.先将sjtu3d文件夹保存在./目录下（即c盘用户目录下）。然后将给出的trian_val和test数据集保存下载到sjtu3d文件夹里面。
2.模型存放位置在my_model里面，存放路径为D盘目录即可
3.运行fuxian.py文件即可，寻找D盘目录下test文件夹里面的submission.csv文件即可。
