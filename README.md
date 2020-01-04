# SJTU_M3DV

---------项目说明---------

本项目用于SJTU倪冰冰老师的机器学习课程大作业。

---------编译环境---------

系统配置：Win10系统，GeForce GTX 1050
	
框架：pytorch
	
编译工具：VS2017+python3.6

---------程序文件说明-------------

emersion.py:用来复现出最好结果的程序
	
test.py:主程序，生成Submission文件，并展示loss图像
	
read_data.py: 读取数据程序
	
mylenet3d.py:模型程序，用以生成训练和测试所用的模型
	
my_train.py:训练程序
	
my_test.py:测试程序

---------复现步骤--------------
	
1.将所有的文件（包括.py与.pth）下载下来，与测试集文件夹放在同一目录下
	
2.运行emersion.py文件，生成的emersion.csv文件即为复现结果。
