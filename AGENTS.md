# 农作物叶片病害识别系统

## 项目概述
本项目旨在开发一个基于深度学习的农作物叶片病害识别系统，能够准确识别多种农作物叶片病害。

## 数据集
- 训练集：AgriculturalDisease_trainingset
- 验证集：AgriculturalDisease_validationset
- 包含多种病害类别，编号从0到58

## 系统架构
- 数据加载模块：负责加载和预处理图像数据
- 模型定义模块：定义CNN模型架构
- 训练脚本：模型训练和验证
- 推理脚本：模型推理和预测
- 工具函数模块：辅助功能

## 技术栈
- Python
- PyTorch
- OpenCV
- PIL
- NumPy