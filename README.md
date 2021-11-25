# 太极图形课S1-显式弹性物体仿真

## 背景简介
本文基于显式时间积分和弹簧质点系统以及线性有限元系统，实现了弹性悬臂梁的仿真。
基于隐式时间积分和弹簧质点系统，实现了布料的仿真，参考文件[1](https://www.cs.cmu.edu/~baraff/papers/sig98.pdf)。
## 成功效果展示

### FEM demo
![fem demo](./data/fem.gif)


### Implicit mass spring demo
左侧：Direct solver；右侧： Conjugate Gradient (CG) solver。
![Mass spring demo](./data/direct_vs_cg.gif)

## 运行环境
```
[Taichi] version 0.8.3, llvm 10.0.0, commit 021af5d2, win, python 3.8.10
```

## 运行方式
在安装了taichi的情况下，可以直接运行：
```
python3 [*].py
```

在运行 `implicit_mass_spring_system.py`时，可以通过命令行参数 `-cg` 来控制是否使用 **CG** solver。
```
# 使用Direct solver
python implicit_mass_spring_system.py  

# 使用CG solver
python implicit_mass_spring_system.py  -cg
```