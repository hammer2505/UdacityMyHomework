# 项目4：增强学习
## 训练智能出租车学会驾驶

这份报告是根据评估要求整理的格式及内容，利用的markdown书写的内容，并输出为pdf文件，如果在内容和格式上有什问题，望指正。 另外在语言上，由于英语并不很好，觉得之前的问答形式还可以应对，但这次需要整体写报告，所以采用了中文。 编程环境为 Ubuntu15.10 + Anaconda

~~谢谢reviewer给出的提示，我将程序进行了重写,希望这次能够领会项目的要求~~

---

### 实施基础驾驶智能体
在没有对`smartcab/agent.py`时，利用IDE：Spyder运行
发现程序报错

```
Simulator.__init__(): Error initializing GUI objects;
display disabled.
error: Couldn't open images/car-orange.png
Simulator.run(): Trial 0
```

查找原因发现是Spyder没有权限访问，然后就改成在终端下输入

```sudo python smartcab/agent.py```

成功执行程序，发现小红车并不会动。。。

修改`agent.py`
```
# TODO: Select action according to your policy
action = random.choice(('forward', 'left', 'right'))
```
以及
```
e.set_primary_agent(a, enforce_deadline = False)
# specify agent to track
```

运行结果如下所示

![效果图1](/photo/截图1.png)

![效果图2](/photo/截图2.png)

由于随机选择方向，小车顺利到达目的地的成功率太低，就不统计了。

---

### 确认并更新状态
由于增强学习之前并没有接触过，觉得公开课上的有点费力，提个小小的建议，能不能像监督式学习那样提供一些文档以增加理解，自己找了一份[Q-learning 算法的简明教程](http://www.cnblogs.com/stevenbush/p/3359603.html)
。感谢riviewer的提醒，傻傻地我想了一天终于想出来了一种状态的表示方式，我将状态state设为小车应该前往的方向，小车的环境，即红绿灯和三个方向是否有车来进行结合编码。
程序如下：
```
# TODO: Update state
self.state = 0    
dic1 = {None:0, 'forward':1, 'right':2, 'left':3 }
dic2 = {'green':0, 'red':1}
self.state = self.state + 1000 * dic1[self.next_waypoint]
self.state = self.state + 100  * dic2[inputs['light']]
self.state = self.state + 10   * dic1[inputs['oncoming']]
self.state = self.state + 1     * dic1[inputs['left']]
```
解释：
我将状态state表示为4位数字，各个位表示意义如下：
第一位表示小车的应该运行的大方向即变量`self.next_waypoint`  
第二位表示红绿灯状态  
第三位表示小车正面来车运行方向   
第四位表示小车左侧来车运行方向  
各位意义如程序中建立的两个字典所示。  

状态state时为了减少计算量，减少了重复的状态：小车右侧来车运行方向，因为在交通规则里右侧来车并不会影响小车的前进或停止。

在选择状态state是放弃了将小车的位置作为状态state，因为小车在哪里并不能作为评判标准，不可能在某一路口，不管红绿灯和目的地方向，固定前行的方向，所以就抛弃了位置作为小车状态state。

![效果图3](/photo/Screenshot1.png)


---
### 实现 Q 学习
Q-learning 公式：

$$ Q(s,a) = R(s,a) + \gamma \cdot max{Q(s^\sim,a^\sim)} $$  

>智能出租车每成功完成一段行程就会获得奖励。“成功完成”行程的意思是，在预先指定的时间限制（利用路线方案算出）内，将乘客送到预定的目的地（某个十字路口）。
此外，如果它在十字路口进行正确的移动，则会获得较小的奖励；如果进行错误的移动，则会获得小惩罚；如果违反交通规则和/或造成事故，则会获得较大的惩罚。

读程序发现，程序已经在 `environment.py` 写好了奖励程序。
建立 $Q(s,a)$ 矩阵， 矩阵的行就是小车的运动方向，即a，一共是4行，列是state。
Q初始化
```
global Q
Q = [[0 for col in range(9999)] for row in range(4)]
```

寻找方向a,在这里`epsilon`我选择为参数0,即前进方向只寻找使Q最大的方向
```
# TODO: Select action according to your policy  
action = random.choice(('forward', 'left', 'right', None))
epsilon = 0 # in my opinion, epsion is only equal to 0 or 1
Qold = Q[dic1[action]][int(self.state)]

Qmax = Q[1][int(self.state)]
actionbest = 'forward'
if Q[2][int(self.state)] > Qmax:
    Qmax = Q[2][int(self.state)]
    actionbest = 'right'
if Q[3][int(self.state)] > Qmax:  
    Qmax = Q[3][int(self.state)]
    actionbest = 'left'
elif Q[0][int(self.state)] > Qmax:
    Qmax = Q[0][int(self.state)]
    actionbest = None

if epsilon:
    action = action
else:
action = actionbest
```

Q值更新，这里 $\alpha$ 设为了0.8,  $\gamma$ 设为了0.9
```
# TODO: Learn policy based on state, action, reward
alpha = 0.8
gamma = 0.9

self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
inputs = self.env.sense(self)
self.state_1 = 0    
self.state_1 = self.state_1 + 1000 * dic1[self.next_waypoint]
self.state_1 = self.state_1 + 100  * dic2[inputs['light']]
self.state_1 = self.state_1 + 10   * dic1[inputs['oncoming']]
self.state_1 = self.state_1 + 1     * dic1[inputs['left']]

Qmax = Q[1][int(self.state_1)]
actionbest = 'forward'
if Q[2][int(self.state_1)] > Qmax:
    Qmax = Q[2][int(self.state_1)]
    actionbest = 'right'
if Q[3][int(self.state_1)] > Qmax:  
    Qmax = Q[3][int(self.state_1)]
    actionbest = 'left'
elif Q[0][int(self.state_1)] > Qmax:
    Qmax = Q[0][int(self.state_1)]
    actionbest = None

Qnew = (1- alpha) * Qold + alpha * (reward + gamma * Qmax)
Q[dic1[action]][int(self.state)] = Qnew
```
运行效果如下所示
![](/photo/截图4.png)

可以发现即使是在Q-learning的开始阶段reward净奖励数大于0, 而且可以成功到达目的地。统计之后发现成功率为82%。即只有18次超过了规定时间，没有到达目的地。在没有实行Q-learning算法时，reward会为负，而且经常无法按时到达目的地。

### 参数调整

我尝试对Q初值均进行调整，以便能够找到更好的小汽车。
1. 将 Q 初值全部设为1.0, 成功率为95%，运行日志为log2
2. 将 Q 初值全部设为-1.0, 成功率为100%，运行日志为log3
说明当Q值初值全部为1.0时，Q矩阵更容易收敛，小车的成功率更高。

我尝试对epsilon调整，不理解epsilon如何为一个小数，即0.4×forward+0.6×None 的输出应该为什么

我尝试对alpha进行调整，以便能够找到更好的小车，（其他参数不变）
1. 将alpha参数设0.5, 成功率为96%， 运行日志为log4
2. 将alpha参数设0.2, 成功率为100%， 运行日志为log5

我尝试对gamma进行调整，以便能够找到更好的小车，（其他参数不变）
1. 将gamma参数设0.5, 成功率为98%， 运行日志为log6
2. 将gamma参数设0.2, 成功率为99%， 运行日志为log7

综上，尝试将所有最优的参数选出，Q初值0.0,alpha参数0.2,gamma为0.9为最优运行结果。

另外针对reviewer提出的小车left次数过少问题，原因是再索引Qmax时最后比较 $Q(left)$，只有 $Q(left)$ 比所有其他可能都大时才能出现(log1,第28次结果)
![](/photo/Screenshot2.png)

### 总结
从最终驾驶智能体能够在遵守交通规则的情况下按时到达目的地，学习最佳策略的关系依赖于一定数量的训练，形成收敛的 $Q(s,a)$ 矩阵，目前已经实现了$Q(s,a)$ 矩阵在较短的时间内收敛，我觉得之后可以考虑在训练时间，即多久后reward永远大于等于0和小车到达目的地的时间两方面对小车的性能进一步提高。
