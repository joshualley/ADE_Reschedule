#coding: utf-8
import scipy.io as sio
import numpy as np
import pandas as pd
import mADE

data_small = sio.loadmat('data/input.mat')
data_big = sio.loadmat('data/input_big.mat')
data_big_re = sio.loadmat('data/input_big_re.mat')


def init_big():
	D1 = 56  # 工序总数
	D2 = 5  # 工件种类数
	D3 = 40  # 机器总数
	Np = 300  # 种群大小
	CR = 0.8  # 交叉概率
	Gm = 101  # 最大代数
	ade = mADE.ADE(CR, Np, Gm, D1, D2, D3, data_big, name='big')
	return ade

def init_small():
	D1 = 15  # 工序总数
	D2 = 6  # 工件种类数
	D3 = 13  # 机器总数
	Np = 300  # 种群大小
	CR = 0.8  # 交叉概率
	Gm = 101  # 最大代数
	ade = mADE.ADE(CR, Np, Gm, D1, D2, D3, data_small, name='small')
	return ade

def init_big_resedule():
	D1 = 56
	RD1 = 93
	D2 = 5
	RD2 = 8
	D3 = 40
	RT = 1000 #重调度时间
	Np = 300
	CR = 0.8
	Gm = 101
	ade = mADE.ADE(CR, Np, Gm, D1, D2, D3, data_big_re, name='big_reshedule',
				   	RD1=RD1, RD2=RD2, RT=RT)
	return ade

def main(name):
	if name == 'small':
		ade = init_small()
	elif name == 'big':
		ade = init_big()
	elif name == 'big_reshedule':
		ade = init_big_resedule()
	else:
		print('error')
		return

	B, X, Y= ade.run()
	U = np.array([[1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]])
	U = U.T
	A = np.concatenate([U, 1-U], axis=1).T
	C = np.matmul(B[:, [1, 2]], A)
	for i in range(len(U)):
		c = list(C[:,i])
		b = min(c)
		best = c.index(b)
		print('第%d种权重方案最优解：' %i)
		print('排序方案：')
		print(X[best, :])
		print('分配方案：')
		print(Y[best, :])
		print('完工时间：%s minutes' %B[best, 0])
		print('等待时间：%s minutes' %B[best, 1])
		print('运输时间：%s minutes' %B[best, 2])

		ade.reset(reshedule=False)
		result = ade.gantt_chart(X[best,:], Y[best,:], i)
		if name == 'big_reshedule':
			pf = pd.DataFrame(result)
			pf = pf.where(pf <= ade.RT).dropna()
			processed_result = np.array(pf)
			pf.columns = ['工件号', '工序号', '加工批量', '机器编号', '起始时间',
						  '结束时间', '加工时间', '排队时间', '运输时间']
			print('已完成工序:\n', pf)
			print('开始进行重调度：')
			ade.reset(reshedule=True, processed_result=processed_result)
			rB, rX, rY = ade.run()

			rC = np.matmul(rB[:, [1, 2]], A)
			#for j in range(len(U)):
			rc = list(rC[:, i])
			rb = min(rc)
			rbest = rc.index(rb)
			ade.gantt_chart(rX[rbest, :],rY[rbest, :], i, i)
	ade.plot_show()


if __name__ == "__main__":
	#main(name='small')
	#main(name='big')
	main(name='big_reshedule')


