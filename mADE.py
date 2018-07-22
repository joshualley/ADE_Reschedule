import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

class ADE():
	def __init__(self, CR, Np, Gm, D1, D2, D3, data, name, RD1=None, RD2=None, RT=None):
		self.name = name
		self.processed_result = None
		self.reshedule = False
		self.scale_factor_1 = 0.8
		self.scale_factor_2 = 0.8
		self.T = 90 #起始温度
		self.T_end = 88
		self.temperature_factor = 0.9999 #降温因子
		self.lamda = 0.9
		self.G = 0
		self.CR = CR
		self.Np = Np
		self.Gm = Gm
		self.D1 = D1
		self.RD1 = RD1
		self.D2 = D2
		self.RD2 = RD2
		self.D3 = D3
		self.RT = RT
		self.constrain1 = data['constrain1']
		self.constrain2 = data['constrain2']
		self.machine = data['machine']
		self.quantity = data['quantity']
		self.traveltime = data['traveltime']
		self.workingtime = data['workingtime']
		self.travel_t = np.zeros(Np)
		self.wait_t = np.zeros(Np)
		self.final_t = np.zeros(Np)

		self.X = None
		self.X_next_1 = None
		self.X_next_2 = None
		self.X_next = None

		self.Y = None
		self.Y_next_1 = None
		self.Y_next_2 = None
		self.Y_next = None

	def init_population(self):
		if self.reshedule:
			D1 = self.RD1
			D2 = self.RD2
		else:
			D1 = self.D1
			D2 = self.D2
		self.X = np.random.random((self.Np, D1))
		self.X_next_1 = np.zeros((self.Np, D1))
		self.X_next_2 = np.zeros((self.Np, D1))
		self.X_next = np.zeros((self.Np, D1))

		self.Y = []
		for k in range(self.Np):
			y = []
			for i in range(D2):
				temp = list(np.random.random((1, self.constrain2[i][0])))
				y.append(temp)
			self.Y.append(y)
		self.Y = np.array(self.Y)
		self.Y_next_1 = np.array(self.Y)
		self.Y_next_2 = np.array(self.Y)
		self.Y_next = np.array(self.Y)

	def reset(self, reshedule, processed_result=None):
		self.reshedule = reshedule
		self.processed_result = processed_result
		self.G = 0
		self.T = 90
		self.travel_t = np.zeros(self.Np)
		self.wait_t = np.zeros(self.Np)
		self.final_t = np.zeros(self.Np)

		self.X = None
		self.X_next_1 = None
		self.X_next_2 = None
		self.X_next = None

		self.Y = None
		self.Y_next_1 = None
		self.Y_next_2 = None
		self.Y_next = None

	def variation(self):
		if self.reshedule:
			D1 = self.RD1
			D2 = self.RD2
		else:
			D1 = self.D1
			D2 = self.D2
		operator = np.exp(1-self.Gm / (self.Gm + 1 - self.G))
		F1 = self.scale_factor_1*(2**operator)
		F2 = self.scale_factor_2 * (2 ** operator)
		for i in range(self.Np):
			j, k, p = 0, 0, 0
			while i==j or i==k or i==p or j==k or j==p or k==p:
				j, k, p = np.random.randint(0, self.Np, 3)
			self.X_next_1[i, :] = self.X[p, :] + F1*(self.X[j, :] - self.X[k, :])
		self.X_next_1 = np.array([[np.random.random() if i>1 or i<0 else i for i in row] for row in self.X_next_1])
		#print(self.X_next_1.shape)

		for i in range(self.Np):
			j, k, p = 0, 0, 0
			while i==j or i==k or i==p or j==k or j==p or k==p:
				j, k, p = np.random.randint(0, self.Np, 3)
			for d in range(D2):
				#print(self.Y[i,d,0])
				self.Y_next_1[i, d, 0] = self.Y[p, d, 0] +F2*(self.Y[j, d, 0] - self.Y[k, d, 0])
				self.Y_next_1[i, d, 0] = np.array([np.random.random() if i > 1 or i < 0 else i for i in self.Y_next_1[i, d, 0]])

	def cross(self):
		if self.reshedule:
			D1 = self.RD1
			D2 = self.RD2
		else:
			D1 = self.D1
			D2 = self.D2
		for i in range(self.Np):
			for j in range(D1):
				if np.random.random() < self.CR or np.random.randint(0,D1) != j:
					self.X_next_2[i, j] = self.X_next_1[i, j]
				else:
					self.X_next_2[i, j] = self.X[i, j]

		for i in range(self.Np):
			for j in range(D2):
				if np.random.random() < self.CR or np.random.randint(0,D2) != j:
					self.Y_next_2[i, j, 0] = self.Y_next_1[i, j, 0]
				else:
					self.Y_next_2[i, j, 0] = self.Y[i, j, 0]

	def choose(self):
		for i in range(self.Np):
			ft1, wt1, tt1 = self.fitness(self.X[i, :], self.Y[i, :, 0])
			ft2, wt2, tt2 = self.fitness(self.X_next_2[i, :], self.Y_next_2[i, :, 0])
			fitness1 = self.lamda * ft1 + (1 - self.lamda) * wt1
			fitness2 = self.lamda * ft2 + (1 - self.lamda) * wt2

			if fitness2 < fitness1:
				self.X_next[i, :] = self.X_next_2[i, :]
				self.Y_next[i, :, 0] = self.Y_next_2[i, :, 0]
				self.final_t[i] = ft2
				self.wait_t[i] = wt2
				self.travel_t[i] = tt2
			else:
				p = np.random.random()
				# 模拟退火优化策略
				if p < np.exp((fitness2 - fitness1) / self.T):
					self.X_next[i, :] = self.X_next_2[i, :]
					self.Y_next[i, :, 0] = self.Y_next_2[i, :, 0]
					self.final_t[i] = ft2
					self.wait_t[i] = wt2
					self.travel_t[i] = tt2
				else:
					self.X_next[i, :] = self.X[i, :]
					self.Y_next[i, :, 0] = self.Y[i, :, 0]
					self.final_t[i] = ft1
					self.wait_t[i] = wt1
					self.travel_t[i] = tt1

	def fitness(self, X, Y):
		if self.reshedule:
			D1 = self.RD1
			D2 = self.RD2
			travel_t = np.sum(self.processed_result[:, 7])
			wait_t = np.sum(self.processed_result[:, 6])
			flag_st = np.zeros(D2)
			flag_et = np.zeros(self.D3)
			flag_st[self.D2:self.RD2] = self.RT
		else:
			D1 = self.D1
			D2 = self.D2
			travel_t = 0
			wait_t = 0
			flag_st = np.zeros(D2)  # 某个工件的开工时间
			flag_et = np.zeros(self.D3)  # 某个机器上的完工时间
		X, Y = self.decode(X, Y)
		J = np.zeros(D2)
		pro_res = self.processed_result
		for i in range(D1):
			part_i = int(X[i])
			process_i = int(J[part_i])

			choosed_machine_index = int(Y[part_i][process_i]) - 1
			machine_i = self.machine[part_i, process_i][0, choosed_machine_index] - 1

			craft_time = int(self.workingtime[part_i, process_i][0, choosed_machine_index])
			quantity = int(self.quantity[part_i, 0])
			t1 = craft_time * quantity

			if process_i == self.constrain2[part_i, 0] - 1:
				t2 = 0
			else:
				choosed_machine_index_plus_1 = int(Y[part_i][process_i + 1]) - 1
				# print('part:%d, process:%d,choosed_machine_index_plus_1:%d' % (part_i, process_i + 1, choosed_machine_index_plus_1))
				machine_i_plus_1 = self.machine[part_i, process_i + 1][0, choosed_machine_index_plus_1] - 1
				t2 = int(self.traveltime[machine_i, machine_i_plus_1]) * quantity

			if self.reshedule and X[i]+1 in pro_res[:, 0] \
					and process_i == pro_res[list(pro_res[:, 0]).index(X[i]+1), 1] - 1:
				processed_order = list(pro_res[:, 0]).index(X[i]+1)
				flag_et[machine_i] = pro_res[processed_order, 5]
				flag_st[part_i] = flag_et[machine_i] + t2
				pro_res = np.delete(pro_res, processed_order, axis=0)
			else:
				travel_t += t2
				repair_t = 0
				error_prop = np.random.random()
				if error_prop < 0.01:
					repair_t = 30

				if flag_st[part_i] <= flag_et[machine_i]:
					wait_t += (flag_et[machine_i] - flag_st[part_i] + repair_t)
					flag_et[machine_i] += (t1 + repair_t)
					flag_st[part_i] = flag_et[machine_i] + t2
				else:
					wait_t += repair_t
					flag_et[machine_i] = flag_st[part_i] + t1 + repair_t
					flag_st[part_i] = flag_et[machine_i] + t2

			J[int(X[i])] += 1
		final_t = np.max(flag_st)

		return final_t, wait_t, travel_t

	def decode(self, X_, Y_):
		if self.reshedule:
			D1 = self.RD1
			D2 = self.RD2
		else:
			D1 = self.D1
			D2 = self.D2
		X, Y = X_.copy(), Y_.copy()
		x_index = np.argsort(X)
		A = np.array([[]])
		for i in range(D2):
			#print(self.constrain2[i,0])
			a = i*np.ones((1, self.constrain2[i,0]))
			A = np.concatenate([A, a], axis=1)

		for i in range(D1):
			#print(X)
			#print(A[0,i])
			X[x_index[i]] = A[0,i]
		for i in range(D2):
			Y[i] = np.ceil(Y[i]*self.constrain1[i, 0:self.constrain2[i,0]])
			#print(Y[i])

		return X, Y

	def gantt_chart(self, X, Y, _i_, _j_=None):
		if self.reshedule:
			D1 = self.RD1
			D2 = self.RD2
			travel_t = np.sum(self.processed_result[:, 6])
			wait_t = np.sum(self.processed_result[:, 7])
			flag_st = np.zeros(D2)
			flag_et = np.zeros(self.D3)
			flag_st[self.D2:self.RD2] = self.RT
		else:
			D1 = self.D1
			D2 = self.D2
			travel_t = 0
			wait_t = 0
			flag_st = np.zeros(D2)  # 某个工件的开工时间
			flag_et = np.zeros(self.D3)  # 某个机器上的完工时间
		J = np.zeros(D2)

		format_result = np.zeros([D1, 9])
		machine_name = ['A1','A2','A3','B1','C1','D1','D2','D3','E1','E2',
						'F1','G1','G2','H1','H2','H3','I1','I2','I3','J1',
						'K1','L1','L2','M1','M2','M3','N1','O1','O2','O3',
						'P1','P2','Q1','Q2','Q3','R1','R2','R3','S1','T1']

		pro_res = self.processed_result
		for i in range(D1):
			part_i = int(X[i])
			process_i = int(J[part_i])
			choosed_machine_index = int(Y[part_i][process_i]) - 1
			machine_i = self.machine[part_i, process_i][0, choosed_machine_index] - 1
			craft_time = int(self.workingtime[part_i, process_i][0, choosed_machine_index])
			quantity = int(self.quantity[part_i, 0])
			t1 = craft_time * quantity
			if process_i == self.constrain2[part_i, 0] - 1:
				t2 = 0  # 本次运输时间
			else:
				choosed_machine_index_plus_1 = int(Y[part_i][process_i + 1]) - 1
				# print('part:%d, process:%d,choosed_machine_index_plus_1:%d' % (part_i, process_i + 1, choosed_machine_index_plus_1))
				machine_i_plus_1 = self.machine[part_i, process_i + 1][0, choosed_machine_index_plus_1] - 1
				t2 = int(self.traveltime[machine_i, machine_i_plus_1]) * quantity


			if self.reshedule and X[i]+1 in pro_res[:, 0] \
					and process_i == pro_res[list(pro_res[:, 0]).index(X[i]+1), 1]-1:
				processed_order = list(pro_res[:, 0]).index(X[i]+1)
				flag_et[machine_i] = pro_res[processed_order, 5]
				flag_st[part_i] = flag_et[machine_i] + t2
				format_result[i, 0] = pro_res[processed_order, 0]
				format_result[i, 1] = pro_res[processed_order, 1]
				format_result[i, 2] = pro_res[processed_order, 2]
				format_result[i, 3] = pro_res[processed_order, 3]
				format_result[i, 4] = pro_res[processed_order, 4]
				format_result[i, 5] = pro_res[processed_order, 5]
				format_result[i, 6] = pro_res[processed_order, 6]
				format_result[i, 7] = pro_res[processed_order, 7]
				format_result[i, 8] = pro_res[processed_order, 8]
				pro_res = np.delete(pro_res, processed_order, axis=0)
			else:

				format_result[i, 0] = part_i + 1
				format_result[i, 1] = process_i + 1
				format_result[i, 2] = quantity
				format_result[i, 3] = machine_i + 1

				format_result[i, 8] = t2
				travel_t += t2

				repair_t = 0
				error_prop = np.random.random()
				if error_prop < 0.01:
					print('工件%d在进行第%d道工序前，机器%s发生故障'%(part_i, process_i, machine_name[machine_i]))
					repair_t = 30

				if flag_st[part_i] <= flag_et[machine_i]:
					format_result[i, 4] = flag_et[machine_i]
					wt = flag_et[machine_i] - flag_st[part_i] + repair_t
					format_result[i, 7] = wt
					wait_t += wt
					flag_et[machine_i] += (t1 + repair_t)
					flag_st[part_i] = flag_et[machine_i] + t2
				else:
					format_result[i, 7] = repair_t
					wait_t += repair_t
					flag_st[part_i] += repair_t
					format_result[i, 4] = flag_st[part_i]
					flag_et[machine_i] = flag_st[part_i] + t1
					flag_st[part_i] = flag_et[machine_i] + t2
				format_result[i, 5] = flag_et[machine_i]
				format_result[i, 6] = format_result[i, 5]-format_result[i, 4]
			J[int(X[i])] += 1

		pf = pd.DataFrame(format_result)
		pf.columns = ['工件号', '工序号','加工批量', '机器编号', '起始时间',
					  '结束时间', '加工时间', '排队时间','运输时间']

		import os
		if self.reshedule:
			for k,row in enumerate(self.processed_result[:,[0, 1]]):
				del_pf = pf.where((pf['工件号'] == row[0]) & (pf['工序号'] == row[1])).dropna()
				pf = pf.drop(index=del_pf.index[0])
			print('初调度第%d种权重时，重调度第%d种权重下方案'%(_j_, _i_))
			pf = pd.DataFrame(np.array(pf))
			pf.columns = ['工件号', '工序号', '加工批量', '机器编号', '起始时间',
						  '结束时间', '加工时间', '排队时间', '运输时间']
			print(pf)
			ppf = pd.DataFrame(self.processed_result)
			ppf.columns = ['工件号', '工序号', '加工批量', '机器编号', '起始时间',
						  '结束时间', '加工时间', '排队时间', '运输时间']

			pf = pd.concat([ppf, pf], axis=0)
			format_result = np.array(pf)
			m = [machine_name[int(i) - 1] for i in pf['机器编号']]
			pf['机器编号'] = m
			dir = 'data/' + self.name + '/reshe_data/'
			if not os.path.exists(dir):
				os.makedirs(dir)
			fname = dir + 'reshedule_' + str(_j_) + '_'+str(_i_)+'.csv'
			pf.to_csv(fname, encoding='utf-8')


		else:
			print('第%d种初调度方案：'%_i_)
			print(pf)
			dir = 'data/'+ self.name + '/she_data/'
			if not os.path.exists(dir):
				os.makedirs(dir)
			fname = dir + 'shedule_' + str(_i_)+ '.csv'
			m = [machine_name[int(i) - 1] for i in pf['机器编号']]
			pf['机器编号'] = m
			pf.to_csv(fname, encoding='utf-8')

		colors = []
		for i in range(D2):
			color = (0.3 + 0.5 * np.random.random(),
					 0.3 + 0.5 * np.random.random(),
					 0.3 + 0.5 * np.random.random())
			colors.append(color)
		if self.reshedule:
			f = plt.figure(_i_+21)
		else:
			f = plt.figure(_i_)
		ax = f.add_subplot(111, aspect='equal')
		h = self.D3*2
		interval = 0
		for i in range(D1):
			x, m, w = format_result[i, [4, 3, 6]]
			m = m - 1
			ax.broken_barh([(x, w)], [h * m + interval, h], facecolors=colors[int(format_result[i, 0] - 1)])
			tx = 'J(' + str(int(format_result[i, 0])) + ',' + str(int(format_result[i, 1])) + ')'
			plt.text(x, h * m + interval, tx)


		plt.xlabel('processing time')
		plt.ylabel('Machine')
		ax.set_yticks(range(int(interval + h / 2), (h + interval) * self.D3, (h + interval)))
		ax.set_yticklabels(range(1, self.D3+1))
		if self.reshedule:
			dir = 'pic/reshedule/'
			if not os.path.exists(dir):
				os.makedirs(dir)
			figname = dir + 'Reshedule_'+str(_i_)+str(_j_)+'.png'
			plt.title('Reshedule_'+str(_i_)+str(_j_))
			f.savefig(figname)

		else:
			dir = 'pic/shedule/'
			if not os.path.exists(dir):
				os.makedirs(dir)
			figname = dir + 'Shedule_' + str(_i_) + '.png'
			plt.title('Shedule_'+str(_i_))
		#plt.show()

		return format_result

	def plot_show(self):
		plt.show()

	def run(self):
		self.init_population()
		min_ft_per_gene = []
		min_wt_per_gene = []
		min_tt_per_gene = []
		fits = []

		plt.ion()
		f = plt.figure(100)
		plt.title('Analysis of Astringency')
		plt.show()
		st = time.time()
		while self.G < self.Gm and self.T > self.T_end:
			self.variation()
			self.cross()
			self.choose()
			self.X = self.X_next
			self.Y = self.Y_next
			ft = np.min(self.final_t)
			wt = np.min(self.wait_t)
			tt = np.min(self.travel_t)
			min_ft_per_gene.append(ft)
			min_wt_per_gene.append(wt)
			min_tt_per_gene.append(tt)
			fitness = self.lamda * ft + (1 - self.lamda) * wt
			fits.append(fitness)
			plt.plot(fits,'-', linewidth=0.5)
			#plt.plot(min_ft_per_gene,'-', linewidth=0.5)
			#plt.plot(min_wt_per_gene,'g-', linewidth=0.5)
			#plt.plot(min_tt_per_gene,'r-', linewidth=0.5)
			plt.xlabel('Generation')
			plt.ylabel('Fitness')
			plt.pause(0.0001)
			time_step = 10
			if self.G % time_step==0:
				temp = time.time()-st
				l_m = int(temp*self.Gm-temp*self.G)/60
				l_s = int(temp*self.Gm-temp*self.G)%60


				print('[Name=%s,第%d代,T=%fK] FT:%smin, WT:%smin, FIT:%s, 耗时:%fs, ETA:%dmin%ds'
					  % (self.name, self.G, self.T, ft, wt, fitness, temp * time_step, l_m, l_s))
			st = time.time()
			self.G += 1
			self.T *= self.temperature_factor
		plt.ioff()
		if not self.reshedule:
			import os
			dir = 'pic/'+self.name+'/'
			if not os.path.exists(dir):
				os.makedirs(dir)
			name = dir+'analysis_of_astringency.png'
			f.savefig(name)

		A = np.array([self.final_t, self.wait_t, self.travel_t]).T
		#print('A:\n',A.shape)
		_, i= np.unique(A.view(A.dtype.descr*A.shape[1]), return_index=True)
		B = A[i, :]
		#print("B:\n",B.shape)
		X,Y = [],[]
		for k in i:
			x, y = self.decode(self.X[k, :], self.Y[k, :, 0])
			X.append(x)
			Y.append(y)
		X = np.array(X)
		Y = np.array(Y)

		return B, X, Y














