import numpy as np 
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def NNloadData(file,scaler):
	data = np.genfromtxt(file, delimiter=',')[1:,:]
	data = data.astype(np.float32)
	with open(file, 'r') as infile:
		reader = csv.DictReader(infile)
		header= reader.fieldnames
	header = [key for key in header]
	header1= header[:5]
	header2= header[5:]
	datan = scaler.transform(data)
	return datan,header1,header2


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		self.fc1 = nn.Linear(5, 20)  # 5*5 from image dimension
		self.fc2 = nn.Linear(20, 40)
		self.fc3 = nn.Linear(40, 2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	
class Net2(nn.Module):

	def __init__(self):
		super(Net2, self).__init__()

		self.fc1 = nn.Linear(5, 20)  # 5*5 from image dimension
		self.fc2 = nn.Linear(20, 40)
		self.fc3 = nn.Linear(40, 2)
		self.dropout = nn.Dropout(0.1)
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		x = self.dropout(x)
		x = self.fc3(x)
		return x

def trainNN(model, train, position, epochs):
	model.train()   # set model in train mode (eg batchnorm params get updated)
	opt = optim.Adam(model.parameters(), lr=0.001)       # create optimizer instance
	criterion = nn.MSELoss()    # create loss layer instance
	train_it = 0
	best_loss = 100
	batch_size = 16
	val_loss = []
	rec_loss = []
	for ep in range(epochs):
		opt = optim.Adam(model.parameters(), lr=0.001)    
		np.random.seed(int(time.time()))
		trainl = np.arange(len(train))
		np.random.shuffle(trainl)
		nval = int(len(train)*0.7)
		traintemp = train[trainl[:nval],:]
		val = train[trainl[nval:],:]
		val = torch.from_numpy(val)
		traintemp = torch.from_numpy(traintemp)
		print("Run Epoch {}".format(ep))
		Trains = torch.utils.data.DataLoader(traintemp,batch_size=batch_size)    
		for Train in Trains:
			train_x = Train[:,:5]
			train_y = Train[:,5:]
			opt.zero_grad()
			outputs = model(train_x)
			total_loss = criterion(outputs, train_y)
			total_loss.backward()
			opt.step()
			if train_it % 1000 == 0:
				print("It {}: Reconstruction Loss: {}".format(train_it, total_loss))
			train_it += 1
		val_output = model(val[:,:5]) 
		val_loss1 = criterion(val_output,val[:,5:])
		val_loss.append(val_loss1)
		train_output = model(traintemp[:,:5])
		total_loss = criterion(train_output, traintemp[:,5:])
		rec_loss.append(total_loss)
		print("It {}: Validation Loss: {}".format(ep, val_loss1))
		if val_loss1 < best_loss:
			torch.save(model.state_dict(), position)
			best_loss = val_loss1
	print("Done!")
	plt.plot(rec_loss,label = 'train_loss')
	plt.plot(val_loss,label = 'val_loss')
	plt.title("Reconstruction Loss")
	plt.legend()
	plt.show()

def predNN(model, test, scaler):
	testx = torch.from_numpy(test[:,:5])
	pred = model(testx).detach().numpy()
	testy = test[:,5:]
	mse = np.sqrt(np.mean((pred- testy)**2,axis =0))
	mse2 = np.sqrt(np.mean((pred[:,[0]]-pred[:,[1]]- testy[:,[0]]+ testy[:,[1]])**2,axis =0))
	mse = mse.tolist()+mse2.tolist()
	mse = [round(i,3) for i in mse]
	testy = scaler.inverse_transform(test)[:,5:]
	pred = scaler.inverse_transform(np.hstack([test[:,:5],pred]))[:,5:]
	return testy, pred, mse

class MPC():
	def __init__(self, inputs, step, model, maxvalue, minvalue, filename):
		self.x = inputs.clone()
		self.step = step
		self.model = model
		self.max = maxvalue
		self.min = minvalue
		self.tmp = self.max-self.min
		self.filename = filename
	def renew(self,i):
		self.iter  = i 
		self.inputs = self.x[i:i+self.step,:]
	def objfun(self,x):
		x = np.tile(x, (self.step,1))
		a = torch.tensor(x)
		a = torch.from_numpy(x).float()
		input_tmp = self.inputs
		input_tmp[:,3] = a[:,0]
		self.model.load_state_dict(torch.load(self.filename))
		self.model.eval() 
		result = self.model(input_tmp)
		result = (result[:,1]*self.tmp[6]+self.min[6])-(result[:,0]*self.tmp[5]+ self.min[5])
		result = sum(result)
		return np.sum(result.detach().numpy())

def runMPC(test, step, model, maxvalue, minvalue, filename):
	bounds = [[0,1]]
	cons = []
	for factor in range(len(bounds)):
		lower, upper = bounds[factor]
		l = {'type': 'ineq', 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
		u = {'type': 'ineq', 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
		cons.append(l)
		cons.append(u)
	testx = torch.from_numpy(np.copy(test[:,:5]))
	mpc = MPC(testx, step, model, maxvalue, minvalue, filename)
	result = []
	lentest = len(testx)
	for i in range(lentest//step):
		mpc.renew(i*step)
		x0 = [testx[i,3].item()]
		res = minimize(mpc.objfun, x0, constraints=cons, method='COBYLA')
		print(i)
		result.extend([res.x[0]]*step)
	return np.array(result)