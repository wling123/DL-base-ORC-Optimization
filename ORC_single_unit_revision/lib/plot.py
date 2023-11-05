import matplotlib
import matplotlib.pyplot as plt
import numpy as np 

def plot_result(data1, data2,mse, header,filename):
    fig, ax1 = plt.subplots(1,3, figsize=(20,3))
    ax1[0].plot(data1[:,0]/1e3, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    ax1[0].plot(data2[:,0]/1e3, color='tab:red', marker='o', markersize=0.1, label='Proxy', alpha=0.7)          
    ax1[0].legend()
    ax1[0].set_title("Power production", fontsize =12)
    ax1[0].set_ylabel('Power [MW]')
#     plt.show()
    ax1[1].plot(data1[:,1]/1e3, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    ax1[1].plot(data2[:,1]/1e3, color='tab:red', marker='o', markersize=0.1, label='Proxy', alpha=0.7)          
    ax1[1].legend()
    ax1[1].set_title("Pump consumption", fontsize =12)
    ax1[1].set_ylabel('Power [MW]')

    ax1[2].plot(data1[:,0]/1e3-data1[:,1]/1e3, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    ax1[2].plot(data2[:,0]/1e3-data2[:,1]/1e3, color='tab:red', marker='o', markersize=0.1, label='Optimized', alpha=0.7)          
    ax1[2].legend()
    ax1[2].set_title("Net", fontsize =12)
    ax1[2].set_ylabel('Power [MW]')
    fig.suptitle('RMSE ='+ str(mse), y=1.1)

    fig.savefig(filename+'.pdf', bbox_inches='tight')


def plot_result2(data1, data2, header, filename):

    fig, ax1 = plt.subplots(1,3, figsize=(20,4))
    ax1[0].plot(data1[:,0]/1e3, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    ax1[0].plot(data2[:,0]/1e3, color='tab:red', marker='o', markersize=0.1, label='Optimized', alpha=0.7)          
    ax1[0].legend()
    ax1[0].set_title("Power production", fontsize =12)
    ax1[0].set_ylabel('Power [MW]')
#     plt.show()
    ax1[1].plot(data1[:,1]/1e3, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    ax1[1].plot(data2[:,1]/1e3, color='tab:red', marker='o', markersize=0.1, label='Optimized', alpha=0.7)          
    ax1[1].legend()
    ax1[1].set_title("Pump consumption", fontsize =12)
    ax1[1].set_ylabel('Power [MW]')
    
    ax1[2].plot(data1[:,0]/1e3-data1[:,1]/1e3, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    ax1[2].plot(data2[:,0]/1e3-data2[:,1]/1e3, color='tab:red', marker='o', markersize=0.1, label='Optimized', alpha=0.7)          
    ax1[2].legend()
    ax1[2].set_title("Net", fontsize =12)
    ax1[2].set_ylabel('Power [MW]')

    temp1 = ((np.mean(data2[:,0])/np.mean(data1[:,0]))-1)*100
    temp2 = ((np.mean(data2[:,1])/np.mean(data1[:,1]))-1)*100
    temp3 = ((np.mean(data2[:,0]-data2[:,1])/np.mean(data1[:,0]-data1[:,1]))-1)*100
    temp1 = str(round(temp1,2))
    temp2 = str(round(temp2,2))
    temp3 = str(round(temp3,2))
    temp4 = str(np.mean(data2[:,0]-data2[:,1])-np.mean(data1[:,0]-data1[:,1]))
    fig.suptitle('Gross power increase '+ temp1+ '%\n'+
                 'Power consumption increase '+ temp2+'%\n'+
                 'Net power increase '+temp3+'%\n'+
                 'Net power increase '+temp4+'KW'
                 , y=1.2)
    fig.savefig(filename+'.pdf', bbox_inches='tight')


def plot_result3(data1, data2, header, filename):

    fig, ax1 = plt.subplots(1,3, figsize=(20,4))
    ax1[0].plot(data1[:,0]/1e3, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    ax1[0].plot(data2[:,0]/1e3, color='tab:red', marker='o', markersize=0.1, label='Optimized', alpha=0.7)          
    ax1[0].legend()
    ax1[0].set_title("Power production", fontsize =12)
    ax1[0].set_ylabel('Power [MW]')
#     plt.show()
    ax1[1].plot(data1[:,1]/1e3, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    ax1[1].plot(data2[:,1]/1e3, color='tab:red', marker='o', markersize=0.1, label='Optimized', alpha=0.7)          
    ax1[1].legend()
    ax1[1].set_title("Pump consumption", fontsize =12)
    ax1[1].set_ylabel('Power [MW]')
    temp = -data1[:,0]+data2[:,0]+data1[:,1]-data2[:,1]
    temp2 = np.add.accumulate(temp)/1e3
    ax1[2].plot(temp2, color='tab:blue', marker='o', markersize=0.1, label='True' , alpha=0.7)
    # ax1[2].plot(data2[:,0]/1e3-data2[:,1]/1e3, color='tab:red', marker='o', markersize=0.1, label='Optimized', alpha=0.7)          
    # ax1[2].legend()
    ax1[2].set_title("Accumulative net energy improvement", fontsize =12)
    ax1[2].set_ylabel('[MWh]')

    temp1 = ((np.mean(data2[:,0])/np.mean(data1[:,0]))-1)*100
    temp2 = ((np.mean(data2[:,1])/np.mean(data1[:,1]))-1)*100
    temp3 = ((np.mean(data2[:,0]-data2[:,1])/np.mean(data1[:,0]-data1[:,1]))-1)*100
    temp1 = str(round(temp1,2))
    temp2 = str(round(temp2,2))
    temp3 = str(round(temp3,2))
    temp4 = str(np.mean(data2[:,0]-data2[:,1])-np.mean(data1[:,0]-data1[:,1]))
    fig.suptitle('Gross power increase '+ temp1+ '%\n'+
                 'Power consumption increase '+ temp2+'%\n'+
                 'Net power increase '+temp3+'%\n'+
                 'Net power increase '+temp4+'KW'
                 , y=1.2)
    fig.savefig(filename+'.pdf', bbox_inches='tight')