
import numpy as np
import csv
from pathlib import Path
import simcentralconnect
import pandas as pd

from scipy.optimize import minimize,Bounds,SR1, rosen, shgo,dual_annealing
from scipy import optimize
from scipy.optimize import rosen, differential_evolution
import scipy

## unit transfer section 

def gpm2ms(data):
    return data/15850.323
def f2k(data):
    return (data + 459.67)* 5/9
def psig2pa(data):
    return(6894.75728*data)

def rpm2mf(data):
    return(400*data/1200)

def amb2turb(temp):
    coef = np.array([ 3.10466479e-1, -1.56390009e+02,  1.99749026e+04])
    predict = np.poly1d(coef)
    y_pred = predict(temp)
    y_pred[y_pred<0.42e3] = 0.42e3
    return y_pred

def pumpcurve(speed):
    coef1 = np.array([1.15621599e+00, 2.34267876e+03, 6.52512148e+05])
    predict = np.poly1d(coef1)
    pred = predict(speed)
    pred_vf = 0.34*speed/1200
    return pred, pred_vf

def loadData(file):
    ## load a datafile for simulation
    data = np.genfromtxt(file, delimiter=',')[1:,:]
    with open(file, 'r') as infile:
        reader = csv.DictReader(infile)
        header= reader.fieldnames
    header = [key for key in header]

    inputs = [2,4,6,15,23]
    input_header = [header[i] for i in inputs]
    input_header

    x = data[:,inputs]

    inputs_names = ['brine_P','brine_T','brine_flow',"pump_speed",'Turbine_outlet_P']
    
    pumprelated = np.array(pumpcurve(x[:,3])).T
    inputs_x = np.hstack([x[:,:4],amb2turb(x[:,4]).reshape([-1,1])])
    return inputs_x, header,data


def writeOutput(result, filename):
    result = np.array(result)
    headernew= ["gross","brine_in_p","brine_out_p","brine_in_t","brine_out_t",
               "brine_out_q","brine_mid_t","pre_in_t","vap_in_t","vap_out_t",
                "turbine_in_p","turbine_out_p","pump_cost","net","superheat","pumpspeed"]
    print(result.shape)
    print(len(headernew))
    output2 = pd.DataFrame(result,columns=headernew)
    output2.to_csv(filename, index=False)




def runSim(inputs_x,Avevaname,optimization = False):
    def open_simulation(sm, simName):
        opened = False
        try:
            opened = sm.OpenSimulation(simName).Result
        except System.AggregateException as ex:
            if not isinstance(ex.InnerException, System.InvalidOperationException) \
                or "simulation doesn't exists" not in ex.InnerException.Message:
                raise
        return opened

    # Connect to SimCentral
    SimNameScriptingEx = Avevaname    
    # Get access to various managers used in this script
    sc = simcentralconnect.connect().Result
    sm = sc.GetService("ISimulationManager")
    # mm = sc.GetService("IModelManager")
    # cm = sc.GetService("IConnectorManager")
    # dm = sc.GetService("IDiagramManager")
    vm = sc.GetService("IVariableManager")
    osm = sc.GetService("IOptimizationSetManager")
    optimizationName = "Pumpn"

    # Example of how to set options.  API logging is ON by default.  Disabled by changing to 'false'
    # sc.SetOptions(repr({'Timeout': 100000, 'EnableApiLogging': 'true'}))

    # Attempt to open the example simulation (it may exist from a previous run)
    opened = open_simulation(sm, SimNameScriptingEx);
    result = []
    for i in range(len(inputs_x)):
        print(i)
        temp_input = inputs_x[i,:]
        ## Set value

        vm.SetVariableValue(SimNameScriptingEx, 'bout' + ".Q", temp_input[2], "m3/s").Result
        vm.SetVariableValue(SimNameScriptingEx, 'bin' + ".P", temp_input[0], "kPa").Result
        vm.SetVariableValue(SimNameScriptingEx, 'bin' + ".T", temp_input[1], "K").Result
        vm.SetVariableValue(SimNameScriptingEx, 'Pumpspeed', temp_input[3]).Result
        vm.SetVariableValue(SimNameScriptingEx, 'Turbine' + ".P2", temp_input[4], "kPa").Result
        ## run optimization
        
        if optimization == True:
            try:
                runResult = osm.RunOptimization(SimNameScriptingEx, optimizationName,5000).Result
            except:
                vm.SetVariableValue(SimNameScriptingEx, 'Pumpspeed', temp_input[3]).Result

            else:
                print('solved')
        #Load result

        gross_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".Power").Result
        pumpcost = vm.GetVariableValue(SimNameScriptingEx, 'Pump' + ".Power").Result
        brinein_p = vm.GetVariableValue(SimNameScriptingEx, 'bin' + ".P").Result
        brineout_p = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".P").Result
        brinein_t = vm.GetVariableValue(SimNameScriptingEx, 'bin' + ".T").Result
        brineout_t = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".T").Result
        brineout_q = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".Q").Result
        brine_v_out_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tto").Result
        R_pre_in_t = vm.GetVariableValue(SimNameScriptingEx, 'Preheater' + ".Tsi").Result
        R_vap_in_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tsi").Result
        R_vap_out_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tso").Result
        R_tur_in_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".P1").Result
        R_tur_out_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".P2").Result
        net = vm.GetVariableValue(SimNameScriptingEx, "Net").Result
        superheat = vm.GetVariableValue(SimNameScriptingEx, "ST").Result
        pumpspeed = vm.GetVariableValue(SimNameScriptingEx, "Pumpspeed").Result
        # pumpspeed = vm.GetVariableValue(SimNameScriptingEx, "Pumpspeed").Result
        
        out = [gross_p, brinein_p,brineout_p,brinein_t,brineout_t,brineout_q,
               brine_v_out_t,R_pre_in_t,R_vap_in_t, R_vap_out_t,R_tur_in_p,R_tur_out_p,pumpcost, net, superheat, pumpspeed]
        result.append(out)

    sm.CloseOpenSimulations().Result

    result = np.array(result)
    return result

class MyBounds:
    def __init__(self, xmax=[1200], xmin=[850] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def runSimandopt(inputs_x,Avevaname):
    def open_simulation(sm, simName):
        opened = False
        try:
            opened = sm.OpenSimulation(simName).Result
        except System.AggregateException as ex:
            if not isinstance(ex.InnerException, System.InvalidOperationException) \
                or "simulation doesn't exists" not in ex.InnerException.Message:
                raise
        return opened

    # Connect to SimCentral
    SimNameScriptingEx = Avevaname    
    # Get access to various managers used in this script
    sc = simcentralconnect.connect().Result
    sm = sc.GetService("ISimulationManager")
    vm = sc.GetService("IVariableManager")
    osm = sc.GetService("IOptimizationSetManager")
    optimizationName = "Pumpn"

    # Example of how to set options.  API logging is ON by default.  Disabled by changing to 'false'
    # sc.SetOptions(repr({'Timeout': 100000, 'EnableApiLogging': 'true'}))
    # 
    bounds = [(800,1200)]
#    bounds = [[800,1200]]
#    cons = []
#    for factor in range(len(bounds)):
#        lower, upper = bounds[factor]
#        l = {'type': 'ineq',
#             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
#        u = {'type': 'ineq',
#             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
#        cons.append(l)
#        cons.append(u)
    
    # Attempt to open the example simulation (it may exist from a previous run)
    # 
    

    def callback(x):
        hisp.append(objective(x))
        history.append(x)

    opened = open_simulation(sm, SimNameScriptingEx);
    result = []
    for i in range(len(inputs_x)):
        print(i)
        temp_input = inputs_x[i,:]
        ## Set value

        vm.SetVariableValue(SimNameScriptingEx, 'bout' + ".Q", temp_input[2], "m3/s").Result
        vm.SetVariableValue(SimNameScriptingEx, 'bin' + ".P", temp_input[0], "kPa").Result
        vm.SetVariableValue(SimNameScriptingEx, 'bin' + ".T", temp_input[1], "K").Result
        vm.SetVariableValue(SimNameScriptingEx, 'Pumpspeed', temp_input[3]).Result
        vm.SetVariableValue(SimNameScriptingEx, 'Turbine' + ".P2", temp_input[4], "kPa").Result
        ## run optimization
        def objective(xxx):
            vm.SetVariableValue(SimNameScriptingEx, 'Pumpspeed', xxx[0]).Result
            net = vm.GetVariableValue(SimNameScriptingEx, "Net").Result
            return -np.array(net) 
        
        x0 = [temp_input[3]]
#        res = minimize(objective, x0, constraints=cons,method='COBYLA')
        history = [x0]
        hisp = [objective(x0)]
        res = minimize(objective, x0,bounds=bounds, callback = callback,options={'ftol': 1e-6,'eps':1e-8})


        optpump = res.x
#        print(optpump)
        vm.SetVariableValue(SimNameScriptingEx, 'Pumpspeed', optpump[0]).Result            
        #Load result

        gross_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".Power").Result
        pumpcost = vm.GetVariableValue(SimNameScriptingEx, 'Pump' + ".Power").Result
        brinein_p = vm.GetVariableValue(SimNameScriptingEx, 'bin' + ".P").Result
        brineout_p = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".P").Result
        brinein_t = vm.GetVariableValue(SimNameScriptingEx, 'bin' + ".T").Result
        brineout_t = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".T").Result
        brineout_q = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".Q").Result
        brine_v_out_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tto").Result
        R_pre_in_t = vm.GetVariableValue(SimNameScriptingEx, 'Preheater' + ".Tsi").Result
        R_vap_in_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tsi").Result
        R_vap_out_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tso").Result
        R_tur_in_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".P1").Result
        R_tur_out_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".P2").Result
        net = vm.GetVariableValue(SimNameScriptingEx, "Net").Result
        superheat = vm.GetVariableValue(SimNameScriptingEx, "ST").Result
        pumpspeed = vm.GetVariableValue(SimNameScriptingEx, "Pumpspeed").Result
        # pumpspeed = vm.GetVariableValue(SimNameScriptingEx, "Pumpspeed").Result
        
        out = [gross_p, brinein_p,brineout_p,brinein_t,brineout_t,brineout_q,
               brine_v_out_t,R_pre_in_t,R_vap_in_t, R_vap_out_t,R_tur_in_p,R_tur_out_p,pumpcost, net, superheat, pumpspeed]
        result.append(out)

    sm.CloseOpenSimulations().Result

    result = np.array(result)
    return result,history,hisp 


def runSimandopt2(inputs_x,Avevaname):
    def open_simulation(sm, simName):
        opened = False
        try:
            opened = sm.OpenSimulation(simName).Result
        except System.AggregateException as ex:
            if not isinstance(ex.InnerException, System.InvalidOperationException) \
                or "simulation doesn't exists" not in ex.InnerException.Message:
                raise
        return opened

    # Connect to SimCentral
    SimNameScriptingEx = Avevaname    
    # Get access to various managers used in this script
    sc = simcentralconnect.connect().Result
    sm = sc.GetService("ISimulationManager")
    vm = sc.GetService("IVariableManager")
    osm = sc.GetService("IOptimizationSetManager")

#    bounds = [(800,1200)]
    bounds = [[800,1200]]
    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds[factor]
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
    
    # Attempt to open the example simulation (it may exist from a previous run)
    opened = open_simulation(sm, SimNameScriptingEx);
    result = []
    for i in range(len(inputs_x)):
        print(i)
        temp_input = inputs_x[i,:]
        ## Set value
        vm.SetVariableValue(SimNameScriptingEx, 'bout' + ".Q", temp_input[2], "m3/s").Result
        vm.SetVariableValue(SimNameScriptingEx, 'bin' + ".P", temp_input[0], "kPa").Result
        vm.SetVariableValue(SimNameScriptingEx, 'bin' + ".T", temp_input[1], "K").Result
        vm.SetVariableValue(SimNameScriptingEx, 'Pumpspeed', temp_input[3]).Result
        vm.SetVariableValue(SimNameScriptingEx, 'Turbine' + ".P2", temp_input[4], "kPa").Result
        ## run optimization
        def objective(xxx):
            vm.SetVariableValue(SimNameScriptingEx, 'Pumpspeed', xxx[0]).Result
            net = vm.GetVariableValue(SimNameScriptingEx, "Net").Result
            return -np.array(net) 
        
        x0 = [temp_input[3]]
        res = minimize(objective, x0, constraints=cons,method='COBYLA')

        optpump = res.x
        vm.SetVariableValue(SimNameScriptingEx, 'Pumpspeed', optpump[0]).Result            
        #Load result

        gross_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".Power").Result
        pumpcost = vm.GetVariableValue(SimNameScriptingEx, 'Pump' + ".Power").Result
        brinein_p = vm.GetVariableValue(SimNameScriptingEx, 'bin' + ".P").Result
        brineout_p = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".P").Result
        brinein_t = vm.GetVariableValue(SimNameScriptingEx, 'bin' + ".T").Result
        brineout_t = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".T").Result
        brineout_q = vm.GetVariableValue(SimNameScriptingEx, 'bout' + ".Q").Result
        brine_v_out_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tto").Result
        R_pre_in_t = vm.GetVariableValue(SimNameScriptingEx, 'Preheater' + ".Tsi").Result
        R_vap_in_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tsi").Result
        R_vap_out_t = vm.GetVariableValue(SimNameScriptingEx, 'Vaporizer' + ".Tso").Result
        R_tur_in_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".P1").Result
        R_tur_out_p = vm.GetVariableValue(SimNameScriptingEx, 'Turbine' + ".P2").Result
        net = vm.GetVariableValue(SimNameScriptingEx, "Net").Result
        superheat = vm.GetVariableValue(SimNameScriptingEx, "ST").Result
        pumpspeed = vm.GetVariableValue(SimNameScriptingEx, "Pumpspeed").Result
        # pumpspeed = vm.GetVariableValue(SimNameScriptingEx, "Pumpspeed").Result
        
        out = [gross_p, brinein_p,brineout_p,brinein_t,brineout_t,brineout_q,
               brine_v_out_t,R_pre_in_t,R_vap_in_t, R_vap_out_t,R_tur_in_p,R_tur_out_p,pumpcost, net, superheat, pumpspeed]
        result.append(out)

    sm.CloseOpenSimulations().Result

    result = np.array(result)
    return result