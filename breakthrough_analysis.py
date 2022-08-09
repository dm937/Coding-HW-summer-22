#Script built to analyse the data from the breakthrough set-up in the custom built rig in the RCCS labs, Heriot-Watt, Edinburgh


#Importing any packages or functions we need for the code
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d
#Defining the breakthrough analysis function including all the inputs
def analyse_breakthrough(MS_csv, flow_csv, T_exp, Low_conc_CO2=False, P_sat_H2O_bath=0, ms_start=0, flow_start=0, breakthrough_start=0, breakthrough_end=999999, filter_window=15, filelabel='test', smoothing_start={'CO2':0, 'H2O':0, 'N2':0, 'He':0}, water_outliers=False, water_outlier_value=2.5E-5, spike_reduction=False, spike_reduction_parameter=1.1, extra_normalisation=False, relative_sensitivities='Default', backgrounds='Default', full_output=False, zero_backgrounds={'CO2':True, 'H2O':True, 'N2':True, 'O2':True, 'He':False}, plot_smoothing=False, initial_sweep=0):
    MS_header_row = 26 #This is the row in the MS raw data file where the headers exist
    P_atm = 1.01325E5 #Atmopsheric pressure [Pa]
    P_exp = P_atm #Experimental pressure [Pa]
    R = 8.314 #J/mol/K (universal gas constant)
    if backgrounds == 'Default':
        background = {'N2':3.78E-9, 'He':1.44E-8, 'CO2':4.16E-11, 'O2':1.86E-10, 'H2O':3.50E-11} #Background MS values to subtract from MS data [torr]
    else:
        background = backgrounds

    Mw =  {'N2':28E-3, 'He':4E-3, 'CO2':44E-3, 'O2':32E-3, 'H2O':18E-3} #Molecular weights of each component [kg/mol]
    max_outlet = 93.59 * 1.66667e-8 #Conversion of the max outlet flow rate in the coriolis to SI units mg/min to kg/s
    max_N2 = 1.66667e-8*100*P_atm/(R*273.15) #Max flow rate in the N2 mass flow meter mol/s
    max_He = 1.66667e-8*141*P_atm/(R*273.15) #Max flow rate in the He mass flow meter mol/s
    coriolis_cal = {'N2':1.06, 'He':1.09, 'CO2':1.09, 'O2':1, 'H2O':1.02} #Coriolis calibration factors
    #Defining relative sensitivities according to whether we select the default values or our own values
    #Also defining whether the experiment is using low concs or higher concs of CO2, whih effects relative sensitivities and the CO2 mass flow rate
    if relative_sensitivities == 'Default':
        if Low_conc_CO2 == True:
            max_CO2 = 1.66667e-8*0.5*P_atm/(R*273.15) #Max flow rate in the CO2 mass flow meter when its set to fluid 2 mol/s 
            RS = {'N2':1, 'He':1.92, 'CO2':0.021, 'O2':0.98, 'H2O':0.0157} #Defining mass spec relative sensitivities [-]
        
        else:
            max_CO2 = 1.66667e-8*2*P_atm/(R*273.15) #Max flow rate in the CO2 mass flow meter when its set to fluid 1 mol/s 
            RS = {'N2':1, 'He':1.92, 'CO2':0.86, 'O2':0.98, 'H2O':0.0157} #Defining mass spec relative sensitivities [-]
    else: #This is if we have selected manual values of relative sensitivities
        if Low_conc_CO2 == True:
            max_CO2 = 1.66667e-8*0.5*P_atm/(R*273.15) #mol/s
            RS = relative_sensitivities #Defining mass spec relative sensitivities [-]
        
        else:
            max_CO2 = 1.66667e-8*2*P_atm/(R*273.15) #mol/s
            RS = relative_sensitivities #Defining mass spec relative sensitivities [-]
    
    H2O_ratio = P_sat_H2O_bath/P_atm #this is calculation that we use to calculate the water inlet flow rate.

    df_MS = pd.read_csv(MS_csv, header=(MS_header_row-1)).drop(['Time', 'Unnamed: 7'], axis=1) #reading the MS csv as a dataframe
    df_MS.loc[:,'Time [s]'] = pd.Series([i/1000 + ms_start for i in df_MS['ms']] , index=df_MS.index) #Adding the MS start time to the times in our dataframe

    df_FM = pd.read_csv(flow_csv, sep=';', names = ['Time', 'CO2', 'Time 2', 'He', 'Time 3', 'N2', 'Time 4', 'Outlet']).drop(['Time 2', 'Time 3', 'Time 4'], axis=1) #reading the flow meter csv as a dataframe
    df_FM.loc[:,'Time [s]'] = pd.Series([i + flow_start for i in df_FM['Time']] , index=df_FM.index) #Adding the flow meter start time to the time in this data
    #This line below is the all important step. We merge the two dataframes based on the times defined above, then we make sure its ordered in time order, and also renaming columns
    df_all = pd.merge(df_MS, df_FM, on='Time [s]',how='outer', sort=True).drop(['Time', 'ms'], axis=1).rename(columns={"Nitrogen": "N2 pressure [torr]", "Water": "H2O pressure [torr]", "Carbon dioxide": "CO2 pressure [torr]", "Oxygen": "O2 pressure [torr]", "Helium": "He pressure [torr]", "CO2": "CO2 flow [%]", "He": "He flow [%]", "N2": "N2 flow [%]", "Outlet": "Outlet flow [%]"}) 

    #In this for loop we are interpolating between each time point that exists for the MS data, to get the flows from the coriolis at that time
        #In this for loop we are interpolating between each time point that exists for the MS data, to get the flows from the coriolis at that time
    for comp in ['CO2', 'N2', 'He', 'Outlet']:
        label = comp + ' flow [%]'
        interpolated_flow = []
        interpolated_flow.append(float("NaN"))
        for i in range(1,len(df_all['Time [s]'])-1):
            if m.isnan(df_all[label][i]) == True:
                time_range = df_all['Time [s]'][i+1] - df_all['Time [s]'][i-1]
                time_int = df_all['Time [s]'][i] - df_all['Time [s]'][i-1]
                time_frac = time_int/time_range
                flow_before = df_all[label][i-1]
                flow_range = df_all[label][i+1] - df_all[label][i-1]
                flow_value = flow_before + flow_range*time_frac
                interpolated_flow.append(flow_value)
            else:
                interpolated_flow.append(df_all[label][i])
        interpolated_flow.append(float("NaN"))
        df_all.loc[:,'Interpolated ' + label] = pd.Series(interpolated_flow, index=df_all.index)
    #Now converting the data time to the time of the breakthrough step
    df_all.loc[:,'Breakthrough time [s]'] = pd.Series([i - breakthrough_start for i in df_all['Time [s]']], index=df_all.index)
    #Now deleting rows without MS values
    df_breakthrough_start = df_all.loc[(abs(df_all['CO2 pressure [torr]']) > 0)]

    #Now we are only taking the part of the dataframe that we are interested in (ignoring drying, cooling, purging etc.)
    df_not_breakthrough = df_breakthrough_start.loc[((df_breakthrough_start['Time [s]'] < breakthrough_start))]
    df_breakthrough = df_breakthrough_start.loc[((df_breakthrough_start['Time [s]'] > breakthrough_start) & (df_breakthrough_start['Time [s]'] < breakthrough_end))]
    df_breakthrough.reset_index(drop=True, inplace=True)
    #Fetching the last values before breakthrough is started
    n2_p = df_not_breakthrough['N2 pressure [torr]'].iloc[-1]
    h2o_p = df_not_breakthrough['H2O pressure [torr]'].iloc[-1]
    co2_p = df_not_breakthrough['CO2 pressure [torr]'].iloc[-1]
    o2_p = df_not_breakthrough['O2 pressure [torr]'].iloc[-1]
    he_p = df_not_breakthrough['He pressure [torr]'].iloc[-1]
    time0 = breakthrough_start
    co2_f = df_not_breakthrough['Interpolated CO2 flow [%]'].iloc[-1]
    n2_f = df_not_breakthrough['Interpolated N2 flow [%]'].iloc[-1]
    he_f = df_not_breakthrough['Interpolated He flow [%]'].iloc[-1]
    out_f = df_not_breakthrough['Interpolated Outlet flow [%]'].iloc[-1]
    b_time = 0

    #Taking initial CO2 value as background 
    if zero_backgrounds['CO2'] == True:
        background['CO2'] = co2_p
    if zero_backgrounds['N2']== True:
        background['N2'] = n2_p
    if zero_backgrounds['H2O']== True:
        background['H2O'] = h2o_p
    if zero_backgrounds['O2']== True:
        background['O2'] = o2_p
    if zero_backgrounds['He']== True:
        background['He'] = he_p
    #Inserting these values as 0 breakthrough time
    df_breakthrough.loc[-1] = [n2_p, h2o_p, co2_p, o2_p, he_p, time0, float("NaN"), float("NaN"), float("NaN"), float("NaN"), co2_f, n2_f, he_f, out_f, b_time]
    df_breakthrough.sort_values('Breakthrough time [s]', inplace=True)
    df_breakthrough.reset_index(drop=True, inplace=True)
    df_breakthrough.drop(['CO2 flow [%]', 'He flow [%]', 'N2 flow [%]', 'Outlet flow [%]'], axis=1, inplace=True)
    


    df_breakthrough.reset_index(drop=True, inplace=True)
    #Here in this for loop we are calculating the corrected MS pressures (subtracting background and overlap, and also correcting for relative sensitivity)
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' pressure [torr]'
        new_label = 'Corrected ' + label
        mole_frac_label = comp + 'mole fraction [-]'
        if comp == 'N2':
            corrected_list = []
            for i in range(len(df_breakthrough['Breakthrough time [s]'])): 
                if ((df_breakthrough[label][i] - background[comp] - 0.114*df_breakthrough['Corrected CO2 pressure [torr]'][i])/RS[comp]) > 0:
                    corrected_list.append((df_breakthrough[label][i] - background[comp] - 0.114*df_breakthrough['Corrected CO2 pressure [torr]'][i])/RS[comp])
                else:
                    corrected_list.append(0)   
            df_breakthrough.loc[:,new_label] = pd.Series(corrected_list, index=df_breakthrough.index)
        else:
            df_breakthrough.loc[:,new_label] = pd.Series([(i - background[comp])/RS[comp] if (i - background[comp])/RS[comp] > 0 else 0 for i in df_breakthrough[label]], index=df_breakthrough.index)
    #Calculating mole fractions in the mass spectrometer in the below for loop
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = 'Corrected ' +comp + ' pressure [torr]'
        new_label = 'True MS ' +comp + ' mole fraction [-]'
        molar_list = []
        for i in range(len(df_breakthrough['Breakthrough time [s]'])): 
            total = df_breakthrough['Corrected CO2 pressure [torr]'][i] + df_breakthrough['Corrected N2 pressure [torr]'][i] + df_breakthrough['Corrected He pressure [torr]'][i] + df_breakthrough['Corrected H2O pressure [torr]'][i] + df_breakthrough['Corrected O2 pressure [torr]'][i]
            molar_list.append(df_breakthrough[label][i]/total)
        df_breakthrough.loc[:,new_label] = pd.Series(molar_list, index=df_breakthrough.index)
    #Calculating an average molecular weight from the calculated MS mole fractions
    df_breakthrough.loc[:,'Fake Outlet average molecular weight [kg/mol]'] = pd.Series([df_breakthrough['True MS CO2 mole fraction [-]'][i]*Mw['CO2']+df_breakthrough['True MS N2 mole fraction [-]'][i]*Mw['N2']+ df_breakthrough['True MS He mole fraction [-]'][i]*Mw['He']+df_breakthrough['True MS O2 mole fraction [-]'][i]*Mw['O2']+df_breakthrough['True MS H2O mole fraction [-]'][i]*Mw['H2O'] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    
    #Now using the "average molecular weight" (this is kind of fake) and the MS mole fractions, and the coriolis data we can calculate the mass flow for each component 
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = 'True MS ' +comp + ' mole fraction [-]'
        new_label = 'True ' +comp + ' mass flow [kg/s]'
        df_breakthrough.loc[:,new_label] = pd.Series([(max_outlet * df_breakthrough['Interpolated Outlet flow [%]'][i]/100) * df_breakthrough[label][i] * Mw[comp] * coriolis_cal[comp]/df_breakthrough['Fake Outlet average molecular weight [kg/mol]'][i] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)  
    
    
    #Now converting the mass flow to molar flow values
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = 'True '+comp + ' mass flow [kg/s]'
        new_label = 'True ' + comp + ' molar flow [mol/s]'
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] / Mw[comp] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    #Now calculating the molar flow rate of the helium through the bypass from the helium mass flow meter
    df_breakthrough.loc[:,'He bypass flow [mol/s]'] = pd.Series([df_breakthrough['Interpolated He flow [%]'][i] * max_He/100 for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    #Now calculating the molar flow in the outlet below. We calculate the helium flow (there is helium initially in the reactor left over from the drying step), from the helium flow in the MS minus the helium flow from the bypass
    for comp in ['He', 'CO2', 'N2', 'H2O', 'O2']:
        label = 'True '+comp + ' molar flow [mol/s]'
        new_label =  comp + ' molar flow [mol/s]'
        if comp != 'He':
            if water_outliers==True and comp == 'H2O':
                df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] if df_breakthrough[label][i] < water_outlier_value else df_breakthrough[label][i-1] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)  
            else:
                df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i]  for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)    
        else:
            df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] - df_breakthrough['He bypass flow [mol/s]'][i] if (df_breakthrough[label][i] - df_breakthrough['He bypass flow [mol/s]'][i]) > 0 else 0  for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)    #print('done corrected molar flows corrections')
    #Summing these molar flows to a total molar flow
    df_breakthrough.loc[:,'Total molar flow [mol/s]'] = pd.Series([df_breakthrough['He molar flow [mol/s]'][i] + df_breakthrough['N2 molar flow [mol/s]'][i] + df_breakthrough['CO2 molar flow [mol/s]'][i] + df_breakthrough['H2O molar flow [mol/s]'][i] + df_breakthrough['O2 molar flow [mol/s]'][i] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    if initial_sweep > 0:
        df_breakthrough.loc[0:initial_sweep,'CO2 molar flow [mol/s]'] = 0
    #Calculating mole fractions from these molar flows
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' molar flow [mol/s]'
        new_label =  comp + ' mole fraction [-]'
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] / df_breakthrough['Total molar flow [mol/s]'][i]  for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)    
    #Calculating concentrations from these mole fractions 
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' mole fraction [-]'
        new_label = comp + ' concentration [mol/m3]'
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] * P_exp/(R*T_exp) for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    #Calculating the inlet flow rates for each component:
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = 'Interpolated '+ comp + ' flow [%]'
        new_label = comp + ' inlet flow [mol/s]'
        if comp != 'CO2' and comp != 'N2' and comp!= 'H2O':
            df_breakthrough.loc[:,new_label] = pd.Series([0*i for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
        elif comp == 'CO2':
            df_breakthrough.loc[:,new_label] = pd.Series([max_CO2*df_breakthrough[label][i]/100 for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
        elif comp == 'N2':
            df_breakthrough.loc[:,new_label] = pd.Series([max_N2*df_breakthrough[label][i]/100 for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
        elif comp == 'H2O':
            df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough['N2 inlet flow [mol/s]'][i]*H2O_ratio/(1-H2O_ratio) for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    #Now calculating inlet mole fractions
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' inlet flow [mol/s]'
        new_label = comp + ' inlet mole fraction [-]'
        mole_frac_list = []
        for i in range(len(df_breakthrough['Breakthrough time [s]'])):
            total = df_breakthrough['CO2 inlet flow [mol/s]'][i] + df_breakthrough['N2 inlet flow [mol/s]'][i] + df_breakthrough['H2O inlet flow [mol/s]'][i] + df_breakthrough['O2 inlet flow [mol/s]'][i] + df_breakthrough['He inlet flow [mol/s]'][i]
            mole_frac_list.append(df_breakthrough[label][i]/total) 
        df_breakthrough.loc[:,new_label] = pd.Series(mole_frac_list, index=df_breakthrough.index)
    #And calculating inlet concentrations
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' inlet mole fraction [-]'
        new_label = comp + ' inlet concentration [mol/m3]'
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] * P_exp/(R*T_exp) for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    #Now normalising mole fractions
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        outlet_label = comp + ' mole fraction [-]'
        inlet_label = comp + ' inlet mole fraction [-]'
        new_label = 'Normalised ' + outlet_label
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[outlet_label][i]/df_breakthrough[inlet_label][i] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    #And normalising concentrations
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        outlet_label = comp + ' concentration [mol/m3]'
        inlet_label = comp + ' inlet concentration [mol/m3]'
        new_label = 'Normalised ' + outlet_label
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[outlet_label][i]/df_breakthrough[inlet_label][i] for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)
    #And normalising molar flow
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        outlet_label = comp + ' molar flow [mol/s]'
        inlet_label = comp + ' inlet flow [mol/s]'
        new_label = 'Normalised ' + outlet_label
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[outlet_label][i]/df_breakthrough[inlet_label][i] for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)
    #Here we get rid of weird spikes maybe caused by condensation
    if spike_reduction == True:
        for i in range(len(df_breakthrough['Breakthrough time [s]'])):
            if (df_breakthrough['Normalised CO2 molar flow [mol/s]'][i] > spike_reduction_parameter) or (df_breakthrough['Normalised H2O molar flow [mol/s]'][i] > spike_reduction_parameter):
                df_breakthrough.drop(i, inplace=True)
    df_breakthrough.reset_index(inplace=True)
    if full_output == False:
    #Building a list of all the outputs we want from the code
        order = ['Breakthrough time [s]']
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet concentration [mol/m3]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' molar flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' concentration [mol/m3]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append('Normalised ' + comp + ' molar flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append('Normalised ' + comp + ' mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append('Normalised ' + comp + ' concentration [mol/m3]')
        df_breakthrough_out = df_breakthrough[order]

        #Finding which row in the dataframe the smoothing should start
        startlist = {}
        index=0
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            for i in range(1,len(df_breakthrough['Breakthrough time [s]'])):
                index = i
                if df_breakthrough['Breakthrough time [s]'][i] > smoothing_start[comp]:
                    break
            startlist[comp] = index
        #Here we smooth the data
        for i in range(1,len(order)):
            label = order[i]
            new_label = 'Smoothed ' + label
            new_list = []
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                if comp in label:
                    component = comp
            filtered_data = uniform_filter1d(pd.Series(df_breakthrough_out[label],index=df_breakthrough_out.index)[startlist[component]:], size=filter_window)
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                if comp in label: 
                    for q in range(startlist[comp]):
                        new_list.append(df_breakthrough_out[label][q])
                        f=q
                    for q in range(len(filtered_data)):
                        new_list.append(filtered_data[q])
                    df_breakthrough_out.loc[:,new_label] = pd.Series(new_list)
        #We can normalise all the values to the final value now for the smoothed data if we like
        if extra_normalisation == True:
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                label = 'Smoothed Normalised ' + comp + ' molar flow [mol/s]'
                series = pd.Series([i/df_breakthrough_out[label].iloc[-1] for i in df_breakthrough_out[label]], index=df_breakthrough_out.index)
            
                df_breakthrough_out.loc[:,'Smoothed renormalised ' + comp + ' molar flow [mol/s]'] = series
        #Saving the output file
        df_breakthrough_out.to_csv(filelabel + '.csv', index=False)
    else:
        order = ['Breakthrough time [s]']
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet concentration [mol/m3]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' molar flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' concentration [mol/m3]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append('Normalised ' + comp + ' molar flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append('Normalised ' + comp + ' mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append('Normalised ' + comp + ' concentration [mol/m3]')
        

        #Finding which row in the dataframe the smoothing should start
        startlist = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            for i in range(1,len(df_breakthrough['Breakthrough time [s]'])):
                index = i
                if df_breakthrough['Breakthrough time [s]'][i] > smoothing_start[comp]:
                    break
            startlist[comp] = index
        #Here we smooth the data
        for i in range(1,len(order)):
            label = order[i]
            new_label = 'Smoothed ' + label
            new_list = []
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                if comp in label:
                    component = comp
            filtered_data = uniform_filter1d(pd.Series(df_breakthrough[label],index=df_breakthrough.index)[startlist[component]:], size=filter_window)
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                if comp in label: 
                    for q in range(startlist[comp]):
                        new_list.append(df_breakthrough[label][q])
                        f=q
                    for q in range(len(filtered_data)):
                        new_list.append(filtered_data[q])
                    df_breakthrough.loc[:,new_label] = pd.Series(new_list)
        #We can normalise all the values to the final value now for the smoothed data if we like
        if extra_normalisation == True:
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                label = 'Smoothed Normalised ' + comp + ' molar flow [mol/s]'
                series = pd.Series([i/df_breakthrough[label].iloc[-1] for i in df_breakthrough[label]], index=df_breakthrough.index)
            
                df_breakthrough.loc[:,'Smoothed renormalised ' + comp + ' molar flow [mol/s]'] = series
        #Saving the output file
        df_breakthrough.to_csv(filelabel + '.csv', index=False)
    if plot_smoothing ==True:
        df_plot = pd.read_csv(filelabel + '.csv')
        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(df_plot['Breakthrough time [s]'], df_plot['Normalised CO2 molar flow [mol/s]'], color='tab:orange', s=10, alpha=0.5)
        ax.scatter(df_plot['Breakthrough time [s]'], df_plot['Normalised N2 molar flow [mol/s]'], color='tab:green', s=10, alpha=0.5)
        ax.scatter(df_plot['Breakthrough time [s]'], df_plot['Normalised H2O molar flow [mol/s]'], color='tab:blue', s=10, alpha=0.5)
        ax.plot(df_plot['Breakthrough time [s]'], df_plot['Smoothed Normalised CO2 molar flow [mol/s]'], color='tab:orange', linestyle='-')
        ax.plot(df_plot['Breakthrough time [s]'], df_plot['Smoothed Normalised N2 molar flow [mol/s]'], color='tab:green', linestyle='-')
        ax.plot(df_plot['Breakthrough time [s]'], df_plot['Smoothed Normalised H2O molar flow [mol/s]'], color='tab:blue', linestyle='-')
        ax.legend(['Smoothed CO2', 'Smoothed N2', 'Smoothed H2O'])
        ax.set(xlabel='Time [s]', ylabel=('Normalised molar flow [-]'))
        ax.set_title('Checking smoothing for ' + filelabel)
        plt.show()


#The function below is very similar to the one above but for regeneration step analysis
def analyse_regen(MS_csv, flow_csv, T_regen, Discount_He=True, Low_conc_CO2=False, P_sat_H2O_bath=0, ms_start=0, flow_start=0, regen_start=0, regen_end=999999, filter_window=15, filelabel='test', smoothing_start={'CO2':0, 'H2O':0, 'N2':0, 'He':0}, water_outliers=False, water_outlier_value=2.5E-5, full_output=False, backgrounds='Default', relative_sensitivities='Default'):
    MS_header_row = 26
    P_atm = 1.01325E5 #Pa
    P_exp = P_atm #Pa
    R = 8.314 #J/mol/K
    if backgrounds == 'Default':
        background = {'N2':3.78E-9, 'He':1.44E-8, 'CO2':4.16E-11, 'O2':1.86E-10, 'H2O':3.50E-11} #Background MS values to subtract from MS data [torr]
    else:
        background = backgrounds
    Mw =  {'N2':28E-3, 'He':4E-3, 'CO2':44E-3, 'O2':32E-3, 'H2O':18E-3} #kg/mol
    max_outlet = 93.59 * 1.66667e-8 #mg/min to kg/s
    max_N2 = 1.66667e-8*100*P_atm/(R*273.15) #mol/s
    max_He = 1.66667e-8*141*P_atm/(R*273.15) #mol/s
    coriolis_cal = {'N2':1.06, 'He':1.09, 'CO2':1.09, 'O2':1, 'H2O':1.02} 
    if relative_sensitivities == 'Default':
        if Low_conc_CO2 == True:
            max_CO2 = 1.66667e-8*0.5*P_atm/(R*273.15) #Max flow rate in the CO2 mass flow meter when its set to fluid 2 mol/s 
            RS = {'N2':1, 'He':1.92, 'CO2':0.021, 'O2':0.98, 'H2O':0.0157} #Defining mass spec relative sensitivities [-]
        
        else:
            max_CO2 = 1.66667e-8*2*P_atm/(R*273.15) #Max flow rate in the CO2 mass flow meter when its set to fluid 1 mol/s 
            RS = {'N2':1, 'He':1.92, 'CO2':0.86, 'O2':0.98, 'H2O':0.0157} #Defining mass spec relative sensitivities [-]
    else: #This is if we have selected manual values of relative sensitivities
        if Low_conc_CO2 == True:
            max_CO2 = 1.66667e-8*0.5*P_atm/(R*273.15) #mol/s
            RS = relative_sensitivities #Defining mass spec relative sensitivities [-]
        
        else:
            max_CO2 = 1.66667e-8*2*P_atm/(R*273.15) #mol/s
            RS = relative_sensitivities #Defining mass spec relative sensitivities [-]
    
    H2O_ratio = P_sat_H2O_bath/P_atm

    df_MS = pd.read_csv(MS_csv, header=(MS_header_row-1)).drop(['Time', 'Unnamed: 7'], axis=1)
    df_MS.loc[:,'Time [s]'] = pd.Series([i/1000 + ms_start for i in df_MS['ms']] , index=df_MS.index)

    df_FM = pd.read_csv(flow_csv, sep=';', names = ['Time', 'CO2', 'Time 2', 'He', 'Time 3', 'N2', 'Time 4', 'Outlet']).drop(['Time 2', 'Time 3', 'Time 4'], axis=1)
    df_FM.loc[:,'Time [s]'] = pd.Series([i + flow_start for i in df_FM['Time']] , index=df_FM.index)

    df_all = pd.merge(df_MS, df_FM, on='Time [s]',how='outer', sort=True).drop(['Time', 'ms'], axis=1).rename(columns={"Nitrogen": "N2 pressure [torr]", "Water": "H2O pressure [torr]", "Carbon dioxide": "CO2 pressure [torr]", "Oxygen": "O2 pressure [torr]", "Helium": "He pressure [torr]", "CO2": "CO2 flow [%]", "He": "He flow [%]", "N2": "N2 flow [%]", "Outlet": "Outlet flow [%]"}) 


        #In this for loop we are interpolating between each time point that exists for the MS data, to get the flows from the coriolis at that time
    for comp in ['CO2', 'N2', 'He', 'Outlet']:
        label = comp + ' flow [%]'
        interpolated_flow = []
        interpolated_flow.append(float("NaN"))
        for i in range(1,len(df_all['Time [s]'])-1):
            if m.isnan(df_all[label][i]) == True:
                time_range = df_all['Time [s]'][i+1] - df_all['Time [s]'][i-1]
                time_int = df_all['Time [s]'][i] - df_all['Time [s]'][i-1]
                time_frac = time_int/time_range
                flow_before = df_all[label][i-1]
                flow_range = df_all[label][i+1] - df_all[label][i-1]
                flow_value = flow_before + flow_range*time_frac
                interpolated_flow.append(flow_value)
            else:
                interpolated_flow.append(df_all[label][i])
        interpolated_flow.append(float("NaN"))
        df_all.loc[:,'Interpolated ' + label] = pd.Series(interpolated_flow, index=df_all.index)
    #Now converting the data time to the time of the breakthrough step
    df_all.loc[:,'Breakthrough time [s]'] = pd.Series([i - regen_start for i in df_all['Time [s]']], index=df_all.index)
    #Now deleting rows without MS values
    df_breakthrough_start = df_all.loc[(abs(df_all['CO2 pressure [torr]']) > 0)]

    #Now we are only taking the part of the dataframe that we are interested in (ignoring drying, cooling, purging etc.)
    df_not_breakthrough = df_breakthrough_start.loc[((df_breakthrough_start['Time [s]'] < regen_start))]
    df_breakthrough = df_breakthrough_start.loc[((df_breakthrough_start['Time [s]'] > regen_start) & (df_breakthrough_start['Time [s]'] < regen_end))]
    df_breakthrough.reset_index(drop=True, inplace=True)
    #Fetching the last values before breakthrough is started
    n2_p = df_not_breakthrough['N2 pressure [torr]'].iloc[-1]
    h2o_p = df_not_breakthrough['H2O pressure [torr]'].iloc[-1]
    co2_p = df_not_breakthrough['CO2 pressure [torr]'].iloc[-1]
    o2_p = df_not_breakthrough['O2 pressure [torr]'].iloc[-1]
    he_p = df_not_breakthrough['He pressure [torr]'].iloc[-1]
    time0 = df_not_breakthrough['Time [s]'].iloc[-1]
    co2_f = df_not_breakthrough['Interpolated CO2 flow [%]'].iloc[-1]
    n2_f = df_not_breakthrough['Interpolated N2 flow [%]'].iloc[-1]
    he_f = df_not_breakthrough['Interpolated He flow [%]'].iloc[-1]
    out_f = df_not_breakthrough['Interpolated Outlet flow [%]'].iloc[-1]
    b_time = 0
    #Inserting these values as 0 breakthrough time
    df_breakthrough.loc[-1] = [n2_p, h2o_p, co2_p, o2_p, he_p, time0, float("NaN"), float("NaN"), float("NaN"), float("NaN"), co2_f, n2_f, he_f, out_f, b_time]
    df_breakthrough.sort_values('Breakthrough time [s]', inplace=True)
    df_breakthrough.reset_index(drop=True, inplace=True)
    df_breakthrough.drop(['CO2 flow [%]', 'He flow [%]', 'N2 flow [%]', 'Outlet flow [%]'], axis=1, inplace=True)
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' pressure [torr]'
        new_label = 'Corrected ' + label
        mole_frac_label = comp + 'mole fraction [-]'
        if comp == 'N2':
            corrected_list = []
            for i in range(len(df_breakthrough['Breakthrough time [s]'])): 
                if ((df_breakthrough[label][i] - background[comp] - 0.114*df_breakthrough['Corrected CO2 pressure [torr]'][i])/RS[comp]) > 0:
                    corrected_list.append((df_breakthrough[label][i] - background[comp] - 0.114*df_breakthrough['Corrected CO2 pressure [torr]'][i])/RS[comp])
                else:
                    corrected_list.append(0)   
            df_breakthrough.loc[:,new_label] = pd.Series(corrected_list, index=df_breakthrough.index)
        else:
            df_breakthrough.loc[:,new_label] = pd.Series([(i - background[comp])/RS[comp] if (i - background[comp])/RS[comp] > 0 else 0 for i in df_breakthrough[label]], index=df_breakthrough.index)
    #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
    #    label = 'Corrected ' +comp + ' pressure [torr]'
    #    new_label = comp + ' mole fraction [-]'
    #    molar_list = []
    #    for i in range(len(df_breakthrough['Breakthrough time [s]'])): 
    #        if Discount_He == True:
    #            total = df_breakthrough['Corrected CO2 pressure [torr]'][i] + df_breakthrough['Corrected N2 pressure [torr]'][i] + df_breakthrough['Corrected H2O pressure [torr]'][i] + df_breakthrough['Corrected O2 pressure [torr]'][i]
    #        else:
    #            total = df_breakthrough['Corrected CO2 pressure [torr]'][i] + df_breakthrough['Corrected N2 pressure [torr]'][i] + df_breakthrough['Corrected He pressure [torr]'][i] + df_breakthrough['Corrected H2O pressure [torr]'][i] + df_breakthrough['Corrected O2 pressure [torr]'][i]
    #        molar_list.append(df_breakthrough[label][i]/total)
    #    df_breakthrough.loc[:,new_label] = pd.Series(molar_list, index=df_breakthrough.index)
    #print('done pressure corrections')
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = 'Corrected ' +comp + ' pressure [torr]'
        new_label = 'True MS ' +comp + ' mole fraction [-]'
        molar_list = []
        for i in range(len(df_breakthrough['Breakthrough time [s]'])): 
            total = df_breakthrough['Corrected CO2 pressure [torr]'][i] + df_breakthrough['Corrected N2 pressure [torr]'][i] + df_breakthrough['Corrected He pressure [torr]'][i] + df_breakthrough['Corrected H2O pressure [torr]'][i] + df_breakthrough['Corrected O2 pressure [torr]'][i]
            molar_list.append(df_breakthrough[label][i]/total)
        df_breakthrough.loc[:,new_label] = pd.Series(molar_list, index=df_breakthrough.index)
    #print('ms mole frac corrections')
    df_breakthrough.loc[:,'Fake Outlet average molecular weight [kg/mol]'] = pd.Series([df_breakthrough['True MS CO2 mole fraction [-]'][i]*Mw['CO2']+df_breakthrough['True MS N2 mole fraction [-]'][i]*Mw['N2']+ df_breakthrough['True MS He mole fraction [-]'][i]*Mw['He']+df_breakthrough['True MS O2 mole fraction [-]'][i]*Mw['O2']+df_breakthrough['True MS H2O mole fraction [-]'][i]*Mw['H2O'] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = 'True MS ' +comp + ' mole fraction [-]'
        new_label = 'True ' +comp + ' mass flow [kg/s]'
        df_breakthrough.loc[:,new_label] = pd.Series([(max_outlet * df_breakthrough['Interpolated Outlet flow [%]'][i]/100) * df_breakthrough[label][i] * Mw[comp] * coriolis_cal[comp]/df_breakthrough['Fake Outlet average molecular weight [kg/mol]'][i] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)  
    
    
    #print('done ms mass flows')
    #df_breakthrough.loc[:,'Outlet molar flow [mol/s]'] = pd.Series([(max_outlet*df_breakthrough['Interpolated Outlet flow [%]'][i]/100)/df_breakthrough['Outlet average molecular weight [kg/mol]'][i] for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = 'True '+comp + ' mass flow [kg/s]'
        new_label = 'True ' + comp + ' molar flow [mol/s]'
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] / Mw[comp] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    #print('done ms molar flows corrections')
    df_breakthrough.loc[:,'He bypass flow [mol/s]'] = pd.Series([df_breakthrough['Interpolated He flow [%]'][i] * max_He/100 for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    for comp in ['He', 'CO2', 'N2', 'H2O', 'O2']:
        label = 'True '+comp + ' molar flow [mol/s]'
        new_label =  comp + ' molar flow [mol/s]'
        if comp != 'He':
            if water_outliers==True and comp == 'H2O':
                df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] if df_breakthrough[label][i] < water_outlier_value else df_breakthrough[label][i-1] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)  
            else:
                df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i]  for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)    
        else:
            df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] - df_breakthrough['He bypass flow [mol/s]'][i] if (df_breakthrough[label][i] - df_breakthrough['He bypass flow [mol/s]'][i]) > 0 else 0  for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)

    #print('done corrected molar flows corrections')
    if Discount_He == False:
        df_breakthrough.loc[:,'Total molar flow [mol/s]'] = pd.Series([df_breakthrough['He molar flow [mol/s]'][i] + df_breakthrough['N2 molar flow [mol/s]'][i] + df_breakthrough['CO2 molar flow [mol/s]'][i] + df_breakthrough['H2O molar flow [mol/s]'][i] + df_breakthrough['O2 molar flow [mol/s]'][i] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    else:
        df_breakthrough.loc[:,'Total molar flow [mol/s]'] = pd.Series([df_breakthrough['N2 molar flow [mol/s]'][i] + df_breakthrough['CO2 molar flow [mol/s]'][i] + df_breakthrough['H2O molar flow [mol/s]'][i] + df_breakthrough['O2 molar flow [mol/s]'][i] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
    #print('done total molar flow')
    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' molar flow [mol/s]'
        new_label =  comp + ' mole fraction [-]'
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] / df_breakthrough['Total molar flow [mol/s]'][i]  for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)  

    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' mole fraction [-]'
        new_label = comp + ' concentration [mol/m3]'
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] * P_exp/(R*T_regen) for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)

    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = 'Interpolated '+ comp + ' flow [%]'
        new_label = comp + ' inlet flow [mol/s]'
        if comp != 'CO2' and comp != 'N2' and comp!= 'H2O':
            df_breakthrough.loc[:,new_label] = pd.Series([0*i for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)
        elif comp == 'CO2':
            df_breakthrough.loc[:,new_label] = pd.Series([max_CO2*df_breakthrough[label][i]/100 for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)
        elif comp == 'N2':
            df_breakthrough.loc[:,new_label] = pd.Series([max_N2*df_breakthrough[label][i]/100 for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)
        elif comp == 'H2O':
            df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough['N2 inlet flow [mol/s]'][i]*H2O_ratio/(1-H2O_ratio) for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)

    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' inlet flow [mol/s]'
        new_label = comp + ' inlet mole fraction [-]'
        mole_frac_list = []
        for i in range(len(df_breakthrough['CO2 mole fraction [-]'])):
            total = df_breakthrough['CO2 inlet flow [mol/s]'][i] + df_breakthrough['N2 inlet flow [mol/s]'][i] + df_breakthrough['H2O inlet flow [mol/s]'][i] + df_breakthrough['O2 inlet flow [mol/s]'][i] + df_breakthrough['He inlet flow [mol/s]'][i]
            mole_frac_list.append(df_breakthrough[label][i]/total) 
        df_breakthrough.loc[:,new_label] = pd.Series(mole_frac_list, index=df_breakthrough.index)

    for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        label = comp + ' inlet mole fraction [-]'
        new_label = comp + ' inlet concentration [mol/m3]'
        df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] * P_exp/(R*T_regen) for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)


    
    #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
    #    outlet_label = comp + ' mole fraction [-]'
    #    inlet_label = comp + ' inlet mole fraction [-]'
    #    new_label = 'Normalised ' + outlet_label
    #    df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[outlet_label][i]/df_breakthrough[inlet_label][i] for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)

    #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
    #    outlet_label = comp + ' concentration [mol/m3]'
    #    inlet_label = comp + ' inlet concentration [mol/m3]'
    #    new_label = 'Normalised ' + outlet_label
    #    df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[outlet_label][i]/df_breakthrough[inlet_label][i] for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)
        
    #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
    #    outlet_label = comp + ' molar flow [mol/s]'
    #    inlet_label = comp + ' inlet flow [mol/s]'
    #    new_label = 'Normalised ' + outlet_label
    #    df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[outlet_label][i]/df_breakthrough[inlet_label][i] for i in range(len(df_breakthrough['CO2 mole fraction [-]']))], index=df_breakthrough.index)
    if full_output == False:
        order = ['Breakthrough time [s]']
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet concentration [mol/m3]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' molar flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' concentration [mol/m3]')
        #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        #    order.append('Normalised ' + comp + ' molar flow [mol/s]')
        #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        #    order.append('Normalised ' + comp + ' mole fraction [-]')
        #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        #    order.append('Normalised ' + comp + ' concentration [mol/m3]')
        df_breakthrough_out = df_breakthrough[order]

        startlist = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            for i in range(1,len(df_breakthrough['Breakthrough time [s]'])):
                index = i
                if df_breakthrough['Breakthrough time [s]'][i] > smoothing_start[comp]:
                    break
            startlist[comp] = index

        for i in range(1,len(order)):
            label = order[i]
            new_label = 'Smoothed ' + label
            #if ((m.isinf(df_breakthrough_out[label][5])) == False) & (m.isnan(df_breakthrough_out[label][5]) == False) :
            #df_breakthrough_out.loc[0:index,new_label] = pd.Series(df_breakthrough_out[label][0:index])
            new_list = []
            filtered_data = uniform_filter1d(df_breakthrough_out[label], size=filter_window)
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                if comp in label: 
                    for q in range(startlist[comp]):
                        new_list.append(df_breakthrough_out[label][q])
                        f=q
                    for q in range(f,len(df_breakthrough_out[label])):
                        new_list.append(filtered_data[q])
                    df_breakthrough_out.loc[:,new_label] = pd.Series(new_list)
            
            
        df_breakthrough_out.to_csv(filelabel + '.csv', index=False)
    else:
        order = ['Breakthrough time [s]']
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' inlet concentration [mol/m3]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' molar flow [mol/s]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' mole fraction [-]')
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            order.append(comp + ' concentration [mol/m3]')
        #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        #    order.append('Normalised ' + comp + ' molar flow [mol/s]')
        #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        #    order.append('Normalised ' + comp + ' mole fraction [-]')
        #for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
        #    order.append('Normalised ' + comp + ' concentration [mol/m3]')
  

        startlist = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            for i in range(1,len(df_breakthrough['Breakthrough time [s]'])):
                index = i
                if df_breakthrough['Breakthrough time [s]'][i] > smoothing_start[comp]:
                    break
            startlist[comp] = index

        for i in range(1,len(order)):
            label = order[i]
            new_label = 'Smoothed ' + label
            #if ((m.isinf(df_breakthrough_out[label][5])) == False) & (m.isnan(df_breakthrough_out[label][5]) == False) :
            #df_breakthrough_out.loc[0:index,new_label] = pd.Series(df_breakthrough_out[label][0:index])
            new_list = []
            filtered_data = uniform_filter1d(df_breakthrough[label], size=filter_window)
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                if comp in label: 
                    for q in range(startlist[comp]):
                        new_list.append(df_breakthrough[label][q])
                        f=q
                    for q in range(f,len(df_breakthrough[label])):
                        new_list.append(filtered_data[q])
                    df_breakthrough.loc[:,new_label] = pd.Series(new_list)
            
            
        df_breakthrough.to_csv(filelabel + '.csv', index=False)


#Now our loading calculation
def loading(processed_data, bed_length, bed_diameter, bed_mass, T_exp, mode='breakthrough', bed_porosity=0.4, regen_data='None', integration_end=1E8):

    if mode == 'breakthrough':
        P_atm = 1.01325E5 #[Pa]
        P_exp = P_atm #[Pa]        
        R = 8.314 
        pi = 3.14159
        bed_area = (pi*bed_diameter**2)/4
        bed_volume = bed_area*bed_length

        #Calculating the bed density. Don't actually need this put left it in
        density = bed_mass/bed_volume


        pellet_density = density/(1-bed_porosity) 
        #Reading the processed data and converting it to a dataframe
        df_start = pd.read_csv(processed_data)
        #Deleting all the data after the integration end time
        df = df_start.loc[((df_start['Breakthrough time [s]'] < integration_end)) ]
        #Calculating volumetric flow, but realise we no longer need this
        df.loc[:,'Volumetric flow in [m3/s]'] = pd.Series([(df['CO2 inlet flow [mol/s]'][i]+df['N2 inlet flow [mol/s]'][i]+df['H2O inlet flow [mol/s]'][i])*R*T_exp/P_exp for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)
        df.loc[:,'Volumetric flow out [m3/s]'] = pd.Series([(df['CO2 molar flow [mol/s]'][i]+df['N2 molar flow [mol/s]'][i]+df['H2O molar flow [mol/s]'][i])*R*T_exp/P_exp for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)
        #Calculating inlet concentrations, but again we don't need to do this
        c_avgin = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' inlet concentration [mol/m3]'
            c_avgin[comp] = df[label].mean()
        #Calculating the average institial inlet velocity, again we don't need this
        v_avgin = df['Volumetric flow in [m3/s]'].mean()/(bed_area*bed_porosity)
        #Now calculating the term inside the integral for each time step
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            label = 'Smoothed Normalised ' + comp + ' molar flow [mol/s]'
            inlet_label = comp + ' inlet mole fraction [-]'
            new_label = comp + ' integral term'
            df.loc[:,new_label] = pd.Series([1- df[label][i]/df[label].iloc[-1] for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)
        #Doing the trapezium rule to calculate the area above the curve for each time step
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            label = comp + ' integral term'
            new_label = comp + ' area integral'
            df.loc[:,new_label] = pd.Series([(df['Breakthrough time [s]'][i]-df['Breakthrough time [s]'][i-1])*0.5*(df[label][i]+df[label][i-1]) if i > 0 else 0 for i in range(len(df['CO2 inlet flow [mol/s]']))] , index=df.index)
        #Summing the area
        sum_area = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            label = comp + ' area integral'
            sum_area[comp] = df[label].sum()
        #Now calculating the volume based loading of the bed
        loading_volume_based = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            loading_volume_based[comp] = bed_porosity*c_avgin[comp]*(sum_area[comp]*v_avgin/bed_length - 1)/(1-bed_porosity)
        #And now converting the mass based loading of the bed
        loading_mass_based = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            loading_mass_based[comp] = loading_volume_based[comp]/pellet_density 
    elif mode == 'regeneration':
        P_atm = 1.01325E5 #[Pa]
        P_exp = P_atm #[Pa]
        R = 8.314 
        pi = 3.14159
        bed_area = (pi*bed_diameter**2)/4
        bed_volume = bed_area*bed_length


        density = bed_mass/bed_volume


        pellet_density = density/(1-bed_porosity) 

        df = pd.read_csv(processed_data)
        df_regen_start = pd.read_csv(regen_data)
        df_regen = df_regen_start.loc[((df_regen_start['Breakthrough time [s]'] < integration_end)) ]

        df.loc[:,'Volumetric flow in [m3/s]'] = pd.Series([(df['CO2 inlet flow [mol/s]'][i]+df['N2 inlet flow [mol/s]'][i]+df['H2O inlet flow [mol/s]'][i])*R*T_exp/P_exp for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)
        df_regen.loc[:,'Volumetric flow out [m3/s]'] = pd.Series([(df_regen['CO2 molar flow [mol/s]'][i]+df_regen['N2 molar flow [mol/s]'][i]+df_regen['H2O molar flow [mol/s]'][i])*R*T_exp/P_exp for i in range(len(df_regen['CO2 inlet flow [mol/s]']))], index=df_regen.index)

        c_avgin = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' inlet concentration [mol/m3]'
            c_avgin[comp] = df[label].mean()

        v_avgin = df['Volumetric flow in [m3/s]'].mean()/(bed_area*bed_porosity)

        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' mole fraction [-]'
            inlet_label = comp + ' inlet mole fraction [-]'
            new_label = comp + ' integral term'
            df_regen.loc[:,new_label] = pd.Series([1 - df_regen[label][i]*df_regen['Volumetric flow out [m3/s]'][i]/(df[inlet_label][i]*df['Volumetric flow in [m3/s]'].mean()) for i in range(len(df_regen['CO2 inlet flow [mol/s]']))], index=df_regen.index)
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' integral term'
            new_label = comp + ' area integral'
            df_regen.loc[:,new_label] = pd.Series([(df_regen['Breakthrough time [s]'][i]-df_regen['Breakthrough time [s]'][i-1])*0.5*(df_regen[label][i]+df_regen[label][i-1]) if i > 0 else 0 for i in range(len(df_regen['CO2 inlet flow [mol/s]']))] , index=df_regen.index)
        sum_area = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' area integral'
            sum_area[comp] = df_regen[label].sum()

        loading_volume_based = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            loading_volume_based[comp] = bed_porosity*c_avgin[comp]*(sum_area[comp]*v_avgin/bed_length - 1)/(1-bed_porosity)

        loading_mass_based = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            loading_mass_based[comp] = loading_volume_based[comp]/pellet_density 
    return loading_mass_based, loading_volume_based #Outputting the loading values
#Now defining our plotting function including all the inputs
def quickplot(data, data_labels, components, variable, filelabel, Normalised=True, Smoothed=False, Save=True):
    dfs=[]
    #Making a list of our dataframes
    for exp in data:
        dfs.append(pd.read_csv(exp))
    #Setting the labels of the columns that we will plot
    if Normalised == True:
        n_label = 'Normalised '
        n2_label = 'Normalised '
    else:
        n_label = ''
        n2_label =''
    if Smoothed == True:
        s_label = 'Smoothed '
    else:
        s_label = ''


    #List of all the linestyles that exist
    styles = ['solid', 'dotted', 'dashed', 'dashdot']
    labels = []
    #Setting the labels of the columns that we will plot
    for comp in components:
        labels.append(s_label + n_label  + comp + ' ' + variable)
    #Defining our plot
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(8,5))
    lines = []
    legends = []
    for exp in range(len(data)): 
        for line in labels:
            if 'CO2' in line:
                lines.append(ax.plot(dfs[exp]['Breakthrough time [s]'], dfs[exp][line], linestyle=styles[exp], color='tab:orange'))
                legends.append(str(data_labels[exp] + ' CO$_2$'))
            elif 'N2' in line:
                lines.append(ax.plot(dfs[exp]['Breakthrough time [s]'], dfs[exp][line], linestyle=styles[exp], color='tab:green'))
                legends.append(str(data_labels[exp] + ' N$_2$'))
            elif 'H2O' in line:
                lines.append(ax.plot(dfs[exp]['Breakthrough time [s]'], dfs[exp][line], linestyle=styles[exp], color='tab:blue'))
                legends.append(str(data_labels[exp] + ' H$_2$O'))
            elif 'He' in line:
                lines.append(ax.plot(dfs[exp]['Breakthrough time [s]'], dfs[exp][line], linestyle=styles[exp], color='tab:red'))
                legends.append(str(data_labels[exp] + ' He'))
            elif 'O2' in line:
                lines.append(ax.plot(dfs[exp]['Breakthrough time [s]'], dfs[exp][line], linestyle=styles[exp], color='tab:purple'))
                legends.append(str(data_labels[exp] + ' O$_2$'))
    #Adding a legend and labels
    ax.legend(legends)
    if Normalised == True:
        if variable == 'mole fraction [-]':
            ax.set(xlabel='Time [s]', ylabel=('Normalised mole fraction [-]'))
        elif variable == 'molar flow [mol/s]':
            ax.set(xlabel='Time [s]', ylabel=('Normalised molar flow [-]'))
        elif variable == 'concentration [mol/m3]':
            ax.set(xlabel='Time [s]', ylabel=('Normalised concentration [-]'))
    elif Normalised == False:
        if variable == 'mole fraction [-]':
            ax.set(xlabel='Time [s]', ylabel=('Mole fraction [-]'))
        elif variable == 'molar flow [mol/s]':
            ax.set(xlabel='Time [s]', ylabel=('Molar flow [mol/s]'))
        elif variable == 'concentration [mol/m3]':
            ax.set(xlabel='Time [s]', ylabel=('Concentration [mol/m3]'))


    #Saving our figure
    if Save == True:
        fig.savefig(filelabel + '.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    return fig, ax


def loadingcustom(processed_data, bed_length, bed_diameter, bed_mass, T_exp, columnlabel, component, mode='breakthrough', bed_porosity=0.4, regen_data='None', integration_end=1E8):

    if mode == 'breakthrough':
        P_atm = 1.01325E5 #[Pa]
        P_exp = P_atm #[Pa]        
        R = 8.314 
        pi = 3.14159
        bed_area = (pi*bed_diameter**2)/4
        bed_volume = bed_area*bed_length


        density = bed_mass/bed_volume


        pellet_density = density/(1-bed_porosity) 

        df_start = pd.read_csv(processed_data)
        df = df_start.loc[((df_start['Breakthrough time [s]'] < integration_end)) ]

        df.loc[:,'Volumetric flow in [m3/s]'] = pd.Series([(df['CO2 inlet flow [mol/s]'][i]+df['N2 inlet flow [mol/s]'][i]+df['H2O inlet flow [mol/s]'][i])*R*T_exp/P_exp for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)
        df.loc[:,'Volumetric flow out [m3/s]'] = pd.Series([(df['CO2 molar flow [mol/s]'][i]+df['N2 molar flow [mol/s]'][i]+df['H2O molar flow [mol/s]'][i])*R*T_exp/P_exp for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)

        c_avgin = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' inlet concentration [mol/m3]'
            c_avgin[comp] = df[label].mean()

        v_avgin = df['Volumetric flow in [m3/s]'].mean()/(bed_area*bed_porosity)

        df.loc[:,'Integral term'] = pd.Series([1 - df[columnlabel][i] for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)
        
        df.loc[:,'Area integral'] = pd.Series([(df['Breakthrough time [s]'][i]-df['Breakthrough time [s]'][i-1])*0.5*(df['Integral term'][i]+df['Integral term'][i-1]) if i > 0 else 0 for i in range(len(df['CO2 inlet flow [mol/s]']))] , index=df.index)
        sum_area = df['Area integral'].sum()

     
        loading_volume_based = bed_porosity*c_avgin[component]*(sum_area*v_avgin/bed_length - 1)/(1-bed_porosity)


        loading_mass_based = loading_volume_based/pellet_density 
    elif mode == 'regeneration':
        P_atm = 1.01325E5 #[Pa]
        P_exp = P_atm #[Pa]
        R = 8.314 
        pi = 3.14159
        bed_area = (pi*bed_diameter**2)/4
        bed_volume = bed_area*bed_length


        density = bed_mass/bed_volume


        pellet_density = density/(1-bed_porosity) 

        df = pd.read_csv(processed_data)
        df_regen_start = pd.read_csv(regen_data)
        df_regen = df_regen_start.loc[((df_regen_start['Breakthrough time [s]'] < integration_end)) ]

        df.loc[:,'Volumetric flow in [m3/s]'] = pd.Series([(df['CO2 inlet flow [mol/s]'][i]+df['N2 inlet flow [mol/s]'][i]+df['H2O inlet flow [mol/s]'][i])*R*T_exp/P_exp for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)
        df_regen.loc[:,'Volumetric flow out [m3/s]'] = pd.Series([(df_regen['CO2 molar flow [mol/s]'][i]+df_regen['N2 molar flow [mol/s]'][i]+df_regen['H2O molar flow [mol/s]'][i])*R*T_exp/P_exp for i in range(len(df_regen['CO2 inlet flow [mol/s]']))], index=df_regen.index)

        c_avgin = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' inlet concentration [mol/m3]'
            c_avgin[comp] = df[label].mean()

        v_avgin = df['Volumetric flow in [m3/s]'].mean()/(bed_area*bed_porosity)

        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' mole fraction [-]'
            inlet_label = comp + ' inlet mole fraction [-]'
            new_label = comp + ' integral term'
            df_regen.loc[:,new_label] = pd.Series([1 - df_regen[label][i]*df_regen['Volumetric flow out [m3/s]'][i]/(df[inlet_label][i]*df['Volumetric flow in [m3/s]'].mean()) for i in range(len(df_regen['CO2 inlet flow [mol/s]']))], index=df_regen.index)
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' integral term'
            new_label = comp + ' area integral'
            df_regen.loc[:,new_label] = pd.Series([(df_regen['Breakthrough time [s]'][i]-df_regen['Breakthrough time [s]'][i-1])*0.5*(df_regen[label][i]+df_regen[label][i-1]) if i > 0 else 0 for i in range(len(df_regen['CO2 inlet flow [mol/s]']))] , index=df_regen.index)
        sum_area = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' area integral'
            sum_area[comp] = df_regen[label].sum()

        loading_volume_based = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            loading_volume_based[comp] = bed_porosity*c_avgin[comp]*(sum_area[comp]*v_avgin/bed_length - 1)/(1-bed_porosity)

        loading_mass_based = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            loading_mass_based[comp] = loading_volume_based[comp]/pellet_density 
    return loading_mass_based, loading_volume_based



def area(processed_data, integration_end=1E8, column_end =' molar flow [mol/s]', comps = ['CO2']):


    df_start = pd.read_csv(processed_data)
    df = df_start.loc[((df_start['Breakthrough time [s]'] < integration_end)) ]





    for comp in comps:
        label = 'Smoothed Normalised ' + comp + column_end
        inlet_label = comp + ' inlet mole fraction [-]'
        new_label = comp + ' integral term'
        df.loc[:,new_label] = pd.Series([1 - df[label][i]/df[label].iloc[-1] for i in range(len(df['CO2 inlet flow [mol/s]']))], index=df.index)
    for comp in comps:
        label = comp + ' integral term'
        new_label = comp + ' area integral'
        df.loc[:,new_label] = pd.Series([(df['Breakthrough time [s]'][i]-df['Breakthrough time [s]'][i-1])*0.5*(df[label][i]+df[label][i-1]) if i > 0 else 0 for i in range(len(df['CO2 inlet flow [mol/s]']))] , index=df.index)
    sum_area = {}
    for comp in comps:
        label = comp + ' area integral'
        sum_area[comp] = df[label].sum()
    return sum_area




#Defining the raw data function including all the inputs
def raw_data(MS_csv, flow_csv, T_exp, Low_conc_CO2=False, P_sat_H2O_bath=0, ms_start=0, flow_start=0, breakthrough_start=0, breakthrough_end=999999, filter_window=15, filelabel='test', smoothing_start={'CO2':0, 'H2O':0, 'N2':0, 'He':0}, water_outliers=False, water_outlier_value=2.5E-5, spike_reduction=False, spike_reduction_parameter=1.1, extra_normalisation=False, relative_sensitivities='Default'):
    MS_header_row = 26 #This is the row in the MS raw data file where the headers exist
    P_atm = 1.01325E5 #Atmopsheric pressure [Pa]
    P_exp = P_atm #Experimental pressure [Pa]
    R = 8.314 #J/mol/K (universal gas constant)
    background = {'N2':3.78E-9, 'He':1.44E-8, 'CO2':4.16E-11, 'O2':1.86E-10, 'H2O':3.50E-11} #Background MS values to subtract from MS data [torr]


    Mw =  {'N2':28E-3, 'He':4E-3, 'CO2':44E-3, 'O2':32E-3, 'H2O':18E-3} #Molecular weights of each component [kg/mol]
    max_outlet = 93.59 * 1.66667e-8 #Conversion of the max outlet flow rate in the coriolis to SI units mg/min to kg/s
    max_N2 = 1.66667e-8*100*P_atm/(R*273.15) #Max flow rate in the N2 mass flow meter mol/s
    max_He = 1.66667e-8*141*P_atm/(R*273.15) #Max flow rate in the He mass flow meter mol/s
    coriolis_cal = {'N2':1.06, 'He':1.09, 'CO2':1.09, 'O2':1, 'H2O':1.02} #Coriolis calibration factors
    #Defining relative sensitivities according to whether we select the default values or our own values
    #Also defining whether the experiment is using low concs or higher concs of CO2, whih effects relative sensitivities and the CO2 mass flow rate
    if relative_sensitivities == 'Default':
        if Low_conc_CO2 == True:
            max_CO2 = 1.66667e-8*0.5*P_atm/(R*273.15) #Max flow rate in the CO2 mass flow meter when its set to fluid 2 mol/s 
            RS = {'N2':1, 'He':1.92, 'CO2':0.021, 'O2':0.98, 'H2O':0.0157} #Defining mass spec relative sensitivities [-]
        
        else:
            max_CO2 = 1.66667e-8*2*P_atm/(R*273.15) #Max flow rate in the CO2 mass flow meter when its set to fluid 1 mol/s 
            RS = {'N2':1, 'He':1.92, 'CO2':0.86, 'O2':0.98, 'H2O':0.0157} #Defining mass spec relative sensitivities [-]
    else: #This is if we have selected manual values of relative sensitivities
        if Low_conc_CO2 == True:
            max_CO2 = 1.66667e-8*0.5*P_atm/(R*273.15) #mol/s
            RS = relative_sensitivities #Defining mass spec relative sensitivities [-]
        
        else:
            max_CO2 = 1.66667e-8*2*P_atm/(R*273.15) #mol/s
            RS = relative_sensitivities #Defining mass spec relative sensitivities [-]
    
    H2O_ratio = P_sat_H2O_bath/P_atm #this is calculation that we use to calculate the water inlet flow rate.

    df_MS = pd.read_csv(MS_csv, header=(MS_header_row-1)).drop(['Time', 'Unnamed: 7'], axis=1) #reading the MS csv as a dataframe
    df_MS.loc[:,'Time [s]'] = pd.Series([i/1000 + ms_start for i in df_MS['ms']] , index=df_MS.index) #Adding the MS start time to the times in our dataframe

    df_FM = pd.read_csv(flow_csv, sep=';', names = ['Time', 'CO2', 'Time 2', 'He', 'Time 3', 'N2', 'Time 4', 'Outlet']).drop(['Time 2', 'Time 3', 'Time 4'], axis=1) #reading the flow meter csv as a dataframe
    df_FM.loc[:,'Time [s]'] = pd.Series([i + flow_start for i in df_FM['Time']] , index=df_FM.index) #Adding the flow meter start time to the time in this data
    #This line below is the all important step. We merge the two dataframes based on the times defined above, then we make sure its ordered in time order, and also renaming columns
    df_all = pd.merge(df_MS, df_FM, on='Time [s]',how='outer', sort=True).drop(['Time', 'ms'], axis=1).rename(columns={"Nitrogen": "N2 pressure [torr]", "Water": "H2O pressure [torr]", "Carbon dioxide": "CO2 pressure [torr]", "Oxygen": "O2 pressure [torr]", "Helium": "He pressure [torr]", "CO2": "CO2 flow [%]", "He": "He flow [%]", "N2": "N2 flow [%]", "Outlet": "Outlet flow [%]"}) 
    df_all.reset_index(drop=True, inplace=True)
        #In this for loop we are interpolating between each time point that exists for the MS data, to get the flows from the coriolis at that time
    for comp in ['CO2', 'N2', 'He', 'Outlet']:
        label = comp + ' flow [%]'
        interpolated_flow = []
        interpolated_flow.append(float("NaN"))
        for i in range(1,len(df_all['Time [s]'])-1):
            if m.isnan(df_all[label][i]) == True:
                time_range = df_all['Time [s]'][i+1] - df_all['Time [s]'][i-1]
                time_int = df_all['Time [s]'][i] - df_all['Time [s]'][i-1]
                time_frac = time_int/time_range
                flow_before = df_all[label][i-1]
                flow_range = df_all[label][i+1] - df_all[label][i-1]
                flow_value = flow_before + flow_range*time_frac
                interpolated_flow.append(flow_value)
            else:
                interpolated_flow.append(df_all[label][i])
        interpolated_flow.append(float("NaN"))
        df_all.loc[:,'Interpolated ' + label] = pd.Series(interpolated_flow, index=df_all.index)
    #Now converting the data time to the time of the breakthrough step
    df_all.loc[:,'Breakthrough time [s]'] = pd.Series([i - breakthrough_start for i in df_all['Time [s]']], index=df_all.index)
    #Now deleting rows without MS values
    df_breakthrough_start = df_all.loc[(abs(df_all['CO2 pressure [torr]']) > 0)]
    #df_breakthrough_start = df_all
    #Now we are only taking the part of the dataframe that we are interested in (ignoring drying, cooling, purging etc.)
    df_not_breakthrough = df_breakthrough_start.loc[((df_breakthrough_start['Time [s]'] < breakthrough_start))]
    df_breakthrough = df_breakthrough_start.loc[((df_breakthrough_start['Time [s]'] > breakthrough_start) & (df_breakthrough_start['Time [s]'] < breakthrough_end))]
    df_breakthrough.reset_index(drop=True, inplace=True)
    #Fetching the last values before breakthrough is started
    n2_p = df_not_breakthrough['N2 pressure [torr]'].iloc[-1]
    h2o_p = df_not_breakthrough['H2O pressure [torr]'].iloc[-1]
    co2_p = df_not_breakthrough['CO2 pressure [torr]'].iloc[-1]
    o2_p = df_not_breakthrough['O2 pressure [torr]'].iloc[-1]
    he_p = df_not_breakthrough['He pressure [torr]'].iloc[-1]
    time0 = breakthrough_start
    co2_f = df_not_breakthrough['Interpolated CO2 flow [%]'].iloc[-1]
    n2_f = df_not_breakthrough['Interpolated N2 flow [%]'].iloc[-1]
    he_f = df_not_breakthrough['Interpolated He flow [%]'].iloc[-1]
    out_f = df_not_breakthrough['Interpolated Outlet flow [%]'].iloc[-1]
    b_time = 0
    #Inserting these values as 0 breakthrough time
    df_breakthrough.loc[-1] = [n2_p, h2o_p, co2_p, o2_p, he_p, time0, float("NaN"), float("NaN"), float("NaN"), float("NaN"), co2_f, n2_f, he_f, out_f, b_time]
    df_breakthrough.sort_values('Breakthrough time [s]', inplace=True)
    df_breakthrough.reset_index(drop=True, inplace=True)
    df_breakthrough.drop(['CO2 flow [%]', 'He flow [%]', 'N2 flow [%]', 'Outlet flow [%]'], axis=1, inplace=True)

    df_breakthrough.to_csv(filelabel + '.csv', index=False)







