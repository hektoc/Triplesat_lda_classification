# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:15:26 2020

@author: Fantomas
"""
import os
import numpy as np
import re
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D

# Скрипт для расчета индексов из спектральных срезов обучающей выборки спутникового изображения TripleSat
# с последующим построением зависимостей индексов от индексов. Также выполняется запись рассчитанных индексов в файлы ...IndexLevels.txt
# (спектральные срезы TripleSat должны быть предварительно вручную извлечены из программы Envi и записаны в текстовые файлы, расположенные в папках, соответствующих классам)
# файлы ...IndexLevels.txt затем используются как входные данные обуч. выборки для алгоритма LDA в программе triplesat.py
# для построения карты классификации по всему изображению TripleSat

def MCARI2(spectrum):
    ro1 = spectrum[3]
    ro2 = spectrum[2]
    ro3 = spectrum[1]
    return 1.5 * (2.5 * (ro1 - ro2) - 1.3 * (ro1 - ro3))/ sqrt((2*ro1 + 1)**2 - (6*ro1 - 5*sqrt(ro2)) - 0.5)

def MTVI(spectrum):
    ro1 = spectrum[3]
    ro2 = spectrum[1]
    ro3 = spectrum[2]
    return 1.2 * (1.2 *(ro1 - ro2) - 2.5 * (ro3 - ro2))

def MTVI2(spectrum):
    ro1 = spectrum[3]
    ro2 = spectrum[1]
    ro3 = spectrum[2]
    return (1.5 * (1.2 *(ro1 - ro2) - 2.5 * (ro3 - ro2)))/sqrt((2*ro1 + 1)**2 - (6*ro1 - 5*sqrt(ro2)) - 0.5)

def SIPI(spectrum):
    ro1 = spectrum[3]
    ro2 = spectrum[0]
    ro3 = spectrum[2]
    return (ro1 - ro2)/(ro1 - ro3)

def LV(spectrum):
    ro1 = spectrum[1]
    ro2 = spectrum[2]
    ro3 = spectrum[0]
    return ro1*ro2/ro3**2

def BR(spectrum):
    ro1 = spectrum[0]
    ro2 = spectrum[2]
    return ro1/ro2

def GNDVI2(spectrum):  
    ro1 = spectrum[3]
    ro2 = spectrum[1]
    return (ro1-ro2)/(ro1+ro2)

def DI1(spectrum):
    ro1 = spectrum[3] 
    ro2 = spectrum[1]
    return ro1-ro2

def SIPI2(spectrum):
    ro1 = spectrum[3] 
    ro2 = spectrum[0] 
    ro3 = spectrum[3] 
    ro4 = spectrum[2] 
    return (ro1-ro2)/(ro3+ro4)

def NPCI(spectrum):
    ro1 = spectrum[2] 
    ro2 = spectrum[0] 
    return  (ro1-ro2)/(ro1+ro2)

def BR625(spectrum):
    ro1 = spectrum[2] 
    ro2 = spectrum[3] 
    return ro1/ro2

def PSNDchla(spectrum):
    ro1 = spectrum[3]
    ro2 = spectrum[2]
    return (ro1-ro2)/(ro1+ro2)

def PSSRa(spectrum):
    ro1 = spectrum[3]
    ro2 = spectrum[2]
    return ro1/ro2

def PSSRc(spectrum):
    ro1 = spectrum[3]
    ro2 = spectrum[0]
    return ro1/ro2

def New2(spectrum):
    ro1 = spectrum[0]
    ro2 = spectrum[2]
    return (ro1-ro2)/(ro1+ro2)

def New13(spectrum):
    ro1 = spectrum[3]
    ro2 = spectrum[1]
    return (ro1/ro2)


def read_index(indexdict):
    val = []
    name = []
    for ind in indexdict:
        name.append(str(ind))
        val.append(float(indexdict[ind]))
    return name, val   

def read_envifile(f):
    text = open(f, "r")
    wl = []
    sp = []
    #et = int((filename.split("#")[1]).split(".")[0]) # ФСР
    skip = [0,1,2]
    for i,line in enumerate(text):
        if i in skip: # Пропуск строк с заголовком 
            continue
        l = line.rstrip() # Удаление символа новой строки \n в конце строки
        #newl = re.sub("^\s+|\n|\r|\s+$", '', l)
        l = re.sub("^\s+|", '', l)
        newl = re.sub("\s+", ' ', l)
        values = np.array(newl.split(" ")) 
        #print (values)
        sp.append(float(values[1]))
        #sp.append(float(values[1])/10000)
        wl.append(float(values[0]))
    return wl, sp
    
indexes = [
            MCARI2,
            MTVI,
            MTVI2,
            SIPI,
            LV,
            BR,
            GNDVI2,
            DI1,
            SIPI2,
            NPCI,
            BR625,
            PSNDchla,
            PSSRa,
            PSSRc,
            New2,
            New13]    
    
#d = "E:/!data/FSR02/FSR_poleti_kajkovo_180620/polet2/TripleSat_kajkovo/flaash2_skip_some_points/"
d = "E:/!data/FSR02/FSR_poleti_kajkovo_180620/polet2/indexes_for_LDA_TripleSat/without_flaash/"
folder = []
for i in os.walk(d):
    folder.append(i)
    
dict_for_vegs = {}
dict_all_spectrs = {}
for address, dirs, files in folder:
    for d in dirs:
        if d:
            subfolder = []
            mas = []
            for i in os.walk(address + "/" + d):
                subfolder.append(i)
            for ad, dd, ff in subfolder:
                print (d)
                for file in ff:
                    indexdict = {} # словарь 
                    sp_name = address+"/" + d + "/" + file
                    if 'Index' in sp_name:
                        continue
                    #print (d, os.path.basename(sp_name))
                    wl, sp = read_envifile(sp_name)
                    if d == "zdorov_rast":
                        lvl = 1
                    elif d == "podlesok":
                        lvl = 2
                    elif d == "usihanie":
                        lvl = 3
                    elif d == "suhostoj":
                        lvl = 4
                    elif d == "pochva":
                        lvl = 5
                    output_file = open(address+"/"+d + "/" + os.path.basename(sp_name) + "_IndexLevels.txt",'w')
                    #запись индексов
                    output_file.write("lvl\t"+str(lvl)+'\n')
                    for func in indexes:
                         output_file.write(str(func.__name__) +' ' + str(round(func(sp),3)) + "\n")
                    output_file.write("\n")
                    output_file.close()
                    for func in indexes:
                        iname = str(func.__name__)
                        ival = round(func(sp),5)
                        if indexdict.get(iname) == None: 
                            indexdict.update({iname:ival})
                        else:
                            print ("Some error", iname, ival, " already exists")
                    # Создание словаря по типам растительности в подпапках для отображения зависимостей одного индекса растительности от другого
                    if dict_for_vegs.get(d) == None: 
                        dict_for_vegs.update({d:[os.path.basename(sp_name).split(".")[0]]})
                    else:
                        dict_for_vegs.get(d).append(os.path.basename(sp_name).split(".")[0]) 
                    
                    if dict_all_spectrs.get(os.path.basename(sp_name).split(".")[0]) == None: 
                        dict_all_spectrs.update({os.path.basename(sp_name).split(".")[0]:indexdict})
                    else:
                        print ("Error Indexes ", d, os.path.basename(sp_name).split(".")[0], " already exists в dict_all_spectrs")
                        #dict_for_vegs.get(d).append(os.path.basename(sp_name)) 
                        
#print (dict_for_vegs, dict_all_spectrs)                        
#dict_all_spectrs имя (os.path.basename(sp_name))    : словарь индексов 
# '08_33_30_739#152': {'MCARI2': 0.24594, 'MTVI': 0.16198, 'MTVI2': 0.22889, 'SIPI': 1.08088, 'LV': 2.90859, 'BR': -0.76, 'GNDVI2': 0.86252, 'DI1': 0.1054, 'SIPI2': 0.9899, 'NPCI': 7.33333, 'BR625': 0.04394, 'PSNDchla': 0.91582, 'PSSRa': 22.76, 'PSSRc': -29.94737, 'New2': -7.33333, 'New13': 13.54762}
# dict_for_vegs    тип ('usihanie')  : имена ('08_29_24_386#112')       


    



# Отображение зависимостей одного индекса растительности от другого
#d = "E:/!data/FSR02/FSR_poleti_kajkovo_180620/polet2/for_indexes_2/"
onename = "BR"
names = [str(el.__name__) for el in indexes]
for part_name in names: # некий индекс из списка
    x = []
    y = []
    cap = []
    dx = {}
    dy = {}
    for sp_name in dict_all_spectrs: # итерация по спектрам
        et = sp_name
        name, val = read_index(dict_all_spectrs[sp_name])
        #print (name, val, "NV")
        for num, n in enumerate(name):
            if n== part_name and n!= onename:
                #print et, name[num], val[num]
                if dx.get(et) == None: # et - имя спектра с индексами
                    dx.update({et:[val[num]]})
                else:
                    dx.get(et).append(val[num])  
            elif n == onename:
                #print et, name[num], val[num]

                if dy.get(et) == None: 
                    dy.update({et:[val[num]]})
                else:
                    dy.get(et).append(val[num])  
        #save_plot(wl, sp, (d + key_im + "_speya"))
    
    
    plt.figure(str(part_name) + onename)
    for key in dx:
        if key in dict_for_vegs['zdorov_rast']:
            plt.plot(dx[key][0],dy[key][0],"gs")
            '''if dy[key][0]>1.25 or dy[key][0]<0:
                print (key, dy[key][0], "BR value ZD --")
            else:
                plt.plot(dx[key][0],dy[key][0],"gs")'''

        elif key in dict_for_vegs['suhostoj']:
            plt.plot(dx[key][0],dy[key][0],"ks")
            '''if dx[key][0]>20:
                print (key, dx[key][0], "PSSRa value SUH --")
            else:
                plt.plot(dx[key][0],dy[key][0],"ks")'''
        elif key in dict_for_vegs['usihanie']:
            plt.plot(dx[key][0],dy[key][0],"ms")
            '''if dx[key][0]>12:
                print (key, dx[key][0], "PSSRa value USIH --")
            elif dy[key][0]>0.7:
                print (key, dy[key][0], "BR value USIH --")
            else:
                plt.plot(dx[key][0],dy[key][0],"ms")'''
        elif key in dict_for_vegs['podlesok']:
            plt.plot(dx[key][0],dy[key][0],"cs")
        elif key in dict_for_vegs['pochva']:
            plt.plot(dx[key][0],dy[key][0],"ys")
        else:
            print (key, "UNMATCH")
    plt.title(str(part_name)+u"и " + onename, family = "verdana")
    plt.xlabel(str(part_name), family = "verdana", size=15)
    plt.ylabel(onename, family = "verdana", size=15)
    plt.grid(True)
    black_line = mlines.Line2D([], [], color='k', marker = "s", linestyle = '-.', linewidth=0.7, label= u'Сухая хвоя')
    #red_line = mlines.Line2D([], [], color='r', marker = "s", linestyle = '-.', linewidth=0.7, label= u'Теневые пиксели')
    green_line = mlines.Line2D([], [], color='g', marker = "s", linestyle = '-.', linewidth=0.7, label= u'Здоровая ель')
    magenta_line = mlines.Line2D([], [], color='m', marker = "s", linestyle = '-.', linewidth=0.7, label= u'Ель с признаками усыхания')
    cyan_line = mlines.Line2D([], [], color='c', marker = "s", linestyle = '-.', linewidth=0.7, label= u'Подлесок')
    y_line = mlines.Line2D([], [], color='y', marker = "s", linestyle = '-.', linewidth=0.7, label= u'Почва')
    #o_line = mlines.Line2D([], [], color='orange', marker = "s", linestyle = '-.', linewidth=0.7, label= u'Кукуруза')

    plt.legend(handles=[black_line, green_line, y_line, magenta_line, cyan_line]) 
                    