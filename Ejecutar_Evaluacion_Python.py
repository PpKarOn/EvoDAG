import pandas
import numpy
import os
import subprocess

def BER(y, yh):
    u = numpy.unique(y)
    b = 0
    for cl in u:
        m = y == cl
        b += (~(y[m] == yh[m])).sum() / float(m.sum())
    return (b / float(u.shape[0])) * 100.
datasets = ['banana','titanic','thyroid','diabetis','breast-cancer','flare-solar','heart','ringnorm','twonorm','german','waveform','splice','image']
datasets_size = [100,100,100,100,100,100,100,100,100,100,100,20,20]
datasets_size = [1,1,1,1,1,1,1,1,1,1,1,1,1]

print('Evaluation')
os.chdir('/shared/cnsanchez/EvoDAG')
#os.chdir('/home/claudia/Documentos/DOCTORADO/CODIGO/EvoDAG')
folderRes = '../res/TB/'
#folderRes = '../res/EvoDAG/'
folderData = '../data/'

columns=['dataset','error','fitness','size']
index=numpy.arange(len(datasets))
df = pandas.DataFrame(columns=columns,index=index)
for i in range(len(datasets)):
    archivo = datasets[i]
    print('------------------------------------------------',archivo)
    error = 0
    fitness = 0
    size = 0
    for j in range(1,datasets_size[i]+1):
        p = subprocess.Popen(['EvoDAG-utils --size ' +folderRes+archivo+'_test_data_'+str(j)+ '.model'],stdout=subprocess.PIPE,shell=True)
        size += float(p.stdout.read().decode("utf-8")[6:-1])
        p = subprocess.Popen(['EvoDAG-utils --fitness ' +folderRes+archivo+'_test_data_'+str(j)+ '.model'],stdout=subprocess.PIPE,shell=True)
        fitness += float(p.stdout.read().decode("utf-8")[16:-1])
        archivo1 = folderRes+archivo+'_test_data_'+str(j)+'.predict'
        archivo2 = folderData+archivo+'_test_labels_'+str(j)+'.csv'
        D1 = pandas.read_csv(archivo1,sep=",")
        D2 = pandas.read_csv(archivo2,sep=",")
        error += BER(D1.values,D2.values)
    error /= datasets_size[i]
    size /= datasets_size[i]
    fitness /= datasets_size[i]
    df['dataset'][i] = archivo
    df['error'][i] = error
    df['fitness'][i] = fitness
    df['size'][i] = size
    print(df)
    df.to_csv(folderRes+'ares_1.csv',sep=',')
print(df)
