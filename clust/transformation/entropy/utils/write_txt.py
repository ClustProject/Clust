
import pandas as pd
def write_txt(path1,path2,path3,path4,avg1,avg2,std1,std2):

    file1 = open(path1,'w')
    file1.write('CHF 환자들의 RRIs에 대한 MCRDE 평균값(index=scale).\n')
    file1.close()
    Sr1 = pd.Series(avg1, index = list(range(1,26)))
    Sr1.to_csv(path1,mode='a', index = list(range(1,26)),sep='\t', header=False)
    
    file2 = open(path2,'w')
    file2.write('Healthy 피실험자들의 RRIs에 대한 MCRDE 평균값(index=scale).\n')
    file2.close()
    Sr2 = pd.Series(avg2, index = list(range(1,26)))
    Sr2.to_csv(path2,mode='a', index = list(range(1,26)),sep='\t', header=False)
    
    file3 = open(path3,'w')
    file3.write('CHF 환자들의 RRIs에 대한 MCRDE 표준편차값(index=scale).\n')
    file3.close()
    Sr3 = pd.Series(std1, index = list(range(1,26)))
    Sr3.to_csv(path3,mode='a', index = list(range(1,26)),sep='\t', header=False)
    
    file4 = open(path4,'w')
    file4.write('Healthy 피실험자들의 RRIs에 대한 MCRDE 표준편차값(index=scale).\n')
    file4.close()
    Sr4 = pd.Series(std2, index = list(range(1,26)))
    Sr4.to_csv(path4,mode='a',index = list(range(1,26)),sep='\t', header=False)