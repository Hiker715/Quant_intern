import os
import csv
 
def getline(path,mylist):
    with open(path, "r") as ff:
        reader = csv.reader(ff)  
        lines = 0
        for item in reader:  
            lines += 1
    return path.split('\\')[-1] + "    " + str(lines) 
 
 
if  __name__=="__main__":
    path = "F:\\Intern\\3outputFile\\EPSFile\\EPSAvg_change_perc\\Factor_Exposure_lag_3"  
    filelist = os.listdir(path)  
    mylist = [] 
 
    for  filename  in filelist:
        newpath=path+"\\"+filename 
        mylist.append(getline(newpath, mylist))

    mylist = '\n'.join(mylist)


    with open("F:\\Intern\\3outputFile\\outTXT\\epsRowNumber.txt", 'w') as f:
        f.write(mylist)
    
 
