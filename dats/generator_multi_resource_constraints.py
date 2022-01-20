import numpy as np
import os
import shutil
import glob
import math
from random import randint
#   a time slot is 20 minutes, one days is composed of 24 time slots for every dock
#   
tf = 16
#nbTF_min = 24   # 8 hours * 20 minutes slots
#nbTF_max = 120  # 10 hours and 5 minutes slots

nbDocks_min = 20
nbDocks_max = 60

nbTrucks_min = 3
nbTrucks_max = 5


#minServiceTime = 20
#maxServiceTime = 30

#workingHour_min = 8
#workingHour_max = 10

def create_directory(dir ):
    if os.path.exists(dir):
        try:
            os.rmdir(dir)
        except OSError:
            print("Deletion of the directory %s failed")
        else:
            print("Successfully deleted the directory %s ")
    else:
        try:
            os.mkdir(dir)
        except OSError:
            print("Creation of the directory %s failed")
        else:
            print("Successfully created the directory %s ")

create_directory("instances")

nbScenarios = input("Enter # scenarios:")
print("#Scenarios is: " + nbScenarios)
nbScenarios  = int(nbScenarios)


for dock in range(nbDocks_min, nbDocks_max+1, 2):
    #for truck in range(int((workingHour_min-1)*60.0/maxServiceTime*dock),   int((workingHour_max-1)*60.0/minServiceTime*dock), 2):

    #for truck in range(15, min(150, dock*8), 2):
    for truck in range(nbTrucks_min*dock, min(nbTrucks_max*dock+1, 200), 5):

        print(f"tf-{tf}-d-{dock}-tr-{truck}-Sce-{nbScenarios}-RC.dat\n")
        f= open(f"tf-{tf}-d-{dock}-tr-{truck}-Sce-{nbScenarios}-RC.dat", "w+")
        df_file= open(f"tf-{tf}-d-{dock}-tr-{truck}-Sce-{nbScenarios}-RC.df", "w+")


        f.write("begin{nbTrucks}\n")
        f.write(str(truck))
        f.write("\nend{nbTrucks}\n")

        f.write("begin{nbDocks}\n")
        f.write(str(dock))
        f.write("\nend{nbDocks}\n")

        f.write("begin{nbTF}\n")
        f.write(str(tf))
        f.write("\nend{nbTF}\n")

        f.write("begin{nbScenarios}\n")
        f.write(str(nbScenarios))
        f.write("\nend{nbScenarios}\n")

        nbResourcePmin = 2
        nbResourcePmax = 5
        nbResourceEmin = 0
        nbResourceEmax = 3
        nbResourceVmin = 1
        nbResourceVmax = 4


        availableP = math.ceil(( nbResourcePmax)/2* dock  * 1.0 / 3.0)
        availableE = math.ceil(( nbResourceEmax)/4 * dock   * 1.0 / 3.0)
        availableV = math.ceil(( nbResourceVmax)/4 * dock   * 1.0 / 3.0)
       

        f.write("begin{nbResourcePersonel}\n")
        f.write(str(availableP))
        f.write("\nend{nbResourcePersonel}\n")

        f.write("begin{nbResourceEquipment}\n")
        f.write(str(availableE))
        f.write("\nend{nbResourceEquipment}\n")

        f.write("begin{nbResourceVehicule}\n")
        f.write(str(availableV))
        f.write("\nend{nbResourceVehicule}\n")

        r = np.random.randint(0, tf*0.75, truck)
        p = np.random.randint(tf/8, tf/4, truck)


        delta = np.random.randint(1, 3, truck)
        delay = np.random.randint(1, 3, truck)
        penalty = np.random.randint(5.0, 10.0, truck)

        d = [ min(math.ceil(r[j] + p[j]*1.5+delta[j]+delay[j]), tf) for j in range(truck)]

        

        resourceP = np.random.randint(nbResourcePmin, nbResourcePmax, truck)
        resourceE = np.random.randint(nbResourceEmin, nbResourceEmax, truck)
        resourceV = np.random.randint(nbResourceVmin, nbResourceVmax, truck)


        dat_filecontent = "begin{Tasks}{n\tr\td\tdelta\tpenalty\tp0\tresourcePersonel0\tresourceEquipments0\tresourceVehicle0"
        df_filecontent = "n\tstart\tdeadline\tdocking\tpenalty\tprocessing0\tresourcePersonel0\tresourceEquipments0\tresourceVehicle0"


        for s in range(1,nbScenarios):
            dat_filecontent +=  f"\tp{s}\tresourcePersonel{s}\tresourceEquipments{s}\tresourceVehicle{s}"
            df_filecontent +=   f"\tprocessing{s}\tresourcePersonel{s}\tresourceEquipments{s}\tresourceVehicle{s}"

        dat_filecontent +=  "\n"
        df_filecontent  +=  "\n"



        for j in range(truck):
            dat_filecontent +=  f"{j+1}\t{r[j]}\t{d[j]}\t{delta[j]}\t{penalty[j]}\t{p[j]}\t{resourceP[j]}\t{resourceE[j]}\t{resourceV[j]}\t"
            df_filecontent += f"{j+1}\t{r[j]}\t{d[j]}\t{delta[j]}\t{penalty[j]}\t{p[j]}\t{resourceP[j]}\t{resourceE[j]}\t{resourceV[j]}\t"

            for s in range(1, nbScenarios):
                    
                new_P =  max( randint(-1,2) + p[j], 0) #random.choice([-1, 1])
                new_resourceP = max( randint(-1,2) + resourceP[j], 0) 
                new_resourceE = max( randint(-1,1) + resourceE[j], 0) 
                new_resourceV = max( randint(-1,2) + resourceV[j], 0) 

                dat_filecontent +=  f"{new_P}\t{new_resourceP}\t{new_resourceE}\t{new_resourceV}\t"
                df_filecontent += f"{new_P}\t{new_resourceP}\t{new_resourceE}\t{new_resourceV}\t"

                #print(dat_filecontent)
                #print(df_filecontent)

            dat_filecontent +=  "\n"
            df_filecontent += "\n"


        dat_filecontent += "end{Tasks}\n"
        df_filecontent +="end{Tasks}\n"

        f.write(dat_filecontent)
        df_file.write(df_filecontent)




        #shutil.move(f, '.\\instances')