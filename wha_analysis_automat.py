import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage import morphology
import cv2
import glob
import pandas as pd
import time


img_number_analyzed = 0
img_number = []
scratch_list = []
path = "tov_20221111_ic10/*/*.tif"
disk_size = 50
threshold_sensibility = 0.92
hole_size_fill = 100000
artifact_size_remove = 50000
results_file_name = "tres92_tov20221111_ic10.xlsx"

start_time = time.time()
for file in glob.glob(path):
    #wczytanie zdjecia i przerzucenie do 8bit
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #rozmazanie dla ujednolicenia i zmniejszenia szumow
    blur = cv2.blur(gray,(10,10))
    #liczenie entropii obrazu
    entropy_img = entropy(blur, disk(disk_size))
    # narzucenie tresholdu i wyciagniecie wszystkiego mniejszego
    tresh = threshold_otsu(entropy_img)
    binary = entropy_img <= (threshold_sensibility*tresh)
    #wypelnianie pustych przestrzeni i usuwanie szumow
    cleaned = morphology.remove_small_objects(binary, hole_size_fill)
    invclean = ~cleaned
    cleaned2 = morphology.remove_small_objects(invclean, artifact_size_remove)
    
    #pokazanie plotu i zapisanie obrazu
    #a=np.array(~cleaned2)
    #print(a.shape)
    plt.imshow(~cleaned2)
    plt.savefig(file + ".png", bbox_inches="tight")
    #cv2.imwrite("0h/obraz.jpg", img2)
    #obliczenie wielkosci rany i prezentacja wynikow
    scratch_vol = np.sum(cleaned2 == False)
    file_short = file[19:28] + " " + file[-8:-4]
    print(file_short + " " + str(scratch_vol))
    #print (scratch_vol)
    scratch_list.append(scratch_vol)
    img_number_analyzed += 1
    img_number.append(file_short)
end_time = time.time()
total_time = (end_time - start_time)
time_per_image = (total_time/img_number_analyzed)

print("przeanalizowano zdjec: " + str(img_number_analyzed))
print("czas analizy:" + str(total_time))
print("sredni czas analizy jednego zdjecia: " + str(time_per_image))
print(img_number)
print(scratch_list)

new = pd.DataFrame([img_number, scratch_list])
#new = pd.Series(img_number)
new.to_excel(results_file_name) 