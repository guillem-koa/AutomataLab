def get_cam2_to_front(file_list):
    import numpy as np
    import os
    times = []
    for file in file_list:
        times.append([thing for thing in file.split("_") if ":" in thing][0])

    new_image_files =[]
    for time in list(set(times)):
        images_same_time = [file for file in file_list if time in file]
        # Find the index of the file with a '2' in its filename
        index = -1
        for i, filename in enumerate(images_same_time):
            if 'Cam2' in filename:
                index = i
                break

        # Move the file with a '2' in its filename to the front of the list
        if index != -1:
            images_same_time.insert(0, images_same_time.pop(index))
        
        new_image_files.append(images_same_time)
    
    return [item for sublist in new_image_files for item in sublist]

def get_setting_vars(img_path):
    # Extracting setting variables from filename
    setting_vars = img_path.split("_")

    setting_vars_dict = {"InputType": setting_vars[1], 
                       "Date": [string for string in setting_vars if "-" in string][0], 
                       "Time": [string for string in setting_vars if ":" in string][0],
                       "Plate": [string for string in setting_vars if "P" in string and len(string)<4][0]}
    return setting_vars_dict

def rowscols_2_platedistribution(rows, cols):
    import numpy as np
    plate_distribution = np.empty((len(rows),len(cols)), dtype = 'object')
    counter = {}
    for i, row in enumerate(rows):
        if row not in counter:
            counter[row] = 1
        for j, col in enumerate(cols):
            if len(row)!=0 and len(col)!=0:
                plate_distribution[i,j]=row + "+" + col + "+" + str(counter[row])
            else:
                plate_distribution[i,j]="++"
        counter[row]+=1
    return plate_distribution

def plate_distribution_dict(plate_distribution):
    # Write 'InputName' and 'ReporterName' as 1D arrays following the order left-right & top-down
    rows_dict={"Replica": [], "InputName":[], "ReporterName":[]}
    for i in range(plate_distribution.shape[0]):
        for j in range(plate_distribution.shape[1]):
            rows_dict["Replica"].append(plate_distribution[i,j].split("+")[2]) 
            rows_dict["InputName"].append(plate_distribution[i,j].split("+")[0]) 
            rows_dict["ReporterName"].append(plate_distribution[i,j].split("+")[1])
    return rows_dict

def experiment_results(experiment_dir, plate_distribution):
    import cv2 
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from imageproc import get_pocillos, complete_the_grid, draw_circles, array_from_pic, circle_crop    
    
    # Set experiment directory as working directory
    os.chdir(experiment_dir)
    
    # Initilize empty dictionary which will contain all results of experiments
    output_dict_list = []

    # Get all files in each plate-directory, and filter out the image files
    files = os.listdir(experiment_dir)
    image_files = [file for file in files if file.endswith((".jpg", ".jpeg", ".png", ".gif"))]    

    if len(image_files)!=0: # Condition for considering a PLATE directory

        image_files = get_cam2_to_front(image_files)
        
        #print(plate, "\n")
        #print(image_files)

        # output_dict will have all the results of a single plate (setting variables and measures)
        output_dict = {}

        # Add setting variables to 'output_dict'    
        setting_vars = get_setting_vars(image_files[0])   
        output_dict.update(setting_vars)

        # Add InputName and ReporterName to 'output_dict'
        output_dict.update(plate_distribution_dict(plate_distribution))

        # Add measure variables GFP and OD to 'output_dict'
        for i, img_file in enumerate(image_files):
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            #print(img_file, "\n")
            
            if 'Cam2' in img_file:
                circles = get_pocillos(img_bw)
                new_circles = complete_the_grid(circles)
            
                arr = array_from_pic(img, new_circles, "blue","mean")
                output_dict["od"] = arr.flatten()
                
                # Checks
                draw_circles(img, new_circles)
                #print(arr, "\n")

            if 'Cam1' in img_file:

                arr = array_from_pic(img, new_circles, "green","q9")
                output_dict["gfp"] = arr.flatten()
                
                draw_circles(img, new_circles)
                #print(arr, "\n")
                

            # Update the list with all the plates 'output_dict_list'
            output_dict_list.append(output_dict)

    
    
    # Create a single dataframe with all 'output_dict' information
    df = pd.concat([pd.DataFrame(dictionary) for dictionary in output_dict_list], ignore_index=True)

    # Eliminate rows corresponding to empty
    df = df.loc[(df['InputName']!='') & (df['ReporterName']!='')]
    
    return df