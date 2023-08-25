def get_datetimes(plate_dir): 
    import os
    dates, times = [], []
    image_files= [file for file in os.listdir(plate_dir) if file.endswith('jpg')]

    if len(image_files)>1:
        for file in image_files:
            dates.append([thing for thing in file.split("_") if "-" in thing][0])
            times.append([thing for thing in file.split("_") if ":" in thing][0])

        date_times = [date + '_' + time for date, time in zip(dates, times)]

        return list(set(date_times))
    else:
        print("Not enough images in plate directory!")

def get_cam2_to_front(file_list):
    file_list.sort(key=lambda s: 'Cam2' not in s)
    return file_list

def get_plate_distribution_vars(plate_distribution_path):
    import pandas as pd
    import numpy as np

    plate_distribution= pd.read_excel(plate_distribution_path)

    # inputType
    filename = plate_distribution_path.split('/')[-1]
    inputType = 'Water' if 'ater' in filename else ('Pathogen' if 'ogen' in filename else 'Unknown')

    # Input and Sensors
    inputs = plate_distribution.values[0:8, 0].tolist()
    sensors = plate_distribution.columns[2:12].tolist() + ['Blanc']

    # Replica
    replicas = []
    counter = {}

    for item in inputs:
        if item not in counter:
            counter[item] = 1
        replicas.append(counter[item])
        counter[item] += 1

    # Guess, Concentration & creating 'plate_distribution_vars_dict'
    if inputType == 'Water':
        guess = plate_distribution[plate_distribution.columns[-1]] 

        plate_distribution_vars_dict={"InputName": inputs,
                                  "Strain": "",
                                  "InputType": inputType, 
                                  "Replica": replicas,
                                  "Concentration": "",
                                  "Lio/Gly": "",
                                  "Guess": guess
                                  }
    if inputType == 'Pathogen':
        concentration = plate_distribution[plate_distribution.columns[-4]] 
        strain = plate_distribution[plate_distribution.columns[-3]] 
        liogly = plate_distribution[plate_distribution.columns[-1]] 

        plate_distribution_vars_dict={"InputName": inputs,
                                      "Strain": strain,
                                      "InputType": inputType, 
                                      "Replica": replicas,
                                      "Concentration": concentration,
                                      "Lio/Gly": liogly,
                                      "Guess": ""
                                      }
    
    return (plate_distribution_vars_dict, sensors)

def get_measures(plate_dir, experiment_datetimes):
    import os
    import cv2
    from imageproc import get_pocillos, get_pocillos_yolo, get_positions, complete_the_grid, array_from_pic, draw_circles

    # Get all files in each plate-directory, and filter out the image files
    files = os.listdir(plate_dir)
    image_files = [file for file in files if file.endswith((".jpg", ".jpeg", ".png", ".gif"))]    

    if len(experiment_datetimes)!=0: # Condition for considering a PLATE directory
        
        # output_dict will have all the results of a single plate (setting variables and measures)
        output_list = []

        for datetime in experiment_datetimes:
            datetime_image_files = [file for file in image_files if datetime in file]
            datetime_image_files = get_cam2_to_front(datetime_image_files)
            #print(datetime, datetime_image_files)

            od_gfp_dict = {}
            od_gfp_dict['PictureDate'] = datetime.split('_')[0]
            od_gfp_dict['PictureTime'] = datetime.split('_')[1]

            # Add measure variables GFP and OD to 'output'
            for i, img_file in enumerate(datetime_image_files):
                #img = cv2.imread(img_file)
                img = cv2.imread(os.path.join(plate_dir,img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
                if 'Cam2' in img_file:
                    # Option 1: using yolo to detect wells 
                    #circles = get_positions(get_pocillos_yolo(img))

                    # Option 2: using cv2.HoughCircles to detect wells
                    circles = get_positions(get_pocillos(img_bw))

                    circles = complete_the_grid(circles) if len(circles)<96 else circles
                
                    arr = array_from_pic(img, circles, "blue","mean")
                    od_gfp_dict["od"] = arr
                    
                    # Checks
                    draw_circles(img, circles)
                    #print(arr, "\n")

                if 'Cam1' in img_file:

                    arr = array_from_pic(img, circles, "green","q9")
                    od_gfp_dict["gfp"] = arr
                    
                    draw_circles(img, circles)
                    #print(arr, "\n")

            output_list.append(od_gfp_dict)

    return output_list

def mount_df(plate_distribution_vars, sensors, measures_list):
    import pandas as pd
    df_list = []
    for measure in measures_list:
        plate_distribution_vars_df =  pd.DataFrame(plate_distribution_vars)

        sensors_abs = [sensor + '_abs' for sensor in sensors]
        df_od = pd.DataFrame(measure['od'][:, 1:], columns=sensors_abs)

        sensors_gfp = [sensor + '_gfp' for sensor in sensors]
        df_gfp = pd.DataFrame(measure['gfp'][:, 1:], columns=sensors_gfp)

        date_time_df = pd.DataFrame({'PictureDate': [measure['PictureDate']]*8, 'PictureTime': [measure['PictureTime']]*8})

        df_list.append(pd.concat([plate_distribution_vars_df, date_time_df, df_od, df_gfp], axis=1))

    return pd.concat(df_list)