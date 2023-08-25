def get_pocillos(img_bw):
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    # Circle detection
    # Fourth argument = Distance between circles = should be set to twice the double of the maximum radius
    circles = cv2.HoughCircles(img_bw, cv2.HOUGH_GRADIENT, 1, 100, param1=50,param2=30,minRadius=40,maxRadius=50)
    circles = circles.astype(int)[0,:,:]
    
    # Location algorithm for desired pocillo    
    x_kmeans = KMeans(n_clusters=12, n_init=10).fit(circles[:,0].reshape(-1, 1))
    x_labels, x_centers = x_kmeans.labels_, x_kmeans.cluster_centers_.squeeze()
    
    x_new_labels = np.zeros_like(x_labels)
    for i in range(len(x_centers)):
        x_new_labels[x_labels == i] = np.where(np.argsort(x_centers) == i)[0][0] 

    y_kmeans = KMeans(n_clusters=8, n_init=10).fit(circles[:,1].reshape(-1, 1))
    y_labels, y_centers = y_kmeans.labels_, y_kmeans.cluster_centers_.squeeze()
    
    y_new_labels = np.zeros_like(y_labels)
    for i in range(len(y_centers)):
        y_new_labels[y_labels == i] = np.where(np.argsort(y_centers) == i)[0][0]
        
    # Output is a dataframe where each row corresponds to a different circle.
    # We give coordinates of center, value of radius, row number and column number (in the grid)
    
    return np.column_stack((circles, y_new_labels,  x_new_labels))


def complete_the_grid(circles):
    import numpy as np
    max_row, min_row, max_col, min_col = circles[:,3].max(), circles[:,3].min(), circles[:,4].max(), circles[:,4].min()
    new_circles = []
    new_rad = round(circles[:,2].mean()-3*np.std(circles[:,2]))
    
    # We linear fit the x and y coordinates of the centers of the pocillos of a same row, to predict the missing ones
    for i in range(max_row+1):
        row = circles[circles[:,3]==i]

        if len(row)>1:
            x = row[:,4]
            y0 = row[:,0]
            y1 = row[:,1]

            def f0(n):
                slope0, intercept0 = np.polyfit(x,y0,1)
                return round(intercept0 + n*slope0)
            def f1(n):
                slope1, intercept1 = np.polyfit(x,y1,1)
                return round(intercept1 + n*slope1)

            for j in range(max_col+1):
                new_circles.append([f0(j), f1(j), new_rad, i, j])

    new_circles = np.array(new_circles)
    
    # The following computes an average vertical displacement between two rows of pocillos
    rows_available = np.unique(new_circles[:,3])
    rows_available[0], rows_available[1]
    first_row_available = rows_available[0]
    second_row_available = rows_available[1]
    first_mean_height = np.mean(circles[circles[:, 3]==first_row_available][:,1])
    second_mean_height = np.mean(circles[circles[:, 3]==second_row_available][:,1])
    vertical_displacement = (second_mean_height - first_mean_height)/(second_row_available-first_row_available)
    
    # This code adds missing rows 
    for i in range(min_row, max_row+1):
        if len(new_circles[new_circles[:,3]==i])==0:
            current_row = new_circles[new_circles[:,3]==i-1]
            current_row[:,1] += round(vertical_displacement)
            current_row[:,3] += 1
            new_circles = np.vstack([new_circles, current_row])
    
    new_circles = np.array(new_circles)
    return new_circles


def draw_circles(img, circles):
    import cv2
    if circles is not None:
        import copy
        canvas_img = copy.copy(img)
        circles = circles.astype(int)
        for pt in circles:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(canvas_img, (a, b), r, (0, 255, 0), 7)
            cv2.circle(canvas_img, (a, b), 1, (0, 0, 255), 7)
            
    import matplotlib.pyplot as plt
    plt.imshow(canvas_img)
    plt.show()

def circle_crop(img, circle): 
    import numpy as np
    center, radius = circle[0:2], circle[2]
    cropped_pocillo = img[center[1] - radius: center[1] + radius, center[0] - radius: center[0] + radius]
    width, height = cropped_pocillo.shape[0:2]   
    if width != height:
        return np.array([[[-1,-1,-1]]])
    else:
        import copy
        final_img = copy.copy(cropped_pocillo)
        for x in range(width):
            for y in range(height):
                dist = (x-width//2)**2 + (y-height//2)**2 
                if dist > radius**2:
                    final_img[y,x] = 0
        return final_img

# Gets picture and outputs 8x12 output array. Arguments:
## 3-channel image 'img'
## 'circles' array
## color_channel: "green", or "blue"
## method: "mean", or "q9"

def array_from_pic(img, circles, color_channel, method):
    import numpy as np
    # Initialize the output array to 0
    output_array = np.zeros((8,12))
    grid_positions = circles[:,3:5]
    
    for i, position in enumerate(grid_positions):
        cropped_pocillo = circle_crop(img, circles[i,0:3])
        
        # Choose color channel for our analysis
        if color_channel == "green":
            cropped_pocillo = cropped_pocillo[:,:,1]
        if color_channel == "blue":
            cropped_pocillo = cropped_pocillo[:,:,2]
            
        # Choose which statistic to take for all pocillo-values 
        if method == "mean":
            output_array[position[0], position[1]] = np.mean(cropped_pocillo[cropped_pocillo != 0])
        if method == "q9":
            output_array[position[0], position[1]] = np.percentile(cropped_pocillo[cropped_pocillo != 0], 90)

    return output_array.astype(int)

