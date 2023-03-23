import os, time, sys, cv2, warnings, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from IPython.display import Image


def write_points(lanes_xys, result_img_name):
    lanes_txt = []
    for xys in lanes_xys:
        if len(xys) >= 2 :
            extract_start_end_points = [xys[0], xys[-1]] 
            temp = ' '.join(str(x) for x in sum(extract_start_end_points,()))
            lanes_txt.append(temp)
        
    with open(result_img_name.replace('.jpg','.lines.txt'), 'w', encoding='UTF-8') as f:
        for points in lanes_txt:
            f.write(points + '\n')

def get_cam_e(): 
    ## 현재 카메라 정보를 알 수 없기 때문에 roll, pitch, yaw 임의 지정
    roll_stand = 0 
    pitch_stand = 110
    yaw_stand = 0
    cos_roll = math.cos(math.radians(roll_stand))
    sin_roll = -1*math.sin(math.radians(roll_stand))
    cos_pitch = math.cos(math.radians(pitch_stand)) 
    sin_pitch = math.sin(math.radians(pitch_stand)) 
    cos_yaw = math.cos(math.radians(yaw_stand)) # 이미지 기준이므로 yaw는 무시
    sin_yaw = math.sin(math.radians(yaw_stand)) 
    ## camera extrinsic matrix
    cam_e = np.mat([[cos_roll*cos_yaw, cos_roll*sin_yaw*sin_pitch - sin_roll*cos_pitch, cos_roll*sin_yaw*cos_pitch + sin_roll*sin_pitch, 0],
                    [sin_roll*cos_yaw, sin_roll*sin_yaw*sin_pitch + cos_roll*cos_pitch, sin_roll*sin_yaw*cos_pitch - cos_roll*sin_pitch, 0],
                    [      -1*sin_yaw,                               cos_yaw*sin_pitch,                               cos_yaw*cos_pitch, 1],
                    [               0,                                               0,                                               0, 1]])
    return cam_e

def get_line_point(start_point:list, end_point:list)->list:
    step = 20
    delta_x = (end_point[0] - start_point[0])/(step-1)
    delta_y = (end_point[1] - start_point[1])/(step-1)
    points = []
    for i in range(step):
        points.append(delta_x*i + start_point[0])
        points.append(delta_y*i + start_point[1])
    return points

def load_lines(lanes_xys, img_colrow)->list:
    ori_img_cols, test_img_cols, ori_img_rows, test_img_rows = img_colrow
    ## variables of coordinate transform
    scale_x = ori_img_cols/test_img_cols
    scale_y = ori_img_rows/test_img_rows

    x_original_center = (test_img_cols-1) / 2
    y_original_center = (test_img_rows-1) / 2

    x_scaled_center = (ori_img_cols-1) / 2
    y_scaled_center = (ori_img_rows-1) / 2
    
    n = len(lanes_xys)
    scaled_lines = [[]*n for _ in range(n)]
    for i in range(len(lanes_xys)):
        for j in range(0, len(lanes_xys[i]), 2):
            ## Subtract the center, scale, and add the "scaled center".
            scaled_lines[i].append( (lanes_xys[i][j] - x_original_center)*scale_x + x_scaled_center)
            scaled_lines[i].append( (lanes_xys[i][j+1] - y_original_center)*scale_y + y_scaled_center) 
            
    scaled_lines = sorted(scaled_lines, key=lambda x: x[0])
    return scaled_lines

def get_nearest_point(scaled_lines:list, ref_point)->int:
    lines_list = [] 
    for point in scaled_lines:
        lines_list.append(point[0])
    lines = np.array(lines_list)
    nearest_line = lines_list.index(lines.flat[np.abs(lines - ref_point).argmin()]) # 가장 가까운점 찾기
    return nearest_line

def get_world_coord(line:list, cam_param:list):
    img_coord = [[0],[0],[1]]
    world_line = []
    for i in range(0,len(line),2):
        img_coord[0] = [line[i]]
        img_coord[1] = [line[i+1]]
        world_X, world_Y= img2world(img_coord, cam_param)
        world_line.append(world_X)
        world_line.append(world_Y)
    return world_line

def get_distance(world_point1:list, world_point2:list)->float:
    world_distance = math.sqrt((world_point1[0]-world_point2[0])**2 + (world_point1[1]-world_point2[1])**2)
    return world_distance


def get_linear_fit_line(x,y):
    ## warning 발생 : pip install numpy==1.21 하면 경고가 사라진다고 함
    fit_linear_line = np.polyfit(x, y, 1)
    return fit_linear_line

def prepare_img(image:str, car_center_points, car_width_left_points, car_width_right_points, scaled_lines, vanishing_points, output_text, img_count, result_img_dir, img_colrow):
    ori_img_cols, test_img_cols, ori_img_rows, test_img_rows = img_colrow
    resize_img = cv2.resize(image, (ori_img_cols,ori_img_rows))
    overlay = resize_img.copy()
    
    color = [[255,255,255], [255,0, 51], [0,0,255], [0,255,255]]

    overlay = draw_lines(overlay, car_width_left_points, color[1])
    overlay = draw_lines(overlay, car_width_right_points, color[1])
    overlay = draw_lines(overlay, car_center_points, color[0])
    overlay = draw_lines2(overlay, scaled_lines, color[2])
    overlay = draw_lines(overlay, vanishing_points, color[3])
    
    alpha = 0.5 # 투명도 조절
    result_img = cv2.addWeighted(overlay, alpha, resize_img, 1-alpha, 0)
    
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, 
             cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_ITALIC]
    
    if scaled_lines:
        point = 1600, 50
        result_img = cv2.putText(result_img, output_text[0], point, fonts[0], 1, [255,255,255], 2, cv2.LINE_AA)
        point = 1700, 50
        result_img = cv2.putText(result_img, output_text[1], point, fonts[0], 1, [255,255,255], 2, cv2.LINE_AA)
        point = 1600, 100
        result_img = cv2.putText(result_img, output_text[2], point, fonts[0], 1, [255,255,255], 2, cv2.LINE_AA)
        point = 1700, 100
        result_img = cv2.putText(result_img, output_text[3], point, fonts[0], 1, [255,255,255], 2, cv2.LINE_AA)
    else :
        point = 1600, 50
        result_img = cv2.putText(result_img, output_text[0], point, fonts[0], 1, [255,255,255], 2, cv2.LINE_AA)

    result_img_dir = os.path.join(result_img_dir, str(img_count)+'.jpg')
    cv2.imwrite(result_img_dir, result_img)

    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img

def get_vanishing_line(cam_param)->list:
    world_coord = [[0], [0], [0], [1]]
    img_x = []
    img_y = []
    for world_X in range(-10,10,2):
        world_Y = 10   # y 방향으로 충분히 먼거리 지정 --> 소실라인 확인  
        world_Z = 0 
        world_coord[0] = [world_X]
        world_coord[1] = [world_Y]
        world_coord[2] = [world_Z]
        img_coord = trans_world2img(world_coord, cam_param)
        img_x.append(img_coord[0])
        img_y.append(img_coord[1])
    
    fit_linear_line = get_linear_fit_line(img_x, img_y)
    return fit_linear_line

def trans_world2img(world_coord, cam_param)->list:
    img_coord = cam_param * world_coord
    img_coord = img_coord/img_coord[2]  # [zu,zv,z] -> [u,v,1]
    img_coord = img_coord.flatten().tolist()[0]
    return img_coord

def img2world(img_coord, cam_param):
    world_coord = cam_param.I * img_coord
    world_coord = world_coord/world_coord[2] # [aX, aY, a] -> [X, Y, 1]
    world_XY = world_coord.flatten().tolist()[0]
    world_X = world_XY[0]
    world_Y = world_XY[1]
    return world_X, world_Y

def transform_2d_coord(data, rotate_matrix):
    ori_coord = [[0],[0],[1]]
    transformed_coord = []
    for i in range(0,len(data),2):
        ori_coord[0] = [data[i]]
        ori_coord[1] = [data[i+1]]
        temp_coord = rotate_matrix * ori_coord
        transformed_coord.append(temp_coord[0,0])
        transformed_coord.append(temp_coord[1,0])
    return transformed_coord

def draw_lines(img, line, color):
    line = [round(x) for x in line]
    for i in range(0,len(line),2):
        if len(line) - 2 <= i :
            break
        else :
            img = cv2.line(img, (line[i], line[i+1]), (line[i+2], line[i+3]), color=color, thickness=5)
    return img

def draw_lines2(img, line, color):
    line = [np.round(x).tolist() for x in line]
    for j in range(0,len(line)):
        for i in range(0,len(line[j]),2):
            if len(line[j]) - 2 <= i :
                break
            else :
                img = cv2.line(img, (int(line[j][i]), int(line[j][i+1])), (int(line[j][i+2]), int(line[j][i+3])), color=color, thickness=5)
    return img
