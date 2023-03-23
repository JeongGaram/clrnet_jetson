import os, time, sys, cv2, warnings, math, threading
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from IPython.display import Image
from IPython import display
import argparse

window_title = "Odyssey"

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
]

class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1],
                                                     points[:, 0],
                                                     k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}
        '''
        self.sample_y = range(710, 150, -10) # tusimple
        self.ori_img_w = 1280
        self.ori_img_h = 720
        '''
        self.sample_y = range(589, 230, -20) # culane
        self.ori_img_w = 1640
        self.ori_img_h = 590

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) |
                (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def to_array(self):
        sample_y = self.sample_y
        img_w, img_h = self.ori_img_w, self.ori_img_h
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                              axis=1)
        return lane

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration



class CLRNetDemo():
    def __init__(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        #self.ort_session = onnxruntime.InferenceSession(model_path, )
        self.ort_session = onnxruntime.InferenceSession(model_path, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.conf_threshold = 0.4
        self.nms_thres = 50
        self.max_lanes = 4
        self.sample_points = 36
        self.num_points = 72
        self.n_offsets = 72
        self.n_strips = 71
        '''
        self.img_w = 1280   # tusimple
        self.img_h = 720
        self.ori_img_w = 1280
        self.ori_img_h = 720
        '''
        self.img_w = 1640   # culane
        self.img_h = 590
        self.ori_img_w = 1640
        self.ori_img_h = 590


        self.cut_height = 270

        self.input_width = 800
        self.input_height = 320

        self.sample_x_indexs = (np.linspace(0, 1, self.sample_points) * self.n_strips)
        self.prior_feat_ys = np.flip((1 - self.sample_x_indexs / self.n_strips))
        self.prior_ys = np.linspace(1,0, self.n_offsets)
    
    def softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)


    def Lane_nms(self, proposals,scores,overlap=50, top_k=4):
        keep_index = []
        sorted_score = np.sort(scores)[-1] # from big to small 
        indices = np.argsort(-scores) # from big to small 
        
        r_filters = np.zeros(len(scores))

        for i,indice in enumerate(indices):
            if r_filters[i]==1: # continue if this proposal is filted by nms before
                continue
            keep_index.append(indice)
            if len(keep_index)>top_k: # break if more than top_k
                break
            if i == (len(scores)-1):# break if indice is the last one
                break
            sub_indices = indices[i+1:]
            for sub_i,sub_indice in enumerate(sub_indices):
                r_filter = self.Lane_IOU(proposals[indice,:],proposals[sub_indice,:],overlap)
                if r_filter: r_filters[i+1+sub_i]=1 
        num_to_keep = len(keep_index)
        keep_index = list(map(lambda x: x.item(), keep_index))
        return keep_index, num_to_keep
    
    def Lane_IOU(self, parent_box, compared_box, threshold):
        '''
        calculate distance one pair of proposal lines
        return True if distance less than threshold 
        '''
        n_offsets=72
        n_strips = n_offsets - 1

        start_a = (parent_box[2] * n_strips + 0.5).astype(int) # add 0.5 trick to make int() like round  
        start_b = (compared_box[2] * n_strips + 0.5).astype(int)
        start = max(start_a,start_b)
        end_a = start_a + parent_box[4] - 1 + 0.5 - (((parent_box[4] - 1)<0).astype(int))
        end_b = start_b + compared_box[4] - 1 + 0.5 - (((compared_box[4] - 1)<0).astype(int))
        end = min(min(end_a,end_b),71)
        
        if (end - start)<0:
            return False
        dist = 0
        
        if type(end) == float:
            for i in range(5+start,5 + end.astype(int)):
                if i>(5+end):
                     break
                if parent_box[i] < compared_box[i]:
                    dist += compared_box[i] - parent_box[i]
                else:
                    dist += parent_box[i] - compared_box[i]
        else :
            for i in range(5+start,5 + int(end)):
                if i>(5+end):
                     break
                if parent_box[i] < compared_box[i]:
                    dist += compared_box[i] - parent_box[i]
                else:
                    dist += parent_box[i] - compared_box[i]
        return dist < (threshold * (end - start + 1))


    def predictions_to_pred(self, predictions):
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       )[::-1].cumprod()[::-1]).astype(np.bool))

            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]

            lane_xs = np.double(lane_xs)
            lane_xs = np.flip(lane_xs, axis=0)
            lane_ys = np.flip(lane_ys, axis=0)
            lane_ys = (lane_ys * (self.ori_img_h - self.cut_height) +
                       self.cut_height) / self.ori_img_h
            if len(lane_xs) <= 1:
                continue

            points = np.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                axis=1).squeeze(2)

            lane = Lane(points=points,
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def get_lanes(self, output, as_lanes=True):
        '''
        Convert model output to lanes.
        '''
        decoded = []
        for predictions in output:
            # filter out the conf lower than conf threshold
            scores = self.softmax(predictions[:, :2], 1)[:, 1]

            keep_inds = scores >= self.conf_threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions

            nms_predictions = np.concatenate(
                [nms_predictions[..., :4], nms_predictions[..., 5:]], axis=-1)
    
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[...,
                            5:] = nms_predictions[..., 5:] * (self.img_w - 1)
            
            
            keep, num_to_keep = self.Lane_nms( 
                nms_predictions,
                scores,
                self.nms_thres,
                self.max_lanes)

            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = np.round(predictions[:, 5] * self.n_strips)
            pred = self.predictions_to_pred(predictions)
            decoded.append(pred)
            
        return decoded
    
    def imshow_lanes(self, img, lanes, show=False, out_file=None, width=4):
        lanes = [lane.to_array() for lane in lanes]
        lanes_xys = []

        if not lanes:
            return img, lanes_xys
        else:
            for _, lane in enumerate(lanes):
                xys = []
                for x, y in lane:
                    if x <= 0 or y <= 0:
                        continue
                    x, y = int(x), int(y)
                    xys.append((x, y))
                if xys:
                    lanes_xys.append(xys)
                else:
                    continue
            
            lanes_xys.sort(key=lambda xys : xys[0][0])
            
            for i, xys in enumerate(lanes_xys): ## 필요없는 라인 삭제
                if len(xys) ==1 :
                    del lanes_xys[i]

            # for idx, xys in enumerate(lanes_xys):
            #     for i in range(1, len(xys)):
            #         cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
            return img, lanes_xys
   
    def forward(self, img):
        img_ = img.copy()
        h, w = img.shape[:2]
        img = img[self.cut_height:, :, :]
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = img.astype(np.float32) / 255.0 

        img = np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1))

        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0]
        
        output = self.get_lanes(output)
        res, lanes_xys = self.imshow_lanes(img_, output[0])
        return res, lanes_xys

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

def load_lines(lanes_xys)->list:
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

def prepare_img(image:str, car_center_points, car_width_left_points, car_width_right_points, scaled_lines, vanishing_points, output_text, img_count, result_img_dir):
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
        result_img = cv2.putText(result_img, output_text[0], point, fonts[0], 1, [0,0,0], 2, cv2.LINE_AA)
        point = 1700, 50
        result_img = cv2.putText(result_img, output_text[1], point, fonts[0], 1, [0,0,0], 2, cv2.LINE_AA)
        point = 1600, 100
        result_img = cv2.putText(result_img, output_text[2], point, fonts[0], 1, [0,0,0], 2, cv2.LINE_AA)
        point = 1700, 100
        result_img = cv2.putText(result_img, output_text[3], point, fonts[0], 1, [0,0,0], 2, cv2.LINE_AA)
    else :
        point = 1600, 50
        result_img = cv2.putText(result_img, output_text[0], point, fonts[0], 1, [0,0,0], 2, cv2.LINE_AA)

    # thread로 변경?
    t3 = threading.Thread(target = img_write_thread, args=(result_img, result_img_dir, img_count))
    # result_img_dir = os.path.join(result_img_dir, str('{0:05d}'.format(img_count) + '.jpg'))
    # cv2.imwrite(result_img_dir, result_img)
    #cv2.imshow("runtime",result_img)
    t3.start()
    # result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
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

def run(image, img_count, original_img_dir, result_img_dir):
    
    # 이미지 resize, 라인정보 획득
    image = cv2.resize(image, (test_img_cols, test_img_rows))
    image, lanes_xys = clr.forward(image)
    image = cv2.resize(image, (ori_img_cols, ori_img_rows))
    
    # 원본 이미지 저장 : thread로 변경 필요.
    t2 = threading.Thread(target=img_write_thread, args=(image, original_img_dir, img_count))
    # original_img_dir = os.path.join(original_img_dir, str('{0:05d}'.format(img_count)+'.jpg'))
    # cv2.imwrite(original_img_dir, image)
    t2.start()
    if len(lanes_xys) == 1:
        lanes_xys = [lanes.flatten().tolist() for lanes in np.array(lanes_xys)]
    else:
        lanes_xys = [np.array([lanes]).flatten().tolist() for lanes in lanes_xys]
        
    # 이미지 번호, 라인 정보 출력
    print(img_count, lanes_xys)
    
    # 카메라 정보 획득
    cam_e = get_cam_e()
    cam_param = cam_i * cam_e

    ## 소실라인 확인 및 라인 피팅
    vanishing_line = get_vanishing_line(cam_param=cam_param)
    vanishing_points = []
    for x in range(0, ori_img_cols+96, 96):
        vanishing_points.append(x)
        vanishing_points.append(vanishing_line[0]*x + vanishing_line[1])
        
    ## 기본 라인 정보 확인
    car_center_vanishing_point = [car_center_x, round(vanishing_line[0] * car_center_x + vanishing_line[1])]
    car_center_points = get_line_point(start_point=[car_center_x, 1080] , end_point=car_center_vanishing_point)
    car_width_right_points = get_line_point(start_point=[car_width_right_x, 1080] , end_point=car_center_vanishing_point)
    car_width_left_points = get_line_point(start_point=[car_width_left_x, 1080] , end_point=car_center_vanishing_point)
    
    ## extract lines
    scaled_lines = load_lines(lanes_xys)
    if not scaled_lines:
        scaled_lines = []
        output_text = ['No line is detected!']
        result_img = prepare_img(image, car_center_points, car_width_left_points, car_width_right_points, scaled_lines, vanishing_points, output_text, img_count, result_img_dir)
       
    else:
        ## get nearest line
        nearest_line = get_nearest_point(scaled_lines=scaled_lines, ref_point=car_center_x)

        ## get nearest side (0 : left, 1 : right)
        car_width_x = [[car_width_left_x],[car_width_right_x]]
        nearest_side = get_nearest_point(scaled_lines=car_width_x, ref_point=scaled_lines[nearest_line][0])

        ## ignore world Z axis
        cam_e_ignore_z_axis = np.delete(cam_e, 2, axis=1) 
        cam_param_ignore_z_axis = cam_i * cam_e_ignore_z_axis

        ## get img to world coordinates
        car_width_points = [car_width_left_points, car_width_right_points]
        world_car_center_line = get_world_coord(line=car_center_points, cam_param=cam_param_ignore_z_axis)
        world_ref_line = get_world_coord(line=car_width_points[nearest_side], cam_param=cam_param_ignore_z_axis)
        world_extracted_line = get_world_coord(line=scaled_lines[nearest_line], cam_param=cam_param_ignore_z_axis)

        ## 2d axis transformation
        if world_car_center_line[0] < 0 :
            rotate_matrix = np.mat([[math.cos(math.pi), -1*math.sin(math.pi),  0],
                                    [math.sin(math.pi),    math.cos(math.pi), -1],
                                    [                0,                    0,  1]])
            world_car_center_line = transform_2d_coord(data=world_car_center_line, rotate_matrix=rotate_matrix)
            world_ref_line = transform_2d_coord(data=world_ref_line, rotate_matrix=rotate_matrix)
            world_extracted_line = transform_2d_coord(data=world_extracted_line, rotate_matrix=rotate_matrix)

        ## plot line
        # plot_img(world_car_center_line[:30], 'o')
        # plot_img(world_ref_line[:30], '*')
        # plot_img(world_extracted_line[:20], 's')

        ## get distance
        car2ref_distance = get_distance(world_car_center_line[:2], world_ref_line[:2])
        ref2line_distance = get_distance(world_ref_line[:2], world_extracted_line[:2])

        ## get angle 
        world_ref_start2end_points_X = [world_ref_line[0], world_ref_line[-2]] 
        world_ref_start2end_points_Y = [world_ref_line[1], world_ref_line[-1]]
        ref_line_fit = get_linear_fit_line(world_ref_start2end_points_X, world_ref_start2end_points_Y)

        world_extracted_start2end_points_X = [world_extracted_line[0], world_extracted_line[-2]]
        world_extracted_start2end_points_Y = [world_extracted_line[1], world_extracted_line[-1]]
        extracted_line_fit = get_linear_fit_line(world_extracted_start2end_points_X, world_extracted_start2end_points_Y)

        ref_line_angle = math.degrees(math.atan(ref_line_fit[0]))
        extracted_line_angle = math.degrees(math.atan(extracted_line_fit[0]))

        if extracted_line_angle < 0 :
            extracted_line_angle = 180 + extracted_line_angle
        if ref_line_angle < 0 :
            ref_line_angle = 180 + ref_line_angle

        ## direction
        if scaled_lines[nearest_line][0] - car_width_points[nearest_side][0] >= 0 :
            direction = '오른쪽'
        else : 
            direction = '왼쪽'
        direction_value = f'{round(ref2line_distance/car2ref_distance, 2)}'

        ## angle
        if ref_line_angle - extracted_line_angle > 0 :
            angle = '오른쪽'
        elif ref_line_angle - extracted_line_angle < 0: 
            angle = '왼쪽'
        else : 
            angle = '직진'
        text_angle = f'{round(abs(ref_line_angle-extracted_line_angle), 2)}'

        direction = 'Right' if direction == '오른쪽' else 'Left'
        angle = 'Right' if angle == '오른쪽' else 'Left'
        output_text = [direction, direction_value+' meter', angle, text_angle+' degree']

        ## draw lines and save image file
        # os.makedirs('../result_images/'+input_seq_name, exist_ok=True)
        # img_path = os.path.join(input_seq_name, input_name+'.jpg')
        
        
        # save img count
        result_img = prepare_img(image, car_center_points, car_width_left_points, car_width_right_points, scaled_lines, vanishing_points, output_text, img_count, result_img_dir)

        # end = time.time()
        # lap_time = end-start
        # total_elapsed_time = total_elapsed_time + lap_time
        # count += 1
        # print(f'{lap_time:.5f} sec')  ## include
        # print('')  ## include
        #end of check line if

    # 이미지 출력     
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    #plt.imshow(result_img)
    #plt.show()
    #plt.close()
    #cv2.imshow("Odyssey", result_img)
    cv2.imshow(window_title, result_img)

def img_write_thread(image, img_dir, count):
    count_num = str('{0:05d}'.format(count))
    img_dir = os.path.join(img_dir, count_num + '.jpg')
    cv2.imwrite(img_dir, image)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-o', nargs='*', help='Example) --input camera or --input mp4', default=['camera'], dest='input')

    input_type = parser.parse_args().input

    return input_type

if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    input_type = get_arguments()
    print("Input Type : ", input_type)

    
    ## 기본정보
    # intrinsic matrix of camera (임의로 지정함)
    cam_i = np.mat([[1920,    0,  960, 0], 
                    [   0, 1080,  540, 0],
                    [   0,    0,    1, 0]])
    
    ## 이미지 중심라인, 주행라인, 차폭라인 (임의로 지정함)
    img_center_x = round(cam_i[0,2])
    img_center_y = round(cam_i[1,2])
    car_center_x = 900 
    img_car_width = 600
    car_width_right_x = car_center_x + img_car_width/2
    car_width_left_x = car_center_x - img_car_width/2
    
    ## 원본 이미지, 테스트 이미지 사이즈
    ori_img_cols, ori_img_rows = 1920, 1080
    test_img_cols, test_img_rows = 1640, 590 

    # video or image 입출력 폴더
    # ori_video_dir = './data/video/20221208_143517.mp4' 
    file_name = 'IMG_2459_test.mp4'
    ori_video_dir = f'./{file_name}' 
    # ori_video_dir = './data/video/video.mp4' 
    result_img_dir = f'./results/real_time/{file_name}/test' # image save directory
    original_img_dir = f'./results/real_time/{file_name}/original_image'
    real_time_img_dir = f'./results/real_time/{file_name}/real_time_image'
    os.makedirs(result_img_dir, exist_ok=True)
    os.makedirs(original_img_dir, exist_ok=True)
    os.makedirs(real_time_img_dir, exist_ok=True)

    #Camera Gstreamer Pipeline
    pipeline = " !".join(["v4l2src device=/dev/video0 io-mode=2",
                        "image/jpeg , width=(int)1920, height=(int)1080, framerate=30/1",
                        "nvv4l2decoder mjpeg=1",
                        "nvvidconv",
                        "video/x-raw,format=BGRx",
                        "videoconvert",
                        "video/x-raw, format=BGR",
                        "appsink"
                        ])

    
    # video capture (cap_seconds 시간당 한장)
    cap_seconds = 0.05 #5

    if input_type[0] == 'camera':
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        print('FPS: ', cap_fps)
    else:
        if input_type[0] == 'mp4':
            cap = cv2.VideoCapture(ori_video_dir)
            cap_fps = 30
            print('FPS: ', cap_fps)
        else:
            sys.exit("need --input camera or --input mp4")

    #cap_fps = 30
    cap_multiplier = cap_fps * cap_seconds
    #plt.ion() 

    # CLRNet class 호출, onnx file도 arg로 받도록 수정 필요. 
    clr = CLRNetDemo('./49_230127.onnx')
    
    # save image count
    img_count = 0
    real_img_count = 0
    # frame
    frame = 30
   
    window_handle = cv2.namedWindow(
                window_title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # 
    start = time.time()
    ## 실시간 라인추출 및 각도, 거리 계산
    while(cap.isOpened()):
        #start = time.time()
        cap_frameId = int(round(cap.get(1)))
        ret, image = cap.read()
        if not ret: # 새로운 프레임을 못받아 왔을 때 braek
            break

        # thread : original image write 
        t1 = threading.Thread(target = img_write_thread, args = (image, real_time_img_dir, real_img_count))
        t1.start()
        # print('cap.get(1): ', cap.get(1))
        #if(int(cap.get(1)) % frame == 0):
        #if(int(cap.get(8)) % frame == 0):
        #print('cap_multiplier : ', cap_multiplier)
        #print('cap_frameId % cap_multiplier : ', int(round(cap_frameId % cap_multiplier)))
        
        if int(round(cap_frameId % cap_multiplier)) == 0:
            start = time.time()
            #print('cap_frameId :', cap_frameId)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(img_count)
            run(image, img_count, original_img_dir, result_img_dir)
            
            img_count += 1

            end = time.time()
            print(f"{(end-start):.4f} sec, fps : {(1/(end-start)):.4f}")
            #print(f"{(end-start)/img_count:.4f} sec, fps : {(1/((end-start)/img_count)):.4f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        real_img_count += 1
    #end = time.time()
    #print(f"{end-start:.4f} sec")

    cap.release()
    cv2.destroyAllWindows()
