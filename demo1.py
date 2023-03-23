import cv2
import numpy as np
import torch
import sys
import os
from clrnet.models.nets.detector import Detector
from clrnet.utils import Config
import argparse
from clrnet.utils.net_utils import load_network
import torch.backends.cudnn as cudnn
from clrnet.ops import nms_impl
import timeit
from scipy.interpolate import InterpolatedUnivariateSpline
import time
from angle_calc import *

device = torch.device("cuda:0")

def parse_args():
    parser = argparse.ArgumentParser(
        description='CLRNet models cam demo from pytorch')

    parser.add_argument('--cfg',
                        type=str,
                        default='configs/clrnet/clr_resnet18_culane.py',
                        help='Filename of input torch model')

    parser.add_argument('--load_from',
                        default= '14.pth',
                        help='Filename of pretrained model path')

    parser.add_argument('--video',
                    type=str,
                    # default='daedong_test_image.jpg',
                    default='IMG_2459.mp4',
                    help='Filename of input image or video or cam')

    parser.add_argument('--gpus',
                        nargs='+',
                        default='0',
                        type=int,
                        help='Available number of gpus')

    args = parser.parse_args()
    return args

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]

class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1],
                                                     points[:, 0],
                                                     k=min(3,
                                                           len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}
        
        if False:
            self.sample_y = range(710, 150, -10)
            self.ori_img_w = 1280
            self.ori_img_h = 720
        
        else:
            self.sample_y = range(589, 230, -20)
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

def nms(boxes, scores, overlap, top_k):
    return nms_impl.nms_forward(boxes, scores, overlap, top_k)

class CLRNetDemo():
    def __init__(self):
        self.conf_threshold = 0.4
        self.nms_thres = 50
        self.max_lanes = 4
        self.sample_points = 36
        self.num_points = 72
        self.n_offsets = 72
        self.n_strips = 71
        self.img_w = 1640
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
        end = min(min(end_a,end_b),np.float64(71))
        if (end - start)<0:
            return False
        dist = 0
        for i in range(5+start,5 + end.astype(int)):
            if i>(5+end):
                 break
            if parent_box[i] < compared_box[i]:
                dist += compared_box[i] - parent_box[i]
            else:
                dist += parent_box[i] - compared_box[i]
        return dist < (threshold * (end - start + 1))

    
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
                if xys != []:
                    lanes_xys.append(xys)
            try:
                lanes_xys.sort(key=lambda xys : xys[0][0])
            except:
                pass

            for idx, xys in enumerate(lanes_xys):
                if len(xys) == 1:
                    del lanes_xys[idx]
                '''
                for i in range(1, len(xys)):
                    cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
                '''
            return img, lanes_xys

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
                       )[::-1].cumprod()[::-1]).astype(np.bool_))

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


    def before_forward(self, img: np.ndarray):
        img = cv2.resize(img, (self.ori_img_w, self.ori_img_h), cv2.INTER_CUBIC)
        img_ = img.copy()

        img = img[self.cut_height:, :, :]
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)

        img = img.astype(np.float32) / 255.0 
        
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).squeeze(0)
        img = np.reshape(img, ((1,) + img.shape))

        return img, img_

    
    def forward(self, net, img: torch.Tensor):
        # device = torch.device("cuda:0")
        # img = cv2.resize(img, (self.ori_img_w, self.ori_img_h), cv2.INTER_CUBIC)
        # img_ = img.copy()

        # img = img[self.cut_height:, :, :]
        # img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)

        # img = img.astype(np.float32) / 255.0 
        
        # img = torch.from_numpy(img)
        # img = img.permute(2, 0, 1).squeeze(0)
        # img = np.reshape(img, ((1,) + img.shape))
        
        # net = ne.to(device)
        # img = img.to(device)
        
        output = net(img)#.cpu().detach().numpy() ##CPU

        return output
    

    def after_forward(self, model_output: torch.Tensor, img_: np.ndarray):
        model_output = model_output.cpu().detach().numpy()
        output = self.get_lanes(model_output)

        res, lanes_xys = self.imshow_lanes(img_, output[0])

        return res, lanes_xys


def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.cfg)
    cfg.load_from = args.load_from
    net = Detector(cfg)
    net2 = Detector(cfg)
    net.eval()
    net2.eval()
    load_network(net, cfg.load_from, logger=None) #, remove_module_prefix= True
    load_network(net2, cfg.load_from, logger=None) #, remove_module_prefix= True

    clr = CLRNetDemo()
    
    img_name = args.video
    
    #=======================================================================================
    result_img_dir = './results/real_time/result_image' # image save directory
    original_img_dir = './results/real_time/original_image'
    test_img_dir = './results/test'
    os.makedirs(result_img_dir, exist_ok=True)
    os.makedirs(original_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
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
    img_colrow = [ori_img_cols, test_img_cols, ori_img_rows, test_img_rows]
    cam_e = get_cam_e()
    cam_param = cam_i * cam_e
    #=======================================================================================

    if img_name[:].find('mp4') == -1:
        from glob import glob
        img_name = sorted(glob(f'{img_name}/*.jpg'))
        for ii, img_name in enumerate(img_name):
            start = time.time()
            img = cv2.imread(img_name)
            img, img_ori = clr.before_forward(img)

            output = clr.forward(net.to(device), img.to(device))

            output, lanes_xys = clr.after_forward(output, img_ori)
            # ========================================================================
            image = cv2.resize(output, (ori_img_cols, ori_img_rows))

            original_img_dir = os.path.join(original_img_dir, str(ii).zfill(5) + '.jpg')
            cv2.imwrite(original_img_dir, image)
            
            for lanes in lanes_xys:
                lanes = np.expand_dims(np.array(lanes), axis= 0)
                lanes_xys = [lane.flatten().tolist() for lane in lanes]
                break
            #lanes_xys = [lanes.flatten().tolist() for lanes in np.array(lanes_xys)]

            print(ii, lanes_xys)
            vanishing_line = get_vanishing_line(cam_param=cam_param)
            vanishing_points = []
            for x in range(0, ori_img_cols+96, 96):
                vanishing_points.append(x)
                vanishing_points.append(vanishing_line[0]*x + vanishing_line[1])

            car_center_vanishing_point = [car_center_x, round(vanishing_line[0] * car_center_x + vanishing_line[1])]
            car_center_points = get_line_point(start_point=[car_center_x, 1080] , end_point=car_center_vanishing_point)
            car_width_right_points = get_line_point(start_point=[car_width_right_x, 1080] , end_point=car_center_vanishing_point)
            car_width_left_points = get_line_point(start_point=[car_width_left_x, 1080] , end_point=car_center_vanishing_point)

            ## extract lines
            scaled_lines = load_lines(lanes_xys, img_colrow)
            if not scaled_lines:
                scaled_lines = []
                output_text = ['No line is detected!']
                result_img = prepare_img(image, car_center_points, car_width_left_points, car_width_right_points, scaled_lines, vanishing_points, output_text, ii, result_img_dir, img_colrow)
               
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

                
                # save img count
                result_img = prepare_img(image, car_center_points, car_width_left_points, car_width_right_points, scaled_lines, vanishing_points, output_text, ii, result_img_dir, img_colrow)
              
            # plt.imshow(result_img)




            # ========================================================================
            cv2.imwrite(f'./results/test/output_video{str(ii).zfill(5)}.png', output)
            end = time.time()
            print(end - start, "sec")
    else:
        stream = torch.cuda.Stream()

        cap = cv2.VideoCapture(img_name)

        ii = 0
        while(cap.isOpened()):
            start = time.time()
            ret, image = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            img, img_ori = clr.before_forward(image)

            with torch.cuda.stream(stream):
                output = clr.forward(net.to(device), img.to(device))
                output2 = clr.forward(net2.to(device), img.to(device))
            
            torch.cuda.synchronize()

            output, lanes_xys = clr.after_forward(output, img_ori)
            output2, lanes_xys2 = clr.after_forward(output2, img_ori)

            # output, lanes_xys = clr.forward(net, image)
            end = time.time()
            print(end - start, " sec ", (1/(end - start)), "fps", lanes_xys)
            cv2.imwrite(f'./results/test/output_video{str(ii).zfill(5)}.png', output)
            ii += 1

        cap.release()


    #img = cv2.imread('./daedong_test_image.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #output = clr.forward(net, img)
    
    #cv2.imwrite('output_pth.png', output)
    print("Done!")

if __name__ == '__main__':
    main()
