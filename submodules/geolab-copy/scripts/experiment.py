import sys
import os
WORK_SPACE = os.getcwd()
sys.path.insert(0, os.path.join(WORK_SPACE, 'build'))
import pygeo
import numpy as np
import pandas as pd
import imageio
import lpips 
import metrics
import torch

render_types = [
    'NORMONLY',
    'LATTICE',
    'POSITION',
    'INSTANCE_AO',
    'DEPTH'
]

resolution_types = [
    [256, 256],
    [512, 512],
    [1024, 1024]
]

dataset_types = [
    'obj_prop_all', 'obj_prop_t',
    'dense_lod1_all', 'dense_lod2_all', 'dense_lod3_all', 'dense_lod4_all', 'dense_lod5_all',
    'dense_lod3_t', 'dense_lod4_t', 'dense_lod5_t',
    'NI_prop_all', 'NI_prop_t',
    'ours_632_all', 'ours_632_t', 
    'sparse_lod1_all', 'sparse_lod2_all', 'sparse_lod3_all', 'sparse_lod4_all', 'sparse_lod5_all'
]

# PATH STRUCTURE
# payloads {}/payloads/{dataset_type}_{resolution[0]}/{obj_name}/frame_{n}.npz
# pictures {}/pictures/{dataset_type}_{resolution[0]}/{shader_type}/{obj_name}/picture_{n}.png

def print_picture(enable_post_process = False):
    # input path type {workspace}/payloads/{dataset_type}/{obj_name}(/{frame_n.npz})
    # output paht type {workspace}/pictures/{dataset_type}/{shader_type}/{obj_name}/{picture_n.png}
    resolution = [512, 512]

    camera_pos = pygeo.sample_fibonacci(50)
    camera_pos = np.asarray(camera_pos) * 2.8 * 1.73205080756887729

    for dataset_type in [
        'obj_prop_t',
        'obj_prop_all'
        ]:

        ws = dataset_type.split('_')
        if ws[0] == 'obj':
            renderer = pygeo.SDFRenderer()
        else:
            resolution = [int(ws[-1]), int(ws[-1])]
            renderer = pygeo.SDFRenderer()
        
        # Light Configuration
        renderer.scene.slice_plane_z = 1.0
        renderer.scene.surfaceColor = [1.0, 1.0, 1.0]
        renderer.light.specular = 0.3
        renderer.light.kd = 0.8

        for shader_type in ['normonly', 'lattice']:
            shader = pygeo.ERenderMode.NORMONLY if shader_type == 'normonly' else pygeo.ERenderMode.LATTICE

            if ws[0] == 'obj':
                input_prefix = '/media/cscvlab/DATA/project/dray/dcc_payload_picture_metrics/payloads/{}'.format(dataset_type) 
                output_prefix = '/media/cscvlab/DATA/project/dray/dcc_payload_picture_metrics/pictures/{}_{}/{}'.format(dataset_type, resolution[0], shader_type)
            else:
                input_prefix = '/media/cscvlab/DATA/project/dray/dcc_payload_picture_metrics/payloads/{}'.format(dataset_type)
                output_prefix = os.path.join('/media/cscvlab/DATA/project/dray/dcc_payload_picture_metrics/pictures/', dataset_type, shader_type)

            if ws[0] == 'obj':
                for m in os.listdir(input_prefix):
                    obj_name = m.split('.')[0]
                    if obj_name not in ['1']: # selected objs
                        print('Rendering ', m)
                        obj_path = os.path.join(input_prefix, m)
                        output_path = os.path.join(output_prefix, obj_name)
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)

                        pictures = renderer.render_ray_trace(obj_path, resolution, shader, camera_pos)
                        for i in range(camera_pos.shape[0]):
                            picture = np.asarray(pictures[i]).reshape((resolution[0], resolution[1], 4))
                            picture = np.asarray(picture * 255, dtype=np.uint8)
                            imageio.imwrite(output_path + '/{}.png'.format(i), picture)
            else:
                for m in os.listdir(input_prefix):
                    print('Rendering ', m)
                    input_path = os.path.join(input_prefix, m)
                    output_path = os.path.join(output_prefix, m)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    for i in range(camera_pos.shape[0]):
                        frame = np.load(input_path + '/frame_{}.npz'.format(i))
                        win_res = [frame['arr_0'].shape[0], frame['arr_0'].shape[1]]
                        rows = win_res[0] * win_res[1]
                        
                        points = frame['arr_0'].reshape((rows, 3))
                        normals = frame['arr_1'].reshape((rows, 3))
                        hits = frame['arr_2'].reshape((rows, 1))
                        n_steps = frame['arr_3'].reshape((rows, 1))
                        dis = frame['arr_4'].reshape((rows, 1))
                        payload = pygeo.load_sdfpayload(points, normals, hits, n_steps, dis, win_res)
                        picture = renderer.render(payload, shader, camera_pos[i])
                        picture = np.asarray(picture, dtype=np.uint8).reshape((win_res[0], win_res[1], 4))
                        imageio.imwrite(output_path + '/{}.png'.format(i), picture)

def run_metrics():

    def group_preprocess(pictures):
        result = []
        for i in range(len(pictures)):
            img = torch.from_numpy(pictures[i])
            img = img.float()/255
            result.append(img)
        return result

    def group_psnr(pictures_gt, pictures_sdf):
        result = []
        for i in range(len(pictures_gt)):
            result.append(metrics.compute_psnr(pictures_gt[i], pictures_sdf[i]))
        return result

    def group_ssim(pictures_gt, pictures_sdf):
        result = []
        for i in range(len(pictures_gt)):
            result.append(metrics.compute_ssim(pictures_gt[i], pictures_sdf[i]).item())
        return result

    def group_lpips(vgg, pictures_gt, pictures_sdf, device = 'cuda:0'):
        """
        The value of color is [0,255]^R
        """
        result = []
        for i in range(len(pictures_gt)):
            lpips_ = lpips_vgg(pictures_gt[i].permute([2,0,1]).cuda().contiguous(),
                               pictures_sdf[i].permute([2,0,1]).cuda().contiguous(),
                               normalize=True).item()
            result.append(lpips_)
        return result
            
    gt_type = 'obj'
    # output_path ./metrics/{dataset_type}/{shader_type}/{obj_name}/metrics.csv
    # avg_path ./metrics/{dataset_type}/{shader_type}/{obj_name}/avg.csv
    # total_path ./metrics/total_{normonly/lattice}.csv
    metrics_path = '../dcc_payload_picture_metrics/metrics'
    total = pd.DataFrame(columns=['dataset_type', 'shader_type', 'obj', 'frame', 'psnr', 'ssim', 'lpips'])
    dataset_types = [
        'dense_lod1_all_256',
        'dense_lod2_all_256',
        'dense_lod3_all_256',
        'dense_lod3_t_256',
        'dense_lod4_all_256',
        'dense_lod4_t_256',
        'dense_lod5_all_256',
        'dense_lod5_t_256',
        'NI_prop_all_256',
        'NI_prop_t_256',
        'ours_632_all_256',
        'ours_632_t_256', 
        'sparse_lod1_all_256', 
        'sparse_lod2_all_256', 
        'sparse_lod3_all_256', 
        'sparse_lod4_all_256', 
        'sparse_lod5_all_256'
    ]
    shader_types = ['normonly', 'lattice']

    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

    for dataset_type in dataset_types:
        words = dataset_type.split('_')
        res = words[3]
        type_ = words[2]
        gt_type = 'obj_prop_{}_{}'.format(type_, res)
        
        for shader_type in shader_types:
            input_prefix = os.path.join('../dcc_payload_picture_metrics/pictures', dataset_type, shader_type)
            for m in os.listdir(input_prefix):
                print(dataset_type, shader_type, m, '>>>>>>>>>>>>>>>>>>>>>>>')
                pictures_batch = []
                for p in os.listdir(input_prefix + '/' + m):
                    pictures_batch.append(imageio.imread(os.path.join(input_prefix, m, p))[:, :, 0:3])
                benchmark_batch = []
                for p in os.listdir(os.path.join('../dcc_payload_picture_metrics/pictures', gt_type, shader_type, m)):
                    benchmark_batch.append(imageio.imread(os.path.join('../dcc_payload_picture_metrics/pictures', gt_type, shader_type, m, p))[:, :, 0:3])

                pictures_gt, pictures_sdf = group_preprocess(pictures_batch), group_preprocess(benchmark_batch)
                psnr_g = group_psnr(pictures_gt, pictures_sdf)
                ssim_g = group_ssim(pictures_gt, pictures_sdf)
                lpips_g = group_lpips(lpips_vgg, pictures_gt, pictures_sdf)

                for i in range(len(psnr_g)):
                    total = total.append({
                        'dataset_type': dataset_type,
                        'shader_type' : shader_type,
                        'obj': m,
                        'frame': i,
                        'psnr' : psnr_g[i],
                        'ssim' : ssim_g[i],
                        'lpips' : lpips_g[i]
                    }, ignore_index=True)
    
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    total.to_csv(os.path.join(metrics_path, 'total_256.csv'))
    total_avg = pd.DataFrame(columns=['dataset_type', 'shader_type', 'psnr', 'ssim', 'lpips'])
    for dataset_type in dataset_types:
        for shader_type in shader_types:
            tmp = total[(total.dataset_type == dataset_type) & (total.shader_type == shader_type)].mean(axis=0)
            total_avg = total_avg.append({
                'dataset_type' : dataset_type,
                'shader_type' : shader_type,
                'psnr' : tmp['psnr'],
                'ssim' : tmp['ssim'],
                'lpips' : tmp['lpips']
            }, ignore_index=True)
    total_avg.to_csv(os.path.join(metrics_path,'total_avg_256.csv'))



if __name__ == '__main__':
    print_picture()