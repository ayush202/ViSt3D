import argparse, gc
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imageio
from torchvision import transforms

import util
import time
import warnings

warnings.filterwarnings("ignore")
torch.set_num_threads(18)


def style_transfer2(decoder, content_feats, style_feats, device, args, use_sigmoid):
    # Load weights of mapping layers
    args.disentangle_appearance0 = util.disentangle()
    args.disentangle_appearance1 = util.disentangle1()
    args.disentangle_appearance2 = util.disentangle2()
    args.disentangle_appearance3 = util.disentangle3()
    state_dict4 = torch.load(args.disentangle_appearance)
    args.disentangle_appearance0.model.load_state_dict(state_dict4['disentangle'])
    args.disentangle_appearance1.model.load_state_dict(state_dict4['disentangle1'])
    args.disentangle_appearance2.model.load_state_dict(state_dict4['disentangle2'])
    args.disentangle_appearance3.model.load_state_dict(state_dict4['disentangle3'])


    with torch.no_grad():
        content_feats_appearance = []
        for i in range(len(content_feats)):
            func = getattr(args, f'disentangle_appearance{3-i}')
            feat = func(content_feats[i], device)
            func.model.cpu()
            content_feats[i].cpu()
            content_feats_appearance.append(feat.cpu())

        gc.collect()
        torch.cuda.empty_cache()
        
        if not args.use_style_appearance:
            for i in range(len(style_feats)):
                func = getattr(args, f'disentangle_appearance{3 - i}')
                feat = func(style_feats[i], device).cpu()
                func.model.cpu()
                style_feats[i].cpu()
                args.style_feats_appearance.append(feat.cpu())
                args.use_style_appearance = True

        gc.collect()
        torch.cuda.empty_cache()
        adain_feats_appearance = []
        for i in range(len(content_feats_appearance)):
            feat = util.adaptive_instance_normalization1_3D(content_feats_appearance[i], args.style_feats_appearance[i])
            content_feats_appearance[i].cpu()
            args.style_feats_appearance[i].cpu()
            adain_feats_appearance.append(feat.cpu())
        
        gc.collect()
        torch.cuda.empty_cache()
       
        concatenated_tensor = torch.concat((content_feats[-1], adain_feats_appearance[-1].cpu()), dim=1)
        concatenated_tensor = entangle_feats(concatenated_tensor, device)
        entangle_feats.model.cpu()
        concatenated_tensor.cpu()

        feat = concatenated_tensor.to(device)#adain_feats_appearance[-1].to(device)
        dec1_gpu = decoder.model1_part0.to(device)
        feat = dec1_gpu(feat)
        dec1_gpu.cpu()
        del dec1_gpu
        concatenated_tensor.cpu()
        del concatenated_tensor
        gc.collect()
        torch.cuda.empty_cache()
        
        

        feat = torch.concat((feat, adain_feats_appearance[2].to(device)), dim=1)
        dec1_gpu = decoder.model1_part1.to(device)
        feat = dec1_gpu(feat)
        dec1_gpu.cpu()
        adain_feats_appearance[2].cpu()
        del dec1_gpu
        del adain_feats_appearance[2]
        gc.collect()
        torch.cuda.empty_cache()
        
        device = 'cpu'
        feat = feat.to(device)

        feat = torch.concat((feat, adain_feats_appearance[1].to(device)), dim=1)
        adain_feats_appearance[1].cpu()
        del adain_feats_appearance[1]
        dec_layers = list(decoder.model1_part2.children())
        #print(f'feat {feat.shape}')
        #exit(0)
        dec1 = nn.Sequential(*dec_layers[:7])
        dec2 = nn.Sequential(*dec_layers[7:])
        dec1_gpu = dec1.to(device)
        feat = dec1_gpu(feat)
        dec1_gpu.cpu()
        del dec1_gpu
        gc.collect()
        torch.cuda.empty_cache()
        #feat = feat.cpu()
        dec2_gpu = dec2.to(device)
        feat = dec2_gpu(feat)
        dec2_gpu.cpu()
        del dec2_gpu
        gc.collect()
        torch.cuda.empty_cache()

        
        #device = 'cpu'
        #feat = feat.to(device)
        
        
        feat = torch.concat((feat, adain_feats_appearance[0].to(device)), dim=1)
        adain_feats_appearance[0].cpu()
        del adain_feats_appearance[0]

        decoder.model1_part3.to(device)
        feat = decoder.model1_part3(feat).cpu()
        decoder.model1_part3.cpu()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if use_sigmoid:
          feat = nn.Sigmoid()(feat)

        return feat


def create_style_clip(path, frame_count, output_height, output_width):
    style_img = np.array(Image.open(path).convert('RGB'))
    style_imgs = []
    rot_deg = 0

    for i in range(frame_count):
        style = np.copy(style_img)
        
        data_transform = transforms.Resize((output_height, output_width), interpolation=Image.BICUBIC)
        
        style = transforms.ToTensor()(style).to(device)
        style = data_transform(style)

        style_imgs.append(style)

    return torch.stack(style_imgs)


def encode_with_intermediate1(input, c3d_model, device, style=False):
    enc_layers1 = list(c3d_model.children())
    enc_1_1 = nn.Sequential(*enc_layers1[:3])
    enc_2_1 = nn.Sequential(*enc_layers1[3:6])
    enc_3_1 = nn.Sequential(*enc_layers1[6:9])
    enc_4_1 = nn.Sequential(*enc_layers1[9:14])

    input = input.to(device)
    results = []
    for i in range(4):
        if not style:
            enc_1_1 = enc_1_1.to(device)
            enc_1_1_feat = enc_1_1(input)
            enc_1_1 = enc_1_1.cpu()
            results.append(enc_1_1_feat)
            input = input.cpu()
            del enc_1_1

            enc_2_1 = enc_2_1.to(device)
            enc_2_1_feat = enc_2_1(enc_1_1_feat)
            enc_2_1 = enc_2_1.cpu()
            results.append(enc_2_1_feat)
            del enc_2_1

            enc_3_1 = enc_3_1.to(device)
            enc_3_1_feat = enc_3_1(enc_2_1_feat)
            enc_3_1 = enc_3_1.cpu()
            results.append(enc_3_1_feat)
            del enc_3_1

            enc_4_1 = enc_4_1.to(device)
            enc_4_1_feat = enc_4_1(enc_3_1_feat)
            enc_4_1 = enc_4_1.cpu()
            results.append(enc_4_1_feat)
            enc_4_1_feat = enc_4_1_feat.cpu()
            del enc_4_1
            for i in range(len(results)):
                results[i] = results[i].cpu()
            
            enc_1_1_feat = enc_1_1_feat.cpu()
            enc_2_1_feat = enc_2_1_feat.cpu()
            enc_3_1_feat = enc_3_1_feat.cpu()
             
            torch.cuda.empty_cache()
            gc.collect()
            return merge_feats(results)


def merge_feats(content_feats):
    results = []
    _, c, t, h, w = content_feats[-1].shape
    
    for i in range(1, len(content_feats) - 1):
        results.append(nn.functional.interpolate(content_feats[i], (t, h, w)))
        
    results.append(content_feats[-1])
    return content_feats, torch.cat(results, dim=1)



parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_video', type=str,
                    help='File path to the content video')
parser.add_argument('--style_path', type=str,
                    help='File path to the style image')
parser.add_argument('--c3d', type=str, required=True)
parser.add_argument('--decoder', type=str, required=True)
parser.add_argument('--disentangle_appearance', type=str, default=None)
parser.add_argument('--entangle_feats', type=str, default=None)
parser.add_argument('--reverse_clip', action='store_true', help='reverse each alternate clip for better \
                     temporal consistency')

parser.add_argument('--save_ext', default='.mp4',
                    help='The extension name of the output video')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()
start = time.time()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

use_sigmoid = True

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# --style_path should be given
assert (args.style_path)
if args.style_path:
    style_path = Path(args.style_path)


c3d_model = util.c3d_model
# Load C3D model
c3d_model.load_state_dict(torch.load(args.c3d, map_location=torch.device('cpu')))
c3d_model.eval()


decoder = util.Decoder()
entangle_feats = util.entangle()

state_dict5 = torch.load(args.decoder)
decoder.model1_part0.load_state_dict(state_dict5['model1_part0'])
decoder.model1_part1.load_state_dict(state_dict5['model1_part1'])
decoder.model1_part2.load_state_dict(state_dict5['model1_part2'])
decoder.model1_part3.load_state_dict(state_dict5['model1_part3'])

entangle_feats.model.load_state_dict(torch.load(args.entangle_feats))

args.unit_norm = False

with torch.no_grad():
    if style_path.suffix in [".jpg", ".png", ".JPG", ".PNG"]:

        output_video_path = output_dir / '{:s}_stylized_{:s}{:s}'.format(
            args.content_video.split('/')[-1], style_path.stem, args.save_ext)
        writer = imageio.get_writer(output_video_path, mode='I', fps=30)

        
        style_img = np.array(Image.open(args.style_path).convert('RGB'))
       
        check = True
        output_height = 0
        output_width = 0
        output_height1 = 0
        output_width1 = 0

        video = cv2.VideoCapture(f'{args.content_video}.mp4')
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 16:
            print(f'total_frames less {total_frames}')
            exit(0)

        
        frame_count = 16
        last_frame = [0, 0]
        count = 0
        length = total_frames  # int(total_frames / (frame_count-2))
        clip_num = 0
        args.use_style_appearance = False
        args.style_feats_appearance = []

        for i in tqdm(range(0, total_frames, frame_count)):
            images = []
            style_imgs = []

            if i + frame_count > total_frames:
                break

            start = i + 1
            end = start + frame_count

            for j in range(start, end):
                ret, img = video.read()
                if not ret:
                    break
                img = np.array(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                
                if check:
                    output_height = 720#img.shape[0]
                    output_width = 1280#img.shape[1]
                    check = False
                data_transform = transforms.Resize((output_height, output_width), interpolation=Image.BICUBIC)

                img = Image.fromarray(img)
                img = transforms.ToTensor()(img).to(device)
                

                img = data_transform(img)
                images.append(img)

            if args.reverse_clip and ((clip_num & 1) == 1):
              images = images[::-1]
            content_clip = torch.stack(images)
            
            if i == 0:
                style = create_style_clip(args.style_path, frame_count, output_height, output_width)
                
            content_clip = content_clip.transpose(1, 0)
            content_clip = content_clip.unsqueeze(0)
            if i == 0:
                style = style.transpose(1, 0)
                style = style.unsqueeze(0)
            output = torch.empty_like(content_clip)
            
            print(f'img.shape {img.shape}')
            content_clip = content_clip.to(device)
            c3d_model = c3d_model.to(device)

            content_feats, merge_feat = encode_with_intermediate1(content_clip, c3d_model, device)
            merge_feat.cpu()

            c3d_model = c3d_model.to(device)
            content_clip = content_clip.cpu().detach()
            gc.collect()
            torch.cuda.empty_cache()

            if i == 0:
                style = style.to(device)
                style_feats, merge_feat = encode_with_intermediate1(style, c3d_model, device)
                style = style.cpu().detach()
                
            del content_clip
            torch.cuda.empty_cache()
            gc.collect()
            
            output = style_transfer2(decoder, content_feats, style_feats, device, args, use_sigmoid)
            del merge_feat

            output = output.squeeze()
            output = output.cpu().detach()
            output = output.transpose(0, 1)

            count_less_zero = 0
            count_more_one = 0

            counter = 0
            data_transform = transforms.Resize((360, 640), interpolation=Image.BILINEAR)
            
            start = 0
            end = frame_count
            step = 1

            if args.reverse_clip and (clip_num & 1) == 1:
                start = frame_count - 1
                end = -1
                step = -1

            for j in range(start, end, step):
                count_less_zero += torch.count_nonzero(output[j] < 0).item()
                count_more_one += torch.count_nonzero(output[j] > 1).item()
                output1 = output[j] * 255.0
                output1 = np.array(output1)
                output1 = np.transpose(output1, (1, 2, 0))
                writer.append_data(output1)
                count += 1

            clip_num += 1
            total_pixel = torch.prod(torch.tensor(list(output.shape)))
            print(f'count_less_zero : {count_less_zero} percentage : {(count_less_zero / total_pixel) * 100}')
            print(f'count_more_one : {count_more_one} percentage : {(count_more_one / total_pixel) * 100}')
            print(f'output.shape : {output.shape}')
            print(f'count {count}')

        #del style_feat
        torch.cuda.empty_cache()

print(f'execution time {(time.time() - start) / 60 }')
