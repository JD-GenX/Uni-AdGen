from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ast
import numpy as np
from src.utils.funcs import *
from ..coco.data_coco import filter_box, resize_and_crop

def adjust_and_normalize_bboxes(bboxes, orig_width, orig_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_norm = round(x1 / orig_width,3)  
        y1_norm = round(y1 / orig_height,3)
        x2_norm = round(x2 / orig_width,3)
        y2_norm = round(y2 / orig_height,3)
        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
    
    return normalized_bboxes
    
class BboxDataset_sam(Dataset):
    def __init__(self, dataset, resolution=1024, is_testset=False,):
        self.is_testset = is_testset
        self.dataset = dataset
        self.resolution = resolution
        if self.is_testset:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (resolution,resolution), interpolation=transforms.InterpolationMode.BILINEAR 
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def update_item(self, item):
        #dict_keys(['bbox_info', 'global_caption', 'image_info'])
        # item.update(image=x)
        dirname, filename = item['image_path'][3:].split('/')
        image_path = osp.join('/home/jovyan/myh-data-ceph-shcdt-1/data/SAM/', str(int(dirname)), filename)
        image = Image.open(image_path).convert('RGB')
        bbox_info = item['metadata']['bbox_info']
        global_caption = item['metadata']['global_caption']
        image_info = item['metadata']['image_info']

        height = image_info['height']
        width = image_info['width']
        file_name = image_info['file_name']

        bbox_list = []
        region_captions = []
        detail_region_captions = []
        for box in bbox_info:
            bbox_list.append(box['bbox'])
            region_captions.append(box['description'])
            detail_region_captions.append(box['detail_description'])

        item.update(dict(
            global_caption=global_caption,
            image=image,
            height=height,
            width=width,
            file_name=file_name,
            bbox_list=str(bbox_list),
            region_captions=str(region_captions),
            detail_region_captions=str(detail_region_captions),
        ))

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.is_testset:
            pass
        else:
            self.update_item(item)

        image = item['image']
        image = self.transform(image)
        # import pdb;pdb.set_trace()

        height = int(item['height'])
        width = int(item['width'])
        global_caption = item['global_caption']
        
        # global_caption = "The image depicts a woman reclining on her side in a softly lit setting. She is wearing a white dress with puffed sleeves and a deep V-neckline, adorned with buttons along the front. Her hair is styled in a neat bun. In the foreground, there are two elegant high-heeled shoes with decorative bows, positioned on the floor near her hand. The lighting creates a warm and inviting atmosphere, highlighting the textures and details of both the dress and the shoes. The overall color palette is neutral with soft shadows, emphasizing a sense of elegance and sophistication."
        # global_caption = "The image shows a well-appointed bedroom featuring a large wooden bed with a high headboard. The bed is centrally positioned against a cream-colored wall, adorned with a framed artwork above it. The bedding includes a white duvet and pillows, complemented by a decorative brown and white patterned throw draped over the foot of the bed. On the left side, there is a wooden nightstand with a vintage-style lamp and a small cup, while on the right side, another nightstand holds a digital clock and a potted plant. A robotic vacuum cleaner is visible on the floor in front of the bed. The overall ambiance is cozy and inviting, with natural elements and warm tones."
        # global_caption = "The image features a tub of ice cream prominently displayed on a white surface. The ice cream container is labeled “梦与星河” (Dream and Galaxy) and “地球双色冰淇淋” (Earth Two-Color Ice Cream), with an artistic depiction of Earth and a galaxy on its packaging. The container has a blue lid that complements the cosmic theme. To the left of the ice cream tub, there is a small rectangular plate with several chocolate cookies stacked on it. In the background, a small white bowl contains a green powder, possibly matcha, and a gold spoon rests nearby. A transparent glass cup is partially visible in the foreground. The setting is enhanced by a deep blue curtain in the background, adding to the overall sophisticated and dreamy atmosphere."
        # global_caption = "The image features a wooden box planter filled with a variety of colorful succulent plants. The box, with “100% Arabica Cafedo Brasil” and “Haut N.73843W” printed on its side, is positioned on a white surface. The succulents are arranged in a visually appealing manner, showcasing a range of colors from vibrant greens to deep reds and soft pinks. To the left of the planter, there is a wooden perpetual calendar displaying the number “06,” set against a background that includes a green leafy pattern. The overall setting is bright and minimalistic, highlighting the natural beauty of the succulents."
        # global_caption = "The background of the image is red, with the logo of Weilong and the words “Moyu Shuang” at the top left corner. In the center-left position, there is a box of Weilong Moyu Shuang, with product graphics and textual descriptions on the box. To the right of the box is a white bowl filled with spicy Moyu Shuang, which is orange-yellow in color, abundant in quantity, and wave-shaped. To the right of the bowl, there is a yellow circular label with the text “Total 20 Packs.” At the bottom of the image is a yellow banner with the words “QQ Bouncy, Spicy Satisfaction”. On the right side, there is a model wearing a white shirt, standing in the image, positioned towards the right."
        # global_caption = 'The image depicts an indoor office setting. At the top of the image, there is text that reads “Durable and Stable as a Rock,” followed by “Easily Supports Complex Home Needs.” On the right side of the image, there is a white printer placed on a wooden desk, with some paper being printed or already printed out. Scattered across the desk are various documents and papers, with an open folder on the left containing drawings and files. On the left side of the desk, near the background, there is a brown wooden cabinet. A white ceramic pot with green plants is placed on top of the cabinet, and books are neatly arranged inside the cabinet. The background features light-colored curtains and walls, with a socket and switch visible on the wall.'
        # global_caption = 'The image features a bottle of Ariel professional antibacterial laundry detergent. The bottle is white with a streamlined design, and it has a green cap with a transparent spout at the top. The front of the bottle displays the Ariel logo and the product name “Professional Antibacterial Laundry Detergent,” along with the phrase “Japanese Antibacterial Technology” below. The bottle features green patterns and functional icons, indicating features like “Sterilization,” “Disinfection,” and “Antibacterial.” The background offers a view from inside a washing machine, showing the stainless steel drum with a metallic blue-green hue, conveying a sense of cleanliness and technology.'
        # global_caption = 'This image is an advertisement for a men’s watch, showcasing an exquisite quartz timepiece. The watch features a classic and elegant design with a black dial, where white Arabic numeral hour markers are clearly visible. The inner circle displays 24-hour format numerals. The dial has three hands: the hour and minute hands are silver, while the second hand is red, adding a touch of dynamism to the visual. There is a date display window on the right side of the dial, enhancing the functionality of the watch. The case is made of metal with a glossy finish, paired with a black leather strap, giving it a luxurious appearance. The background consists of a blurred gray texture, highlighting the prominence of the watch. In the lower-left corner, there is a red promotional label with the text “京喜自营 包邮,” emphasizing the sales channel and delivery service.'
        # global_caption = 'Create an advertisement image for a men’s watch, focusing on a classic and elegant quartz timepiece. The watch should have a black dial with clearly visible white Arabic numeral hour markers, and an inner circle displaying 24-hour format numerals. Include three hands: silver for the hour and minute, and red for the second hand to add a dynamic touch. Position a date display window on the right side of the dial. The watch case should be metallic with a glossy finish, paired with a luxurious black leather strap. Set the watch against a blurred gray textured background to emphasize its prominence. In the lower-left corner, add a red promotional label with the text “京喜自营 包邮” to highlight the sales channel and delivery service.'
        # global_caption = "A bottle of cosmetics oil is placed on a stone, with virtual flowers and leaves in the background, golden light, close-up, and natural scenery."
        # global_caption = "A bottle of cosmetics sits on a small flat rock with moss and a few white flowers, clear foreground and bright blurred background with sunlight, National Geographic photo."
        # global_caption = "a bottle of  cosmetics sitting on top of a flat gray rock with moss next to a little tender green plants and  a little white flower with tender green leaf on a morandi green blurred clean background"
        # global_caption  = 'In a cozy and comfortable living room, sunlight filters through the curtains, casting a warm glow on the plush sofa. A few art pieces adorn the walls, adding a touch of homely warmth. In the corner of the room stands a stylish water dispenser, seamlessly blending with the overall decor of the living room. A soft rug lies on the floor, and a small vase with flowers sits on the coffee table alongside a few magazines, creating an atmosphere of relaxation and ease. The water dispenser is not just a practical appliance but an integral part of the living room, offering convenient hydration for family and guests.'
        global_caption = 'In a cozy and comfortable living room, sunlight filters through the curtains, casting a warm glow on the plush sofa. A few art pieces adorn the walls, adding a touch of homely warmth. In the corner of the room stands a stylish water dispenser, seamlessly blending with the overall decor of the living room. A soft rug lies on the floor, and a small vase with flowers sits on the coffee table alongside a few magazines, creating an atmosphere of relaxation and ease. The water dispenser is not just a practical appliance but an integral part of the living room, offering convenient hydration for family and guests.'
        region_bboxes_list = item['bbox_list']
        detail_region_caption_list = item['detail_region_captions']
        region_caption_list = item['region_captions']
        file_name = item['file_name']

        region_bboxes_list = ast.literal_eval(region_bboxes_list)
        region_bboxes_list = adjust_and_normalize_bboxes(region_bboxes_list,width,height)
        region_bboxes_list = np.array(region_bboxes_list, dtype=np.float32)

        region_caption_list = ast.literal_eval(region_caption_list)
        detail_region_caption_list = ast.literal_eval(detail_region_caption_list)

        if self.is_testset:
            pass
        else:
            image_pil = to_pil(image)
            obj_bbox = region_bboxes_list * [width, height, width, height]
            obj_class = detail_region_caption_list

            obj_bbox[:,2] = obj_bbox[:,2] - obj_bbox[:,0]
            obj_bbox[:,3] = obj_bbox[:,3] - obj_bbox[:,1]
            image_pil, obj_bbox = resize_and_crop(image_pil, obj_bbox)
            image =  to_ts(image_pil)
            image = image*2-1
            obj_bbox, obj_class = filter_box(obj_bbox, obj_class)
            obj_bbox = obj_bbox/384
            obj_bbox = obj_bbox.reshape(-1,4)
            obj_bbox[:,2] = obj_bbox[:,0] + obj_bbox[:,2]
            obj_bbox[:,3] = obj_bbox[:,1] + obj_bbox[:,3]
            
            region_bboxes_list = obj_bbox
            detail_region_caption_list = obj_class
        
        if None in detail_region_caption_list:
            detail_region_caption_list = region_caption_list
        if None in region_caption_list:
            return self.__getitem__(self, idx+1)

        return {
            'image': image,
            'global_caption': global_caption,
            'detail_region_caption_list': detail_region_caption_list,
            'region_bboxes_list': region_bboxes_list,
            'region_caption_list': region_caption_list,
            'file_name': file_name,
            'height': height,
            'width': width
        }

