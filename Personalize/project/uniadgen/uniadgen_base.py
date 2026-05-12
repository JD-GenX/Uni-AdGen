import sys;sys.path.insert(0, './three_party/Janus')
import gc
from rich import print
import os.path as osp
import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from src.utils.funcs import *
from peft import LoraConfig, set_peft_model_state_dict, PeftModel, get_peft_model, TaskType
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToPILImage, ToTensor
to_pil = ToPILImage()
to_ts = ToTensor()
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from src.utils.causal_loss import ForCausalLMLoss
from tokenizers import AddedToken
import traceback
from tqdm import tqdm, trange
from torchvision.transforms import ToTensor
from project.base.base_system import Base_System
from lightning.pytorch.utilities import CombinedLoader
from .dataset.set_dataset import get_dataset
from transformers.models.llama.modeling_llama import *
from src.models.dinov2_adapter import Dinov2_Adapter
import copy
from .dataset.code_hico.dataset.util import image_normalize
import torchvision.transforms as T
from janus.models.processing_vlm import VLChatProcessorOutput, BatchedVLChatProcessorOutput

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class System(Base_System):
    def __init__(self, 
        args=None,
        accelerator=None,
    ) -> None:
        super().__init__()
        if args.test and args.test_data.data_name=='1k':
            args.max_test_len=-1

        self.args = args
        self.accelerator = accelerator

        model_path = self.args.janus_path
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )

        # mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        # self.mmgpt = mmgpt

        self.custom_token_num = 64

        self.vl_chat_processor = vl_chat_processor
        self.ori_image_proc = self.vl_chat_processor.image_processor.__class__.__call__
        self.vl_chat_processor.image_processor.__class__.__call__ = self.hack_image_proc

        self.tokenizer = tokenizer
        self.vl_gpt = vl_gpt
        if self.vl_gpt.vision_model.vision_tower.ignore_head:
            vl_gpt.vision_model.vision_tower.attn_pool = None

        if self.args.use_special_tokens:
            res = tokenizer.add_tokens([
                AddedToken("<grounding>", special=True),
                AddedToken("</grounding>", special=True),
                AddedToken("<box>", special=True),
                AddedToken("</box>", special=True),
                AddedToken("<ref>", special=True),
                AddedToken("</ref>", special=True),
            ])
            print('\nadd special tokens', res)

        if self.args.use_numhw_tokens:
            hw_list = []
            for i in range(100):
                hw_list.append(AddedToken(f"<h{i}>", special=True))
                hw_list.append(AddedToken(f"<w{i}>", special=True))
            res = tokenizer.add_tokens(hw_list)
            print('\nadd hw_num tokens', res)

        img_size = self.args.janus_hw
        self.image_token_num_per_image = (self.args.janus_hw//16)**2
        self.vl_chat_processor.num_image_tokens = self.image_token_num_per_image
        self.vl_chat_processor.image_processor.image_size = self.args.janus_hw

        self.prepare_trainable()
    

    def hack_image_proc(self, image, return_tensors='pt'):
        if isinstance(image, torch.Tensor):
            class ImagesOutputs:
                def __init__(self, pixel_values):
                    self.pixel_values = pixel_values
            return ImagesOutputs(image)
        else:
            return self.ori_image_proc(
                self.vl_chat_processor.image_processor,#self
                image, #images
                return_tensors=return_tensors
            )

    def fuse_image(self, image, alpha_map):

        # transform
        transform = T.Compose([
            image_normalize()
        ])
        
        # Extract the alpha channel from the alpha map
        alpha = alpha_map[3:,:,:]

        alpha_image = transform(alpha_map[:3, :, :, ])
        
        # Calculate the blended image
        blended_image = (image * (1 - alpha)) + (alpha_image * alpha)
    
        return blended_image
    
    def prepare_trainable(self,):
        self.trainable = []
        self.non_trainable = []

        self.freeze_params(self.parameters())

        if self.args.gradient_checkpointing_enable:
            self.vl_gpt.language_model.gradient_checkpointing_enable()

        if self.args.tuning_mode == 'all':
            self.trainable.append(self)
        elif self.args.tuning_mode == 'lm':
            self.trainable.append(self.vl_gpt.language_model)
        elif self.args.tuning_mode == 'lora':
            transformer_lora_config = LoraConfig(
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )#15MB
            self.vl_gpt.language_model.enable_input_require_grads()
            self.vl_gpt = get_peft_model(self.vl_gpt, transformer_lora_config)

            if self.args.tune_token_when_lora and (self.args.use_special_tokens or self.args.use_numhw_tokens):
                self.unfreeze_params(self.vl_gpt.language_model.model.embed_tokens.parameters())
        elif self.args.tuning_mode == 'lora_control':
            controlnet_components = [
                "adapter", "Dinov2_Adapter",
                "condition_projection", 
                "adapter_mlp", 
                "condition_mlp",
                "condition_layers",
                "controlnet", "control_layer",
                "text_mask_generator",
                "visual_mask_generator",
                "dual_modality_fusion_encoder"
            ]
            
            all_modules = dict(self.vl_gpt.named_modules())
            # print('-----------all_modules---------')
            # print(all_modules)
            
            lora_target_modules = []
            for name, module in all_modules.items():
                is_target = any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj"])
                is_excluded = any(excluded in name for excluded in controlnet_components)
                
                if is_target and not is_excluded:
                    lora_target_modules.append(name)
            
            # print('-----------lora_target_modules---------')
            # print(lora_target_modules)

            transformer_lora_config = LoraConfig(
                r=self.args.lora_rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_target_modules,  
            )
            
            self.vl_gpt.language_model.enable_input_require_grads()

            self.vl_gpt = get_peft_model(self.vl_gpt, transformer_lora_config)

            if self.args.tune_token_when_lora and (self.args.use_special_tokens or self.args.use_numhw_tokens):
                self.unfreeze_params(self.vl_gpt.language_model.model.embed_tokens.parameters())
            
            self.trainable.append(self.vl_gpt.language_model.model.adapter)
            self.trainable.append(self.vl_gpt.language_model.model.adapter_mlp)
            self.trainable.append(self.vl_gpt.language_model.model.condition_mlp)
            self.trainable.append(self.vl_gpt.language_model.model.condition_layers)
            self.trainable.append(self.vl_gpt.visual_mask_generator)
            self.trainable.append(self.vl_gpt.text_mask_generator)
            self.trainable.append(self.vl_gpt.dual_modality_fusion_encoder)
            self.non_trainable.append(self.vl_gpt.gen_vision_model)
            self.non_trainable.append(self.vl_gpt.vision_model)

        elif self.args.tuning_mode == 'lora_ranni':
            peft_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                lora_dropout=0.05,
                bias="none",
            )#30MB
            self.vl_gpt = get_peft_model(self.vl_gpt, peft_config)
        elif self.args.tuning_mode == 'stage1':
            self.trainable.append(self.vl_gpt.aligner)
            self.trainable.append(self.vl_gpt.gen_aligner)
            self.trainable.append(self.vl_gpt.gen_head)
        elif self.args.tuning_mode == 'stage2':
            self.trainable.append(self)
            self.non_trainable.append(self.vl_gpt.vision_model)
            self.non_trainable.append(self.vl_gpt.gen_vision_model)
        elif self.args.tuning_mode == 'stage2_lora':
            self.trainable.append(self)
            self.non_trainable.append(self.vl_gpt.vision_model)
            self.non_trainable.append(self.vl_gpt.gen_vision_model)
        elif self.args.tuning_mode == 'stage3':
            self.trainable.append(self)
            self.non_trainable.append(self.vl_gpt.gen_vision_model)
            
        else:
            assert False

        for module in self.non_trainable:
            self.freeze_params(module.parameters())

        for module in self.trainable:
            self.unfreeze_params(module.parameters())


    def wrap_t2i_prompt(self, 
        caption="a yellow car in front of the tree"
    ):
        conversation = [
            {
                "role": "<|User|>",
                "content": caption,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag

        inputs_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        inputs_ids = torch.LongTensor(inputs_ids)
        return prompt, inputs_ids

    def wrap_p2i_prompt(self, 
        image,
        sku_title=None, 
        prompt=None,
        in_stage1=False,
        word_list=None,
        custom_token_num=576,
        history_titles=None
    ):
        placeholders = '<image_placeholder>' * len(image)
        title_placeholders = '<title_placeholder>' * len(history_titles)

        question = "Generate an engaging title using only the words from the word list '{}' in similar style to the history title set '{}', and based on the original title '{}', create a promising product marketing image for the produce in the image in similar style to history image set {}. (Directly provide the title and prompt without preface explanation)".format(word_list, title_placeholders, sku_title, placeholders)
        
        conversation = [
                {"role": "<|User|>",
                "content": question,
                "images": [image],},
                {"role": "<|Assistant|>", "content": f"{prompt}"},
            ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        if in_stage1:
            prompt = sft_format
        else:
            prompt = sft_format + self.vl_chat_processor.image_start_tag

        inputs_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        inputs_ids = torch.LongTensor(inputs_ids)

        if in_stage1:
            inputs_ids = inputs_ids[...,:-1]

        ##
        device = torch.tensor([0]).device
        inputs_ids = inputs_ids.to(device)

        # add title tokens to the input_ids
        title_token_mask: torch.BoolTensor = inputs_ids == self.vl_chat_processor.title_id
        title_indices = title_token_mask.nonzero()
        if len(title_indices) > 0:
            inputs_ids, num_title_tokens = self.vl_chat_processor.add_title_token(
                title_indices=title_indices.squeeze(-1).tolist(),
                input_ids=inputs_ids,
            )

        # add image tokens to the inputs_ids
        image_token_mask: torch.BoolTensor = inputs_ids == self.vl_chat_processor.image_id
        image_indices = image_token_mask.nonzero()
        inputs_ids, num_image_tokens = self.vl_chat_processor.custom_add_image_token(
            image_indices=image_indices,
            input_ids=inputs_ids,
            custom_token_num = self.custom_token_num,
        )

        # load images
        images = []
        for image_i in image:
            images.append(Image.fromarray(image_i.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)))
        images_outputs = self.vl_chat_processor.custom_image_processor.preprocess(images, return_tensors="pt")

        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=inputs_ids,
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=num_image_tokens,
        )

        if len(title_indices) > 0:
            prepare.num_title_tokens = num_title_tokens
            prepare.history_titles = history_titles if history_titles else []
        else:
            prepare.num_title_tokens = torch.IntTensor([])
            prepare.history_titles = []

        return prepare
    
    def wrap_uni_prompt(self, 
        caption="a yellow car in front of the tree",
        grounding=None,
        in_stage1=False,
    ):
        conversation = [
            {
                "role": "<|User|>",
                "content": caption,
            },
            {"role": "<|Assistant|>", "content": f"{grounding}"},#可能dropout
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )

        if in_stage1:
            prompt = sft_format
        else:
            prompt = sft_format + self.vl_chat_processor.image_start_tag

        inputs_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        inputs_ids = torch.LongTensor(inputs_ids)

        if in_stage1:
            inputs_ids = inputs_ids[...,:-1]
        return prompt, inputs_ids

    def wrap_mmu_prompt(self, 
        question="a yellow car in front of the tree",
        image=None,
        answer="",
    ):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": f"{answer}"},
        ]

        if isinstance(image, torch.Tensor):
            pil_images = image
        else:
            pil_images = load_pil_images(conversation)

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.device)

        prepare_inputs['pixel_values'] = prepare_inputs['pixel_values'].to(torch.bfloat16)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        return prepare_inputs, inputs_embeds

    def decode_text(self, inputs_ids):
        return self.tokenizer.decode(inputs_ids, skip_special_tokens=False)

    def decode_plan_text_batch(self, inputs_ids):
        texts = ["<grounding>"+self.decode_text(t) for t in inputs_ids]
        new_texts = []
        for text in texts:
            end_pos = text.find("</grounding>")
            if end_pos != -1:
                result = text[:end_pos + len("</grounding>")]
            else:
                result = "<grounding>"+"</grounding>"
            new_texts.append(result)
        return new_texts

    def decode_p2i_text_batch(self, inputs_ids):
            texts = ["<prompt>"+self.decode_text(t) for t in inputs_ids]
            new_texts = []
            for text in texts:
                end_pos = text.find("</prompt>")
                if end_pos != -1:
                    result = text[:end_pos + len("</prompt>")]
                else:
                    result = "<prompt>"+"</prompt>"
                new_texts.append(result)
            return new_texts
    
    def get_pr_grounding_part(self, text):
        pos = text.find("<grounding>")
        if pos != -1:
            text = text[pos:]
        return text
    
    def decode_mmu_text_batch(self, inputs_ids):
        new_ids = []
        for ids in inputs_ids:
            try:
                pos = torch.where(ids==self.tokenizer.eos_token_id)[0][0].item()
                ids = ids[:pos]
            except:
                pass
            new_ids.append(ids)
        inputs_ids = [t for t in new_ids]
        texts = [self.decode_text(t) for t in inputs_ids]
        return texts

    def parse_multimodal_output(self, texts):
        """
        解析多模态模型的固定格式输出
        Args:
            texts (list): 输入文本列表，格式为 "<prompt>xxx</prompt>"

        Returns:
            list: 包含提取内容的列表
        """
        results = []
        for text in texts:

            text = text.strip()
            pattern = r'<prompt>(.+?)</prompt>'
            match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)

            if not match:
                print("match is empty")
                content = ''
            else:
                content = match.group(1).strip()

            results.append(content)

        return results


    @torch.inference_mode()
    def uni_generate(
        self,
        batch = None,
        gen_path = None,
        batch_idx = None,
        accelerator = None, ###
        prompt: str = None,
        temperature: float = 1,
        parallel_size: int = 4,#16
        cfg_weight: float = 5,
        patch_size: int = 16,
        pred_layout = True,
        pred_image = True,
        save_local = True,
        use_uni_prompt_in_t2i = True,
        is_mmu = False,
        **kwargs,
    ):
        import random
        import time
        random.seed(time.time())
        index_local = random.randint(0,100)
        print(self.args.janus_hw)

        parallel_size = self.args.parallel_size
        img_size = self.args.janus_hw
        image_token_num_per_image = (self.args.janus_hw//16)**2

        print('\n uni...')

        gt_image = batch['image']
        base_caption = batch['base_caption']
        gt_grounding = batch['gt_grounding']

        print("(uni_generate)base_caption(modify):", base_caption)

        bs = len(base_caption)

        self.vl_gpt.eval()
        # self.mmgpt.eval()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

            if pred_layout:
                if is_mmu:
                    prepare_inputs = batch['prepare_inputs_infer']
                    inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                    attention_mask = prepare_inputs['attention_mask']
                else:
                    inputs_ids = batch['uni_stage1_inputs_ids']
                    attention_mask = batch['uni_stage1_attention_mask']
                    inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(inputs_ids.to(self.device))
                outputs = self.x2t(inputs_embeds, attention_mask)

                is_p2i = True
                if is_mmu or is_p2i:
                    pr_grounding = self.decode_mmu_text_batch(outputs)
                else:
                    pr_grounding = self.decode_plan_text_batch(outputs)

                if pred_image:
                    bs = len(pr_grounding)
                    print("pr_grounding", pr_grounding)
                    all_inputs_ids = []
                    for base_caption_i, grounding_prompt in zip(base_caption, pr_grounding):
                        _, inputs_ids = self.wrap_uni_prompt(base_caption_i, grounding_prompt)  ## 使用新的pr_grounding更新uni2阶段的输入
                        all_inputs_ids.append(inputs_ids)
                    uni_inputs_ids, uni_attention_mask = self.pad_input_ids(all_inputs_ids)
                    uni_attention_mask = torch.cat([uni_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
                    batch.update(dict(
                        uni_inputs_ids=uni_inputs_ids.to(self.device),
                        uni_attention_mask=uni_attention_mask.to(self.device)
                    ))
            else:
                pr_grounding = gt_grounding

            if pred_image:
                batch_new = self.t2i_infer_collate_batch(batch, use_uni=use_uni_prompt_in_t2i)
                cfg_emb=None
                cfg_inputs_ids = batch_new['cfg_inputs_ids']
                cfg_attention_mask = batch_new['cfg_attention_mask']

                func = self.t2i
                pr_image, edit_mask = func(
                    None, parallel_size, image_token_num_per_image, cfg_weight, temperature, img_size, patch_size, gt_image, batch, 
                    mask=cfg_attention_mask, 
                    tokens=cfg_inputs_ids,
                    emb=cfg_emb,
                )

                pr_image = pr_image.float()
            else:
                pr_image = gt_image
                edit_mask = None

        self.vl_gpt.train()
        self.clean(accelerator)

        if save_local:
            data = dict(
                base_caption=base_caption, gt_grounding=gt_grounding, pr_grounding=pr_grounding if pred_layout else ''
            )

            json_path = osp.join(gen_path, str(batch_idx)+'_'+str(index_local)+'_layout.json')
            save_json(json_path, data)

            if gt_image is None:
                    gt_image = pr_image
            if gt_grounding is None:
                gt_grounding = pr_grounding
            vis = torch.cat([gt_image, pr_image], dim=0)  ## 16
            vis = denorm_pt(vis)
            img_path = osp.join(gen_path, str(batch_idx)+'_'+str(index_local)+'.png')
            save_img(vis, img_path, bs=bs)
            

            img_each_path = osp.join(gen_path, str(batch_idx))
            mkdir(img_each_path)
            for i in range(len(vis)):
                col = i % bs
                row = i // bs
                to_pil(vis[i]).save(f"{img_each_path}/{row}_{col}.png")
        return dict(
            pr_grounding=pr_grounding, 
            pr_image=pr_image,
        )

    @torch.inference_mode()
    def generate(self,
        mmgpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
        prompt: str,
        temperature: float = 1,
        parallel_size: int = 4,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
    
        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id
    
        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    
        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
    
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
    
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
    
    
        pr_image = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = pr_image.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    
        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
    
        os.makedirs('generated_samples', exist_ok=True)
        for i in range(parallel_size):
            save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
            PIL.Image.fromarray(visual_img[i]).save(save_path)
        return pr_image
    
    @torch.inference_mode()
    def uni_generate_p2i(
        self,
        batch = None,
        gen_path = None,
        batch_idx = None,
        accelerator = None, 
        prompt: str = None,
        temperature: float = 1,
        parallel_size: int = 4,
        cfg_weight: float = 5,
        patch_size: int = 16,
        pred_layout = True,
        pred_image = True,
        save_local = True,
        use_uni_prompt_in_t2i = True,
        is_mmu = False,
        **kwargs,
    ):
        import random
        import time
        random.seed(time.time())
        index_local = random.randint(0,100)

        parallel_size = self.args.parallel_size
        img_size = self.args.janus_hw
        image_token_num_per_image = (self.args.janus_hw//16)**2

        print('\n p2i...')

        gt_image = batch['image']
        trans_image = batch['trans_image']
        base_caption = batch['base_caption']
        gt_grounding = batch['gt_grounding']
        prompt = batch['prompt']
        sku_title = batch['sku_title']
        condition = batch['control']
        word_list = batch['word_list']
        history_images = batch['history_images']
        history_titles = batch['history_titles']

        print("(p2i_generate)prompt(modify):", prompt)

        bs = len(base_caption)

        self.vl_gpt.eval()
        # self.mmgpt.eval()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

            if pred_layout:
                prepare_inputs = batch['p2i_stage1_prepare_inputs']
                inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                # pad_input_ids
                padded_all_inputs_ids = prepare_inputs['input_ids']
                attention_mask = prepare_inputs['attention_mask']

                if self.args.test or self.args.func is not None:
                    pass
                else:
                    if inputs_embeds.shape[1] > self.args.max_seq_len:
                        print('mmu exceeds maximum length')
                        start = inputs_embeds.shape[1] - self.args.max_seq_len
                        inputs_embeds = inputs_embeds[:, start:]
                        padded_all_inputs_ids = padded_all_inputs_ids[:, start:]
                        attention_mask = attention_mask[:, start:]

                outputs = self.x2t(inputs_embeds, attention_mask)
                
                pr_grounding = self.decode_p2i_text_batch(outputs)
                pr_title = self.parse_multimodal_output(pr_grounding)
                print("pr_grounding", pr_grounding)
                print("pr_title", pr_title)

                if pred_image:
                    bs = len(pr_grounding)
                    prepares = []
                    for sku_title_i, trans_image_i, pr_prompt, word_list_i, history_images_i, history_titles_i in zip(sku_title, trans_image, pr_grounding, word_list, history_images, history_titles):
                        history_titles_i = history_titles_i.strip().split('<title_split>')
                        prepare = self.wrap_p2i_prompt(history_images_i, sku_title_i, pr_prompt, word_list_i, history_titles=history_titles_i, custom_token_num=self.custom_token_num)
                        prepares.append(prepare)
 
                    batch.update(dict(
                        uni_prepare_inputs=prepares
                    ))
            else:
                pr_grounding = gt_grounding

            if pred_image:

                prepares = batch['uni_prepare_inputs']
                prepare_inputs = self.vl_chat_processor.custom_batchify(prepares, custom_token_num=self.custom_token_num)
                inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                # pad_input_ids
                padded_all_inputs_ids = prepare_inputs['input_ids']
                attention_mask = prepare_inputs['attention_mask']
                if self.args.test or self.args.func is not None:
                    pass
                else:
                    if inputs_embeds.shape[1] > self.args.max_seq_len:
                        print('mmu exceeds maximum length')
                        start = inputs_embeds.shape[1] - self.args.max_seq_len
                        inputs_embeds = inputs_embeds[:, start:]
                        padded_all_inputs_ids = padded_all_inputs_ids[:, start:]
                        attention_mask = attention_mask[:, start:]
                attention_mask = torch.cat([attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
                batch.update(dict(
                        uni_inputs_ids=padded_all_inputs_ids.to(self.device),
                        uni_attention_mask=attention_mask.to(self.device),
                    ))           

                batch_new, pad_p2i = self.t2i_infer_collate_batch(batch, use_uni=use_uni_prompt_in_t2i)
                
                if pad_p2i:
                    padded_inputs_ids = batch_new['cfg_inputs_ids'][0::2]
                    for i in len(prepares):
                        prepares[i]['input_ids'] = padded_inputs_ids[i]
                    prepare_inputs = self.vl_chat_processor.custom_batchify(prepares, custom_token_num=self.custom_token_num)
                    inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                                   
                
                
                cfg_emb=None
                cfg_inputs_ids = batch_new['cfg_inputs_ids']
                cfg_attention_mask = batch_new['cfg_attention_mask']

                func = self.t2i
                pr_image, edit_mask = func(
                    None, parallel_size, image_token_num_per_image, cfg_weight, temperature, img_size, patch_size, gt_image, batch, 
                    mask=cfg_attention_mask, 
                    tokens=cfg_inputs_ids[1::2],
                    emb=inputs_embeds,
                    condition=condition, 
                )
                pr_image = pr_image.float()
            else:
                pr_image = gt_image
                edit_mask = None

        self.vl_gpt.train()
        self.clean(accelerator)

        if save_local:
            data = dict(
                base_caption=base_caption, gt_grounding=gt_grounding, prompt=prompt, pr_grounding=pr_grounding, pr_title=pr_title, gt_title=batch['gt_title'] if pred_layout else ''
            )

            json_path = osp.join(gen_path, str(batch_idx)+'_'+str(index_local)+'_layout.json')
            save_json(json_path, data)

            if gt_image is None:
                gt_image = pr_image
            if gt_grounding is None:
                gt_grounding = pr_grounding
            vis = torch.cat([gt_image, pr_image], dim=0)  ## 16
            vis = torch.cat([vis, condition], dim=0)  ## 16
            vis = denorm_pt(vis)
            img_path = osp.join(gen_path, str(batch_idx)+'_'+str(index_local)+'.png')
            save_img(vis, img_path, bs=bs)
            

            img_each_path = osp.join(gen_path, str(batch_idx))
            mkdir(img_each_path)
            for i in range(len(vis)):
                col = i % bs
                row = i // bs
                to_pil(vis[i]).save(f"{img_each_path}/{row}_{col}.png")
        return dict(
            pr_grounding=pr_grounding, 
            pr_image=pr_image,
        )

    def trans_gr_to_creati(self, prompt):
        pattern = r"<ref>(.*?)</ref><box>\[(.*?)\]</box>"
        matches = re.findall(pattern, prompt)
        prompts = []
        boxes = []
        for desc, box in matches:
            ori_x1, ori_y1, ori_x2, ori_y2 = map(int, box.split(","))
            x1 = ori_x1/1000
            x2 = ori_x2/1000
            y1 = ori_y1/1000
            y2 = ori_y2/1000
            prompts.append(desc)
            boxes.append([x1,y1,x2,y2])
        return boxes, prompts

    def vis_image(self, vis, pr_grounding):
        vis = denorm_pt(vis)
        assert isinstance(pr_grounding, list)
        try:
            assert len(vis) == len(pr_grounding)
        except:
            import pdb;pdb.set_trace()

        creati_style=True
        if creati_style:
            h = 384
            out_vis = []
            for i in range(len(vis)):
                image = to_pil(vis[i])
                boxes, caps = self.trans_gr_to_creati(pr_grounding[i])
                show_input = {"boxes":scale_boxes(boxes,h,h), "labels":caps}
                bbox_visualization_img = bbox_visualization(image,show_input)
                out_vis.append(to_ts(bbox_visualization_img))
            out_vis = torch.stack(out_vis, 0)
            vis = donorm_pt(out_vis)
        else:
            for i in range(len(vis)):
                img = self.draw_boxes_on_image(
                    to_pil(vis[i]),
                    pr_grounding[i],
                )
                vis[i] = donorm_pt(to_ts(img))
        return vis

    def clean(self, accelerator):
        torch.cuda.empty_cache()
        gc.collect()
        if accelerator is not None:
            accelerator.free_memory()

    def draw_boxes_on_image(self, *args):
        return draw_boxes_on_image(*args, use_centerhw=self.args.use_centerhw)

    def x2t(self, inputs_embeds, attention_mask=None):
        return self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
            )

    def t2i(self, inputs_ids, parallel_size, image_token_num_per_image, cfg_weight, temperature, img_size, patch_size, gt_image=None, batch=None, mask=None, tokens=None, emb=None, condition=None):
        generator = torch.Generator(device='cuda')
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)

        if True:
            with torch.no_grad():
                gt_images = gt_image.bfloat16()
                bs = gt_images.shape[0]
                gt_labels = self.vl_gpt.gen_vision_model.encode(gt_images)[-1][-1].reshape(bs,-1) # torch.Size([8, 576])

        else:
            gt_labels = None

        if tokens is None and emb is None:
            tokens = torch.zeros((parallel_size*2, len(inputs_ids)), dtype=torch.int).cuda()
            for i in range(parallel_size*2):
                tokens[i, :] = inputs_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.vl_chat_processor.pad_id
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
        else:
            if tokens is None:
                inputs_embeds = emb
            else:
                bs, token_num, token_dim = emb.shape
                tokens = torch.cat([tokens]*parallel_size)
                neg_inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
                emb = torch.cat([emb]*parallel_size)
                inputs_embeds = torch.stack([emb, neg_inputs_embeds], dim=1).view(bs*2, token_num, token_dim)
            mask = torch.cat([mask]*parallel_size)

        num_gen = inputs_embeds.shape[0] // 2
        ## vl_gpt, (8, 576)
        generated_tokens = self.sample_image(inputs_embeds, num_gen, image_token_num_per_image, mask, cfg_weight, temperature, generator, batch, gt_labels, condition)

        ## (8, 3, 384, 384)
        dec = self.vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[num_gen, 8, img_size//patch_size, img_size//patch_size])

        if self.args.use_teacher_forcing:
            mask_image = batch['edit_region']
            bs = mask_image.shape[0]
            mask_image = resize_pt(mask_image.reshape(bs,1,24,24).repeat(1,3,1,1), self.args.janus_hw).to(dec)
        else:
            mask_image = None

        return dec, mask_image

    def sample_image(self, inputs_embeds, num_gen, image_token_num_per_image, mask, cfg_weight, temperature, generator, batch, gt_labels, condition):
        generated_tokens = torch.zeros((num_gen, image_token_num_per_image), dtype=torch.int).cuda()
        # bs = len(condition)
        # print("initial, inputs_embeds", inputs_embeds.shape)  ## [bs*2, *, 2048]
        for i in tqdm(range(image_token_num_per_image)):
            outputs = self.vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds, 
                attention_mask=mask.to(self.device) if mask is not None else None,
                use_cache=True, 
                past_key_values=outputs.past_key_values if i != 0 else None,
                condition=condition,
                img_index=i
            )
            hidden_states = outputs.last_hidden_state ## ## [bs*2, 1, 2048]
            
            logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])  ## torch.Size([bs*2, 16384])
            logit_cond = logits[0::2, :]  ## torch.Size([bs, 16384])
            logit_uncond = logits[1::2, :]  ## torch.Size([bs, 16384])    

            if self.args.cfg_weight is not None:
                cfg_weight = self.args.cfg_weight
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)  # (bs,16384)

            next_token = torch.multinomial(probs, num_samples=1, generator=generator) # (bs,1)  

            # gt_image token
            if self.args.use_teacher_forcing:
                edit_region = batch['edit_region']
                bs = len(edit_region)
                for bid in range(bs):
                    if edit_region[bid,i].item() != 0:
                        next_token[bid,0] = gt_labels[bid,i]

            generated_tokens[:, i] = next_token.squeeze(dim=-1)


            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)  ## [bs*2, 1, 2048]

        return generated_tokens

    def t2i_infer_collate(self, batch):
        batch = default_collate(batch)

        ##### t2i
        bs = len(batch['prompt'])
        all_inputs_ids = []
        for prompt in batch['prompt']:
            wrapped_prompt, inputs_ids = self.wrap_t2i_prompt(prompt)
            all_inputs_ids.append(inputs_ids)

        max_length = max(map(len, all_inputs_ids))

        padded_all_inputs_ids = torch.ones((bs, max_length))*self.vl_chat_processor.pad_id
        padded_all_attention_mask = torch.zeros((bs, max_length))
        for i, inputs_ids in enumerate(all_inputs_ids):
            padded_all_inputs_ids[i, -len(inputs_ids):] = inputs_ids
            padded_all_attention_mask[i, -len(inputs_ids):] = 1
        padded_all_inputs_ids = padded_all_inputs_ids.int()
        padded_all_attention_mask = torch.cat([padded_all_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
        padded_all_attention_mask = padded_all_attention_mask.int()
        
        batch.update(dict(
            cfg_inputs_ids=padded_all_inputs_ids,
            cfg_attention_mask=padded_all_attention_mask
        ))
        return batch

    def t2i_infer_collate_batch(self, 
        batch, 
        use_uni=False,
    ):
        bs = len(batch['prompt'])

        if use_uni:
            t2i_inputs_ids = batch['uni_inputs_ids']
            t2i_attention_mask = batch['uni_attention_mask']
            pad_t2i=False
        else:
            assert False
            t2i_inputs_ids = batch['t2i_inputs_ids']
            t2i_attention_mask = batch['t2i_attention_mask']

        max_length = t2i_inputs_ids.shape[-1]

        if self.args.use_neg_box:  ## default True
            neg_all_inputs_ids = []
            for base_caption, grounding_prompt in zip(batch['neg_base_caption'], batch['neg_gt_grounding']):
                _, inputs_ids = self.wrap_uni_prompt(base_caption, grounding_prompt)
                neg_all_inputs_ids.append(inputs_ids)

            max_length_neg = max([len(t) for t in neg_all_inputs_ids])
            if max_length_neg > max_length:
                need_pad = max_length_neg - max_length
                
                t2i_inputs_ids = torch.cat([torch.ones((bs,need_pad)).to(t2i_inputs_ids)*self.vl_chat_processor.pad_id, t2i_inputs_ids], dim=1)

                t2i_attention_mask = torch.cat([torch.zeros((bs,need_pad)).to(t2i_attention_mask)*self.vl_chat_processor.pad_id, t2i_attention_mask], dim=1)
                max_length = max_length_neg
                pad_t2i = True

            uni_inputs_ids, uni_attention_mask = self.pad_input_ids(neg_all_inputs_ids, max_length)
            uni_attention_mask_image = torch.cat([uni_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
            neg_ids = uni_inputs_ids
            neg_mask = uni_attention_mask_image
        else:
            # _, neg_inputs_ids = self.wrap_uni_prompt(self.args.neg_prompt, '<grounding></grounding>')
            _, neg_inputs_ids = self.wrap_uni_prompt(self.args.neg_prompt, '')
            # _, neg_inputs_ids = self.wrap_t2i_prompt(self.args.neg_prompt)

            max_length_neg = neg_inputs_ids.shape[-1]
            if max_length_neg > max_length:
                need_pad = max_length_neg - max_length
                
                t2i_inputs_ids = torch.cat([torch.ones((bs,need_pad)).to(t2i_inputs_ids)*self.vl_chat_processor.pad_id, t2i_inputs_ids], dim=1)

                t2i_attention_mask = torch.cat([torch.zeros((bs,need_pad)).to(t2i_attention_mask)*self.vl_chat_processor.pad_id, t2i_attention_mask], dim=1)
                max_length = max_length_neg
                pad_t2i=True


            neg_ids, neg_mask = self.pad_input_ids([neg_inputs_ids]*bs, max_length=max_length)
            neg_mask = torch.cat([neg_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)

        neg_mask2 = neg_mask

        padded_all_inputs_ids = torch.stack([t2i_inputs_ids, neg_ids.to(self.device)], dim=1).view(bs*2,-1)  
        # padded_all_inputs_ids = t2i_inputs_ids  
        padded_all_attention_mask = torch.stack([t2i_attention_mask, neg_mask2.to(self.device)], dim=1).view(bs*2,-1) 
        # padded_all_attention_mask = t2i_attention_mask
        
        batch.update(dict(
            cfg_inputs_ids=padded_all_inputs_ids.int(),
            cfg_attention_mask=padded_all_attention_mask.int()
        ))
        return batch, pad_t2i
    
    def pad_input_ids(self, all_inputs_ids, max_length=None):
        bs = len(all_inputs_ids)

        if self.args.debug_max_seq_len is not None:
            # print('debugging...')
            max_length = self.args.debug_max_seq_len
        if max_length is None:
            # import pdb;pdb.set_trace()
            max_length = max(map(len, all_inputs_ids))

        padded_all_inputs_ids = torch.ones((bs, max_length))*self.vl_chat_processor.pad_id
        padded_all_attention_mask = torch.zeros((bs, max_length))
        for i, inputs_ids in enumerate(all_inputs_ids):
            padded_all_inputs_ids[i, -len(inputs_ids):] = inputs_ids
            padded_all_attention_mask[i, -len(inputs_ids):] = 1

        if self.args.test or self.args.func is not None:
            pass
        else:
            if padded_all_inputs_ids.shape[1] > self.args.max_seq_len:
                print('pad_input_ids: extend max_seq_len!!!') ## todo
                print(padded_all_inputs_ids.shape)

                num_start = padded_all_inputs_ids.shape[1] - self.args.max_seq_len
                padded_all_inputs_ids = padded_all_inputs_ids[:, num_start:]
                padded_all_attention_mask = padded_all_attention_mask[:, num_start:]

        return padded_all_inputs_ids.int(), padded_all_attention_mask.int()

    def mmu_collatexx(self, batch, pass_default=False):
        if pass_default:
            pass
        else:
            try:
                batch = default_collate(batch) #cpu
            except Exception as e:
                print(e)
                print(batch[0].keys())
                # print(batch[1].keys())
                traceback.print_exc()
                # import pdb;pdb.set_trace()
        return batch
    
    def mmu_collate(self, batch, pass_default=False):
        if pass_default:
            pass
        else:
            try:
                batch = default_collate(batch) #cpu
            except Exception as e:
                print(e)
                print(batch[0].keys())
                print(batch[1].keys())
                print(batch[0]['image'].shape)
                print(batch[0]['trans_image'].shape)
                print(batch[0]['trans_image_alpha'].shape)
                print(batch[0]['white_image'].shape)
                print(batch[1]['image'].shape)
                print(batch[1]['trans_image'].shape)
                print(batch[1]['trans_image_alpha'].shape)
                print(batch[1]['white_image'].shape)
                traceback.print_exc()
                import pdb;pdb.set_trace()

        if self.args.func == 'minicpm_cap':
            return batch
        # import pdb;pdb.set_trace()
        bs = len(batch['prompt'])

        ##### t2i
        all_inputs_ids = []
        for prompt in batch['prompt']:
            _, inputs_ids = self.wrap_t2i_prompt(prompt)
            all_inputs_ids.append(inputs_ids)
        t2i_inputs_ids, t2i_attention_mask = self.pad_input_ids(all_inputs_ids)
        t2i_attention_mask = torch.cat([t2i_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
        batch.update(dict(
            t2i_inputs_ids=t2i_inputs_ids,
            t2i_attention_mask=t2i_attention_mask
        ))

        ### uni
        all_inputs_ids = []
        for base_caption, grounding_prompt in zip(batch['base_caption'], batch['gt_grounding']):
            _, inputs_ids = self.wrap_uni_prompt(base_caption, grounding_prompt)
            all_inputs_ids.append(inputs_ids)
        # import pdb;pdb.set_trace()
        uni_inputs_ids, uni_attention_mask = self.pad_input_ids(all_inputs_ids)

        uni_attention_mask_image = torch.cat([uni_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
        batch.update(dict(
            uni_inputs_ids=uni_inputs_ids,
            uni_attention_mask=uni_attention_mask_image
        ))
        # uni_attention_mask_image: bs, seq

        # uni_stage1
        all_inputs_ids = []
        for base_caption, grounding_prompt in zip(batch['base_caption'], batch['gt_grounding']):
            _, inputs_ids = self.wrap_uni_prompt(base_caption, "<grounding>", in_stage1=True)
            all_inputs_ids.append(inputs_ids)
        uni_inputs_ids, uni_attention_mask = self.pad_input_ids(all_inputs_ids)
        batch.update(dict(
            uni_stage1_inputs_ids=uni_inputs_ids,
            uni_stage1_attention_mask=uni_attention_mask
        ))

        ### mmu
        all_prepares = []
        image = batch['image']
        answer = batch['prompt']
        question = "Please describe this image and then give the description and bounding box of each object in the image."
        for i in range(len(image)):
            conversation = [
                {"role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image[i:i+1]],},
                {"role": "<|Assistant|>", "content": f"{answer[i]}"},
            ]
            prepare = self.vl_chat_processor.process_one(
                prompt=None, 
                conversations=conversation, 
                images=image[i:i+1]
            )
            all_prepares.append(prepare)
        prepare_inputs = self.vl_chat_processor.batchify(all_prepares)
        batch.update(dict(
            prepare_inputs=prepare_inputs,
        ))
 

        ## p2i
        prepares = []
        for sku_title, trans_image, prompt, word_list, history_images, history_titles in zip(batch['sku_title'], batch['trans_image'],batch['prompt'], batch['word_list'], batch['history_images'], batch['history_titles']):
            history_titles = history_titles.strip().split('<title_split>')
            prepare = self.wrap_p2i_prompt(history_images, sku_title, prompt, word_list=word_list, history_titles=history_titles, custom_token_num=self.custom_token_num)
            prepares.append(prepare)
        prepare_inputs = self.vl_chat_processor.custom_batchify(prepares, custom_token_num=self.custom_token_num)
        p2i_attention_mask = prepare_inputs['attention_mask']
        p2i_attention_mask_image = torch.cat([p2i_attention_mask, torch.ones((bs, self.image_token_num_per_image))], dim=-1)
        prepare_inputs['attention_mask'] = p2i_attention_mask_image
        
        batch.update(dict(
            p2i_prepare_inputs=prepare_inputs
        ))

        # p2i_stage1
        prepares = []
        for sku_title, trans_image, prompt, word_list, history_images, history_titles in zip(batch['sku_title'], batch['trans_image'],batch['prompt'], batch['word_list'], batch['history_images'], batch['history_titles']):
            history_titles = history_titles.strip().split('<title_split>')
            prepare = self.wrap_p2i_prompt(history_images, sku_title, '<prompt>', in_stage1=True, word_list=word_list, history_titles=history_titles, custom_token_num=self.custom_token_num)
            prepares.append(prepare)
        prepare_inputs = self.vl_chat_processor.custom_batchify(prepares, custom_token_num=self.custom_token_num)

        batch.update(dict(
            p2i_stage1_prepare_inputs=prepare_inputs
        ))

        ### mmu_infer
        all_prepares = []
        image = batch['image']
        answer = batch['prompt']
        question = "Please describe this image and then give the description and bounding box of each object in the image."
        for i in range(len(image)):
            conversation = [
                {"role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image[i:i+1]],},
                {"role": "<|Assistant|>", "content": ""},
            ]
            prepare = self.vl_chat_processor.process_one(
                prompt=None, 
                conversations=conversation, 
                images=image[i:i+1]
            )
            all_prepares.append(prepare)
        prepare_inputs = self.vl_chat_processor.batchify(all_prepares)
        batch.update(dict(
            prepare_inputs_infer=prepare_inputs,
        ))
        return batch
    
    def setup_data(self, accelerator):
        args = self.args
        if args.debug:
            args.max_val_len = 1

        test_dataset, test_dataloader = get_dataset(
            args,
            args.test_data.data_name, 
            args.test_data.batch_size, 
            is_test=True,
            collate_fn=self.mmu_collate,
        )

        test_dataloader = accelerator.prepare(test_dataloader)
        print(f"test_dataloader: {args.test_data.data_name}, {len(test_dataloader)}")
        self.test_dataset = test_dataset
        self.test_dataloader = test_dataloader

        if self.args.test or self.args.func is not None:
            train_dataset = test_dataset
            train_dataloader = test_dataloader
            self.train_dataset = train_dataset
            self.train_dataloader = train_dataloader
        else:
            iterables_train = {}
            flow2task = {}
            train_datasets = []
            dataset_dict = {}
            for flow_id, data_item in enumerate(args.train_data):
                if self.args.debug:
                    data_item.batch_size = 2
                if self.args.no_full or self.args.debug:
                    if data_item.data_name == 'hico_full':
                        data_item.data_name = 'hico'
                    elif isinstance(data_item.data_name, list):
                         for i in range(len(data_item.data_name)):
                            if data_item.data_name[i] == 'hico_full':
                                data_item.data_name[i] = 'hico'

                dataset_same = dataset_dict.get(str(data_item.data_name), None)
                train_dataset, train_dataloader = get_dataset(
                    args,
                    data_item.data_name, 
                    data_item.batch_size, 
                    collate_fn=self.mmu_collate,
                    dataset=dataset_same,
                )
                dataset_dict[str(data_item.data_name)] = train_dataset

                train_dataloader = accelerator.prepare(train_dataloader)

                print(f"\ntrain_dataset_{flow_id}: {data_item.data_name}, {len(train_dataset)}")
                iterables_train[flow_id] = train_dataloader
                flow2task[flow_id] = data_item.task_type

                train_datasets.append(train_dataset)

            self.flow2task = flow2task

            train_dataloader = CombinedLoader(iterables_train, mode="min_size")
            train_dataloader = iter(train_dataloader)

            print(f"\nAll len(train_dataloader): {len(train_dataloader)}")

            self.train_dataset = train_dataset
            self.train_dataloader = train_dataloader

        return train_dataloader, train_dataset

    def validation(
        self, 
        global_step=0, 
        accelerator=None, 
        test_mode=False,
        val_num=None
    ):
        args = self.args
        test_mode = args.test or test_mode
        val_num = val_num or args.max_test_len

        if test_mode:
            patha = osp.join(args.output_dir, 'test', f"{args.test_data.data_name}_{args.test_data.task_type}_{val_num}")
            path = osp.join(patha, f"{global_step}")
            batch_path = osp.join(patha, f"{global_step}_batch")
            mkdir(osp.join(path, "gt_image"))
            mkdir(osp.join(path, "pr_image"))
            mkdir(osp.join(path, "image_ids"))
            mkdir(osp.join(path, "gt_image_ids"))
            mkdir(osp.join(path, "trans_image"))
            mkdir(osp.join(path, "fusion_image"))
        else:
            path = osp.join(args.output_dir, 'val')
            batch_path = path
        mkdir(path)
        mkdir(batch_path)

        kwargs = {}
        func = self.uni_generate
        if args.test_data.task_type == 't2i':
            kwargs.update(pred_layout=False)
            kwargs.update(use_uni_prompt_in_t2i=False)
        elif args.test_data.task_type == 'uni_2stage':
            pass
        elif args.test_data.task_type == 'uni':
            kwargs.update(pred_layout=False)
        elif args.test_data.task_type == 'mmu':
            kwargs.update(pred_image=False)
            kwargs.update(is_mmu=True)
        elif args.test_data.task_type == 'plan':
            kwargs.update(pred_image=False)
        elif args.test_data.task_type == 'p2i':
            func = self.uni_generate_p2i
            pass
        else:
            assert False
        rets = []
        for idx, batch in enumerate(tqdm(self.test_dataloader)):
            if val_num != -1 and idx >= val_num: break
            if idx >= self.args.test_start:
                pass
            else:
                continue
            
            batch_str = f'{idx}' if test_mode else f'{global_step}_{idx}'

            print(batch['prompt'])
            
            out = func(
                batch=batch, 
                batch_idx=batch_str,
                gen_path=batch_path, 
                accelerator=accelerator,
                parallel_size=1,
                **kwargs,
            )

            if not test_mode:
                break

            gt_image = batch['image']
            image_id = batch['image_id']
            trans_image = batch['trans_image_alpha']
            edited_image = batch.get('edited_image', None)
            H = batch['H']
            W = batch['W']
            pr_image = out['pr_image']
            pr_grounding = out['pr_grounding']

            bs = gt_image.shape[0]

            for i in range(len(gt_image)):
                if image_id[i] != '':
                    to_pil(denorm_pt(pr_image[i])).save(f"{path}/image_ids/{image_id[i]}.jpg")
                    to_pil(denorm_pt(gt_image[i])).save(f"{path}/gt_image_ids/{image_id[i]}.jpg")

                p = self.args.parallel_size
                if p > 1:
                    for t in range(self.args.parallel_size):
                        to_pil(denorm_pt(pr_image[i*p+t])).save(f"{path}/pr_image/{idx*bs+i}_{t}.png")
                        fusion_image = self.fuse_image(pr_image[i*p+t], trans_image[i])
                        to_pil(denorm_pt(fusion_image)).save(f"{path}/fusion_image/{idx*bs+i}_{t}.png")
                else:
                    to_pil(denorm_pt(pr_image[i*p])).save(f"{path}/pr_image/{idx*bs+i}.png")
                    fusion_image = self.fuse_image(pr_image[i*p], trans_image[i])
                    to_pil(denorm_pt(fusion_image)).save(f"{path}/fusion_image/{idx*bs+i}.png")
                    
                to_pil(denorm_pt(gt_image[i])).save(f"{path}/gt_image/{idx*bs+i}.png")
                to_pil(trans_image[i]).save(f"{path}/trans_image/{idx*bs+i}.png")

                if edited_image is not None:
                    mkdir(f"{path}/edited_image/")
                    to_pil(denorm_pt(edited_image[i])).save(f"{path}/edited_image/{idx*bs+i}.png")

    @property
    def device(self,):
        return self.get_device(self.vl_gpt)

    @property
    def dtype(self):
        return self.get_dtype(self.vl_gpt)
    