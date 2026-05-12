set -x


# inference data_set formate:
# image_idx  + \t + image_encode_by_base64

# encode example:
# image=Image.open(image_string).convert("RGB") #png
# image.save(buffer,format="JPEG")
# code_base64=base64.urlsafe_b64encode(buffer.getvalue())
# buffer.close()

python -m torch.distributed.launch --nproc_per_node=1 main_inference.py \
    --data_path ./your_inference_data_path \
    --nb_knn 20 \
    --pretrained_weights /finetuned_model_path/moco_checkpoint.pth \
    --arch vit_base_with_compress \
    --patch_size 16 \
    --moco_dim=64 \
    --batch_size_per_gpu 2048 \
    --checkpoint_key "student" \
    --dump_features ./models \
    --part_index part-extra-${inference_day}