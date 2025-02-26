For all samples the data normalization was performed using the mean and standard deviation of the dataset.
  ```bash
  transform = transforms.Compose([
                  transforms.Resize(224,224), interpolation=Image.LANCZOS), 
                  transforms.ToTensor(),
                  transforms.Normalize([0.5990, 0.3664, 0.2769], [0.2847, 0.2190, 0.1772])
                  ]) 
  ```


# 1. Videoendoscopies
1. **ViT’s patch-based linear projection (16×16×3):** Run script open a terminal and navigate to the directory containing `vit_patch_linear.py` and define the parameters:

    ```bash
    OUTPUT_DIR='..\data\15 FPS'
    DATA_PATH='..\data\Videoendoscopies'
    DATA_SPLIT='..\official_splits\gastrohun-videoendoscopy-metadata.json'

    !python vit_patch_linear.py \
    --num_workers 0 \
    --input_size 16 \
    --batch_size 15 \
    --data_path ${DATA_PATH}  \
    --output_dir ${OUTPUT_DIR} \
    --official_split ${DATA_SPLIT}  
    ```
2. **ConvNeXt-Tiny pretrained on ImageNet:** Run script open a terminal and navigate to the directory containing `vit_patch_linear.py` and define the parameters:

    ```bash
    OUTPUT_DIR='..\data\15 FPS'
    DATA_PATH='..\data\Videoendoscopies'
    DATA_SPLIT='..\official_splits\gastrohun-videoendoscopy-metadata.json'

    !python vit_patch_linear.py \
    --num_workers 0 \
    --input_size 16 \
    --batch_size 15 \
    --data_path ${DATA_PATH}  \
    --output_dir ${OUTPUT_DIR} \
    --official_split ${DATA_SPLIT}  
    ```
2. **ConvNeXt-Tiny pretrained on Endoscopy:** Run script open a terminal and navigate to the directory containing `vit_patch_linear.py` and define the parameters:

    ```bash
    OUTPUT_DIR='..\data\15 FPS'
    DATA_PATH='..\data\Videoendoscopies'
    DATA_SPLIT='..\official_splits\gastrohun-videoendoscopy-metadata.json'

    !python vit_patch_linear.py \
    --num_workers 0 \
    --input_size 16 \
    --batch_size 15 \
    --data_path ${DATA_PATH}  \
    --output_dir ${OUTPUT_DIR} \
    --official_split ${DATA_SPLIT}  
    ```


### Note:
1. Check the existence of the paths of: 
   - DATA_PATH
   - MODEL_PATH
   - DATA_SPLIT 

