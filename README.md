## VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models

## 📰 News

- **The training, inference and visualization codes are released.** 2025-10
- **🎉🎉🎉 The paper is accepted by NeurIPS 2025.**

## 📂 README Overview


- [VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](#vimorag-video-based-retrieval-augmented-3d-motion-generation-for-motion-language-models)
- [📰 News](#-news)
- [📂 README Overview](#-readme-overview)
- [🎮 Demo](#-demo)
- [🛠️ Full Pipeline](#️-full-pipeline)
  - [Resources](#resources)
  - [Retrieval](#retrieval)
  - [Generation](#generation)
- [📊 Evaluation](#-evaluation)
- [🏋️ Training](#️-training)
  - [stage 1](#stage-1)
  - [stage 2](#stage-2)
- [Acknowledgements](#acknowledgements)

## 🎮 Demo

- Step1: Resources

Download the resources from [Dataset README](./readme_dataset.md).
- Step2: Environment

```shell
cd McDPO
conda env create -f environment.yml
conda activate mcdpo
bash additional_env.sh

```
- Step3: Run
```shell
# merge lora for sft model
python llm_inference.py --merge_lora --model_base ../resources/playground/Phi-3-mini-4k-instruct --model_path ../output/sft_model --out_dir ../output/sft_model/merged_lora

# inference
python llm_inference.py --retrieval_result ../Gemini-MVR/diy_output/retrieval_result.json --out_dir ../output --temperature 0.85 --lora --model_path ../output/dpo_model --llm_seed 2024 --model_base ../output/sft_model/merged_lora --demo_inference

## For visualization
python generate_motion.py --generated_file ../output/start-1.json --out_dir ../output/visual_output --render
```

## 🛠️ Full Pipeline
After you input a sentence, the system automatically retrieves a matching video and uses an LLM to produce high-quality 3D human motion.


### Resources

For detailed instructions, please refer to the [Dataset README](./readme_dataset.md).


### Retrieval

- Environment
```shell
cd Gemini-MVR
conda env create -f environment.yml
conda activate gemini-mvr

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

- Downloading Pretrained Models

> download [ViT-L-14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) 

> download InternVideo-MM-L-14.ckpt from [Here](https://huggingface.co/OpenGVLab/InternVideo1.0/tree/main)

```shell
mv ViT-L-14.pt VimoRAG/resources/
mv InternVideo-MM-L-14.ckpt VimoRAG/resources/
```
- Run
```shell
python prepare_input.py --text "The person is performing a punching motion while standing stationary. He is transitioning from a relaxed stance to a boxing stance, throwing a series of punches, and then returning to the relaxed stance."

bash eval_finetuned_scripts/diy_inference.sh 
```


### Generation
- Environment

```shell
cd McDPO
conda env create -f environment.yml
conda activate mcdpo
bash additional_env.sh

```
- Run
```shell
python llm_inference.py --retrieval_result ../Gemini-MVR/diy_output/retrieval_result.json --out_dir ../output --temperature 0.85 --lora --model_path ../output/dpo_model --llm_seed 2024 --model_base ../output/sft_model/merged_lora --demo_inference

## For visualization
python generate_motion.py --generated_file ../output/start-1.json --out_dir ../output/visual_output --render
```


## 📊 Evaluation

```shell
python evaluate_for_generated_results.py --generated_file ../resources/llm_generated_text/no_motion_r128_a256_bsz8x8_epoch2_new_llmseed2024_test_t2m/merge.json --split test --dataname t2m
```
## 🏋️ Training

### stage 1
Visual Demonstration-Enhanced Instruction Tuning


```shell
bash scripts/stage1.sh
```

- Merge Lora weight
```shell
# merge lora for stage 2
python llm_inference.py --merge_lora --model_base ../resources/playground/Phi-3-mini-4k-instruct --model_path ../output/sft_model --out_dir ../output/sft_model/merged_lora
```
### stage 2
Motion-centric Dual-alignment DPO

<details>
<summary><b>Dataset Preparation Steps (Click to Expand)</b></summary>

Sample the SFT model three times to obtain candidate data.

Note: This step is time-consuming, so we've prepared the data for you in advance. (Check the [Dataset README](./readme_dataset.md))
```shell
python llm_inference.py --retrieval_result ../resources/retrieval_inference_wild/train_t2m_top1_wild_new.json --seed 2024 --llm_seed 2024 --out_dir ../resources/llm_generated_text/no_motion_r128_a256_bsz8x8_epoch2_new_llmseed2024_train --temperature 0.9 --split train --lora --model_path ../output/sft_model --model_base ../resources/playground/Phi-3-mini-4k-instruct
```

- Generate the preference data for McDPO training
```shell
python evaluate_for_generated_results.py --generated_file ../resources/dataset/t2m_r128_a256_bsz8x8_epoch2_new_train_3seed.json --fid_weight 0.9 --match_weight 0.1 --split train --dataname t2m --vqvae_path ../resources/pretrained_vqvae/t2m.pth --sft_file ../resources/dataset/train_top1_t2m_new.json --dpo_file ../resources/dataset/no_motion_r128_a256_bsz8x8_epoch2_new_train_3seed_self.json --dpo_selection
```
</details>


- Training

```shell
bash scripts/stage2.sh
```

## Acknowledgements

- [MotionGPT](https://github.com/qiqiApink/MotionGPT)
- [InternVideo](https://github.com/OpenGVLab/InternVideo)
- [VideoGPT-plus](https://github.com/mbzuai-oryx/VideoGPT-plus)
- [LLaVA-Hound-DPO](https://github.com/RifleZhang/LLaVA-Hound-DPO)