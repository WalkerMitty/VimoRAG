# Dataset and Models

## Stage 1 (Our models)
- Download the dataset and models from [HuggingFace](https://huggingface.co/datasets/Haidong2/VimoRAG) (or [ModelScope](https://modelscope.cn/models/Walkerhai/VimoRAG)) and put them in ``data/``
- Run

```shell
cd data/VimoRAG
unzip McDPO.zip
unzip resources.zip
mv McDPO/checkpoints code/VimoRAG/McDPO/
mv output code/VimoRAG/
mv resources code/VimoRAG
```

## Stage 2 (Pretrained models)
- Download the following three models — [InternVideo2](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4), [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336), and [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) — and place them in the `./resources/playground` directory. The resulting file structure should be as follows:

```
├── playground
│   ├── InternVideo2-Stage2_1B-224p-f4
│   ├── models
│   │   ├── mlp2x_gelu_clip_l14_336px
│   │   └── mlp2x_gelu_internvideo2
│   ├── openai
│   │   └── clip-vit-large-patch14-336
│   └── Phi-3-mini-4k-instruct
```

## Stage 3 (Optional)
- We have already prepared a small subset of the video database in `vimorag_models/resources/motion-x++-videos/haa500`. This subset is sufficient for running the demo. 
- However, if you intend to train your own model, you will need the full retrieval database. We provide download links to the original open-source datasets below. 
- Want to download all the videos in one click? Please send me (182haidong ``at`` gmail.com) copies of the license agreements you’ve obtained for all the datasets, and we’ll provide you with a download link.

- The final directory structure should look like this:
```
└── resources
    ├── motion-x++-videos
    │   └── subset
    │       ├── animation
    │       ├── haa500
    │       ├── humman
    │       ├── idea400
    │       ├── kungfu
    │       ├── music
    │       └── perform
    └── wild_motion_videos
        ├── aslan
        ├── hmdb51
        │   └── mp4_version
        ├── kinetics-400
        │   └── kinetics-400-targz
        │       ├── test
        │       ├── train
        │       └── val
        ├── ntu-rgb
        │   └── avi_version
        ├── penn_action
        │   └── mp4_version
        └── ucf-101
            └── mp4_version
```

Here are the links:

- [motion-x++-videos](https://github.com/IDEA-Research/Motion-X)
- [aslan](https://talhassner.github.io/home/projects/ASLAN/ASLAN-main.html)
- [hmdb51](https://huggingface.co/datasets/SabrianLinnn/hmdb51)
- [kinetics-400](https://github.com/cvdfoundation/kinetics-dataset)
- [ntu-rgb](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- [penn_action](https://dreamdragon.github.io/PennAction/)
- [ucf-101](https://www.crcv.ucf.edu/data/UCF101.php)


# Acknowledgements

We sincerely thank the authors and contributors of the following open-source models and datasets for their valuable work:  
- [MotionBERT](https://github.com/Walter0807/MotionBERT)  
- [MotionGPT](https://github.com/qiqiApink/MotionGPT)  
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
