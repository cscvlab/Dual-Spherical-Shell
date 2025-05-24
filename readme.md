# Dual Spherical Shell (DSS)

This repository is the official implementation of our paper, accepted by IEEE International Conference on Multimedia and Expo (**ICME 2025**):

**Neural Implicit Reconstruction and Fast Rendering Based on Dual Spherical Shell**

__Authors:__ Zijian Wang, Yuqi Liu, Yan Zhao, Binghao Wang, Shen Cai*, Yanting Zhang.

**Links:**  [[Project Page]](www.cscvlab.com) [[Video(bilibili)]](https://www.bilibili.com/video/BV17pdxYJEYM/) [[Video(Youtube)]](https://www.youtube.com/watch?v=oB-wbv7FWp8)

## Method

### Core idea in one sentence
Given a number of pre-computed concentric spheres, local SDF fitting within DSS is enabled, and early termination as well as parallel sphere tracing are facilitated for more efficient SDF rendering.

### Local SDF Fitting within DSS

<p align="center">
 <img src="imgs/localFitting.png" width = "800" alt="ifps" align=center />
</p>

### Early Termination and Parallel Sphere Tracing (S.T.)

<p align="center">
 <img src="imgs/parallelST.png" width = "700" alt="ifps" align=center />
</p>

### Advantages

1. **High-fidelity Reconstruction**: low reconstruction error, in terms of chamfer distance.
2. **Memory and Storage Efficiency**: small number of pre-computed geometric primitives.
3. **Rendering acceleration**: early termination and parallel sphere tracing (brings about 40% speed-up).
4. **NeuS improvement**: new sampling strategy; improvements in both accuracy and speed.

## Dataset
We use Thingi10k and NeRF synthetic datasets, both of which are available from their official website.
### Thingi10k
You can download them at https://ten-thousand-models.appspot.com/
### NeRF synthetic
https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1

## Getting started
### Python dependencies
```
conda env create -f environment.yml
conda activate kaolin_test
pip install torch==1.8.0+cu111 torch-cluster==1.5.9 torch-geometric==1.4.1 torch-scatter==2.0.6 torch-sparse==0.6.10 torch-spline-conv==1.2.1

cd ./submodules/miniball
python setup.py install
cd ..
cd ./kaolin_sphere-0.9.1
python setup.py develop
cd ..
cd ./libigl/python
python setup.py
cd ..
cd ..
cd ./geolab-copy
cmake . -B build
cmake --build build 
```

### Training
```
python train.py
```
### Evaluation
```
python eval.py
python eval_ssim.py
```

## Third-Party Libraries

This code includes code derived from 3 third-party libraries

https://github.com/nv-tlabs/nglod <br>
https://github.com/u2ni/ICML2021 <br>
https://github.com/NVIDIAGameWorks/kaolin <br>

## License
This project is licensed under the terms of the LGPL License (see `LICENSE` for details).

## Citation

```bibtex
@inproceedings{Ding2025IFPS,
  title={Neural Implicit Reconstruction and Fast Rendering Based on Dual Spherical Shell},
  author={Wang, Zijian and Liu, Yuqi and Zhao, Yan and Wang, Binghao and Cai, Shen and Zhang, Yanting},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)}, 
  year={2025},
}
```
