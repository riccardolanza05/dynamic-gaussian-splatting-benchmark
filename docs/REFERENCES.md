# References

Papers studied or benchmarked in this project. The PDFs are not redistributed here; follow the links. A one-page comparison of the five dynamic methods (contributions, limitations, datasets, metrics, acquisition requirements, viewer availability) is in [papers_comparison_table.pdf](papers_comparison_table.pdf).

| # | Paper | Venue | Links | Role in this project |
|---|---|---|---|---|
| 1 | Kerbl, Kopanas, Leimkühler, Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering* | ACM TOG (SIGGRAPH) 2023 | [arXiv:2308.04079](https://arxiv.org/abs/2308.04079) · [code](https://github.com/graphdeco-inria/gaussian-splatting) | static foundation of all methods |
| 2 | Luiten, Kopanas, Leibe, Ramanan. *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis* | 3DV 2024 | [arXiv:2308.09713](https://arxiv.org/abs/2308.09713) · [code](https://github.com/JonathonLuiten/Dynamic3DGaussians) | studied; multi-view only, not benchmarked |
| 3 | Yang, Gao, Zhou, Jiao, Zhang, Jin. *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction* | CVPR 2024 | [arXiv:2309.13101](https://arxiv.org/abs/2309.13101) · [code](https://github.com/ingra14m/Deformable-3D-Gaussians) | benchmarked (Deformable-3DGS) |
| 4 | Wu, Yi, Fang, Xie, Zhang, Wei, Liu, Tian, Wang. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering* | CVPR 2024 | [arXiv:2310.08528](https://arxiv.org/abs/2310.08528) · [code](https://github.com/hustvl/4DGaussians) | benchmarked (4DGaussians) |
| 5 | Yang, Yang, Pan, Zhang. *Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting* | ICLR 2024 | [arXiv:2310.10642](https://arxiv.org/abs/2310.10642) · [code](https://github.com/fudan-zvg/4d-gaussian-splatting) | benchmarked (4DGS native-4D) |
| 6 | Yang, Pan, Zhu, Zhang, Feng, Jiang, Torr. *4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives* | arXiv 2024 (extended version of 5) | [arXiv:2412.20720](https://arxiv.org/abs/2412.20720) | extended description of the native-4D method |
| 7 | Li, Chen, Li, Xu. *Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis* | CVPR 2024 | [arXiv:2312.16812](https://arxiv.org/abs/2312.16812) · [code](https://github.com/oppo-us-research/SpacetimeGaussians) | studied; not benchmarked |
| 8 | Pumarola, Corona, Pons-Moll, Moreno-Noguer. *D-NeRF: Neural Radiance Fields for Dynamic Scenes* | CVPR 2021 | [arXiv:2011.13961](https://arxiv.org/abs/2011.13961) | source of the benchmark dataset |

Community code used in the fudan-zvg notebook: the `render.py` and visualisation approach of [fudan-zvg/4d-gaussian-splatting PR #60](https://github.com/fudan-zvg/4d-gaussian-splatting/pull/60).

## BibTeX

```bibtex
@article{kerbl3Dgaussians,
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal = {ACM Transactions on Graphics},
  volume  = {42},
  number  = {4},
  year    = {2023}
}
@inproceedings{luiten2024dynamic,
  title     = {Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis},
  author    = {Luiten, Jonathon and Kopanas, Georgios and Leibe, Bastian and Ramanan, Deva},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2024}
}
@inproceedings{yang2024deformable,
  title     = {Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
  author    = {Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
@inproceedings{wu20244dgaussians,
  title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
  author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
@inproceedings{yang2024real,
  title     = {Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting},
  author    = {Yang, Zeyu and Yang, Hongye and Pan, Zijie and Zhang, Li},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
@article{yang20244dgs_native,
  title   = {4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives},
  author  = {Yang, Zeyu and Pan, Zijie and Zhu, Xiatian and Zhang, Li and Feng, Jianfeng and Jiang, Yu-Gang and Torr, Philip H. S.},
  journal = {arXiv preprint arXiv:2412.20720},
  year    = {2024}
}
@inproceedings{li2024spacetime,
  title     = {Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis},
  author    = {Li, Zhan and Chen, Zhang and Li, Zhong and Xu, Yi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
@inproceedings{pumarola2021dnerf,
  title     = {D-NeRF: Neural Radiance Fields for Dynamic Scenes},
  author    = {Pumarola, Albert and Corona, Enric and Pons-Moll, Gerard and Moreno-Noguer, Francesc},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```
