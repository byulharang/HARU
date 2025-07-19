# HARU
**Hierarchical cross-Attenion Resnet based U-net** <br> for **depth map inpainting** with partial panorama and ambisonic room impulse response

We propose **novel model** for **novel task** with **physical analysis** aligned with acoustic theory

**Key phrases:** *Inpainting, RIR, Depth Map, Cross-Attention, U-Net, Acoustic Interpretability*

---

**2025 URP Winter/Spring** <br>
**Advisor:** Jung-Woo Choi (KAIST EE) <br>
**Final Paper for URP:** 🧾[PDF](https://drive.google.com/file/d/16QuwoCr_mHgrDiSO7QP3mxkWWsUO02gW/view?usp=sharing) *editted with **CVPR** format* <br> 
**Final Presentation:** 🔬 [Google Slide](https://docs.google.com/presentation/d/1msZiOX9Xy62TaJzaRabDi3YwDLrduo0H3MF4yPv1NOQ/edit?usp=sharing) <br>
**Data & Experiment Logs:** 🌐 [Notion](https://kiwi-primrose-e33.notion.site/URP-16d30761238f8068aec6f9576ef4bee2?source=copy_link)

---

# Abstract
﻿Reconstructing the indoor structures is crucial for augmented/extended reality (AR/XR) applications or interactions. The structural information can be obtained from indoor depth panoramas; however, acquiring a complete depth panorama under typical conditions remains challenging due to the limited field of view. We propose the novel model that reconstruct the full depth panorama from the partial panorama and room impulse response. The model acheive about 20 dB and 0.844 for peak signal-to-noise ratio (PSNR) and structual similiarity index map (SSIM) evaluation metrics, respectively.
 
<img src="https://github.com/byulharang/HARU/blob/main/images/Architecture.png" alt="Proposed Model Flow" width="900" />

# Result 
Our proposed transformer(TR) based model outperform the CNN based model (from 음향학회) in
* **Peak-signal-to-noise (PSNR):** High value refer better image quality
* **Structural similiarity index map (SSIM):** High value refer better structural, luminance, contrast quality

<img src="https://github.com/byulharang/HARU/blob/main/images/metric.png" alt="PSNR and SSIM" width= "800" />

* **Perceptual Quality with naive eyes:** TR based model estimate plausible result
  * *Kindly ask the GT is little different by uncontrolled randomness*

<img src="https://github.com/byulharang/HARU/blob/main/images/perceptual%20result.png" alt="Perceptual Result Comparison" width="900" />

# Future Work
* We consider **Diffusion or Flow** based model with HARU as the noise estimator or vector field function respectively <br>
* Plug in the room material segmentation with regards of sound absorb coefficents. 
* Generalize model with reverberent speech and other type of multi channel reciever (not only FoA, *first order ambisonic*)
    * Considering Sound Enhancement model like [DeFTAN II](https://arxiv.org/abs/2308.15777) or Dereverberation model 
* Extend acoustic based analysis and compare with other similiar task-purposed network
    * 🗒️ [VisualEcho](https://arxiv.org/abs/2005.01616)
    * 🗒️ [Beyond Image to Depth](https://arxiv.org/abs/2103.08468)
    * 🗒️ [BatVision](https://arxiv.org/abs/1912.07011)
     
Still, the other works use sound as Lidar system, **while we concentrate on the acoustic property of RIR**
*Representative properties: direct sound, early reflection, EDC curve slope, RT60, C50, etc.)

# Dataset

RIR and corresponding Panorama provided by <br>
* ⚙️ [Soundspace2.0](https://arxiv.org/abs/2206.08312)
* 📊 [Matterport3D](https://arxiv.org/abs/1709.06158)
