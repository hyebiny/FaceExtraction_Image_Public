# Face Matting 
The goal is matting occlusions that is located in front of human face.
To find out which occlusion generation technique is the best, I utilized serverl matting networks. 

* ResNet18
* <a href="https://github.com/yucornetto/MGMatting" target="_blank">MGMattings</a> (mask-guided)
* <a href="https://github.com/qlyoo/aematter" target="_blank">AEMatter</a> (trimap-guided)
* <a href="https://github.com/ZHKKKe/MODNet" target="_blank">MODNet</a> (trimap-free)
* <a href="https://peterl1n.github.io/RobustVideoMatting/#/" target="_blank">RobustVideoMatting</a> (trimap-free, video-based)



# Occlusion Matting Dataset
For train, valid, and test dataset, I utilized several public dataset.

### Face Dataset
* <a href="https://github.com/tkarras/progressive_growing_of_gans" target="_blank">CelebAHQ</a>
* <a href="https://celebv-hq.github.io/" target="_blank">CelebAHQ-video</a>
### Matting Dataset
* <a href="https://github.com/nowsyn/SIM" target="_blank">SIM</a>
* <a href="https://github.com/JizhiziLi/GFM" target="_blank">AM2k</a>
### Hand Dataset
* <a href="https://sites.google.com/view/11khands" target="_blank">11k</a>
* <a href="https://github.com/MandyMo/HIU-DMTL" target="_blank">hiu</a>
### Occlusion Segmentation Paper
* <a href="https://github.com/face3d0725/FaceExtraction" target="_blank">FaceOcc</a>
* <a href="https://github.com/kennyvoo/face-occlusion-generation" target="_blank">NatOcc, RandOcc</a>



# Compare the results

|           | MSE   | SAD    |
|-----------|-------|--------|
| resnet18  | 0.068 | 27.161 |
| MGMatting | 0.042 | 17.689 |
| AEMatter  | 0.321 | 89.271 |
| MODnet    | 0.035 | 14.600 |
| RVM       | 0.015 | 2.312  |

If you want to see the full image comparison, visit other repository &rarr; <a href="https://github.com/kennyvoo/face-occlusion-generation" target="_blank">Video_Face_Matting_Public</a>

# Train
