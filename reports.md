**Индивидуальное задание** --- разработка dewrapper.
<!-- blank line -->
----
<!-- blank line -->
#### Отчёт от 20.03.2024
1. Добавлены новые размеченные данные (суммарно на текущий момент получается 87 изображений с разметкой).
2. **По индивидуальному заданию**:
   
   Постановка задачи:
   
   - **Данные** представляют собой набор пар изображений, на одном из которых смятый штрихкод, на другом --- разглаженный.
   - **Оценка качества** в базовом варианте будет оцениваться визуально по разглаженным с помощью нейронной сети изображениям.

Сейчас разбираюсь с [кодом](https://github.com/mhashas/Document-Image-Unwarping-pytorch/tree/master) [из статьи](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_DocUNet_Document_Image_CVPR_2018_paper.pdf). 
(Разобралась с запуском, теперь разбираюсь с тем, что он делает) 

**Ближайшие планы**: 
1. Для кода выше:
   попробовать разные архитектуры нейросетей (DEEPLAB, DEEPLAB_50, DEEPLAB_34, DEEPLAB_18, DEEPLAB_MOBILENET, DEEPLAB_MOBILENET_DILATION, UNET, UNET_PAPER, UNET_PYTORCH, PSPNET) и лоссы (DOCUNET_LOSS, SSIM_LOSS, SSIM_LOSS_V2, MS_SSIM_LOSS, MS_SSIM_LOSS_V2, L1_LOSS, SMOOTH_L1_LOSS, MSE_LOSS).
<!-- blank line -->
----
<!-- blank line -->
