# SNanimeGAN
anime GAN with spectral normalization and hinge version of loss


listed below are the experiments at 218th epoch
![218th epoch v1003 SNGANanime256](https://user-images.githubusercontent.com/44658049/115674086-4432ed00-a388-11eb-8827-c22fd05cf0a4.png)
I don't remember what is the difference between version 1002 and 1003, but they did perform differently
![218 th epoch v1002 SNGANanime256](https://user-images.githubusercontent.com/44658049/115674261-6f1d4100-a388-11eb-9113-4cddcfb02666.png)

it will be updated when the results at further epoch are obtained


two new version of codes is used to train. that is 1007 and 1008
the difference between 1007 and 1008 is whether offical spectral norm in pytorch is used. the hand-written SN is used in 1007 and official SN is applied
in 1008 in order to compare and figure out if I make mistake in code of Spectral normalization.

217th 
![ v1008 217th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/116780902-57496980-aaba-11eb-9157-eff9ddccecd1.png)
218th 
![ v1008 218th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/116780904-59132d00-aaba-11eb-9033-2bc87c78b972.png)
219th 
![ v1008 219th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/116780906-5adcf080-aaba-11eb-8210-9509737defcf.png)
here are the images generated at 217th 218th 219 epoches of code of version 1008

I found a mistake in versions before 1006 which used avg_pool as globle sum layer for discriminator, and thus even at 218th epoch, it still did not gain any good performance 
