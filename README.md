# SNanimeGAN
anime GAN with spectral normalization

listed below are the experiments at 218th epoch
![218th epoch v1003 SNGANanime256](https://user-images.githubusercontent.com/44658049/115674086-4432ed00-a388-11eb-8827-c22fd05cf0a4.png)
I don't remember what is the difference between version 1002 and 1003, but they did perform differently
![218 th epoch v1002 SNGANanime256](https://user-images.githubusercontent.com/44658049/115674261-6f1d4100-a388-11eb-9113-4cddcfb02666.png)

it will be updated when the results at further epoch are obtained


two new version of codes is used to train
![ v1008 217th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/116780902-57496980-aaba-11eb-9157-eff9ddccecd1.png)
![ v1008 218th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/116780904-59132d00-aaba-11eb-9033-2bc87c78b972.png)
![ v1008 219th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/116780906-5adcf080-aaba-11eb-8210-9509737defcf.png)
here are the generated images at 217th 218th 219 iterations of code of version 1008

I found a mistake in versions before 1004 which used avg_pool as globle sum layer, and thus even at 218th epoch, it still did not gain any good performance 
