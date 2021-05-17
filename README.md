# SNanimeGAN
anime GAN with spectral normalization and hinge version of loss

code runs on python 3.6 + pytorch 1.8.1


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

I found a mistake in versions before 1006 which used avg_pool as global sum layer for discriminator, and thus even at 218th epoch, it still did not gain any good performance 

the result at 1990+ epoches 

![ v1008 1998th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/117770795-5362d780-b270-11eb-9e75-6558149f479f.png)
![ v1008 1999th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/117770803-55c53180-b270-11eb-953a-10bcff1478c5.png)
![ v1008 2000th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/117770814-5958b880-b270-11eb-9173-a73d5e4d0787.png)
![ v1008 1997th epoch  v1000a   SNGANanime256](https://user-images.githubusercontent.com/44658049/117770824-5d84d600-b270-11eb-9bb1-6fadf9e5916f.png)

image generated on v1015

![ v1015cp 188th epoch  v1000b   SNGANanime128](https://user-images.githubusercontent.com/44658049/118420953-044af580-b6fb-11eb-8e8b-a4576e34f47c.png)
![ v1015cp 189th epoch  v1000b   SNGANanime128](https://user-images.githubusercontent.com/44658049/118420955-057c2280-b6fb-11eb-9598-1791e1cdb869.png)
![ v1015cp 187th epoch  v1000b   SNGANanime128](https://user-images.githubusercontent.com/44658049/118420958-0614b900-b6fb-11eb-9669-4a2f3ae6b7e3.png)

on v1011

![ v1011cp 361th epoch  v1000b   SNGANanime128](https://user-images.githubusercontent.com/44658049/118420968-0dd45d80-b6fb-11eb-9dba-766a00a0720b.png)
![ v1011cp 360th epoch  v1000b   SNGANanime128](https://user-images.githubusercontent.com/44658049/118420971-0f058a80-b6fb-11eb-8442-d0e066063b6f.png)


on v1016

![ v1016CP 172th epoch  v1000b   SNGANanime128](https://user-images.githubusercontent.com/44658049/118420999-1cbb1000-b6fb-11eb-8946-047848f57e67.png)
![ v1016CP 173th epoch  v1000b   SNGANanime128](https://user-images.githubusercontent.com/44658049/118421002-1dec3d00-b6fb-11eb-8d56-8e16145efa8f.png)



