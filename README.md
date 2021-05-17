# ImprovedGANS

The Implementation of improved techniques for training GANS as in Salimans et. al 2015
Includes semi-supervised gans.

# Contents
# 1. Plain Semi-Supervised GAN or SGAN
# 2. SGAN with feature matching loss.
# 3. SGAN with mini-batch discrimination.
(The latter performs poorly on the semi supervised task)

Salimans et. al introduced many imroved techniques for stable training of Generative Adversarial Networks, some of those techniques will be found here.
The result images are found in the images folder.
The images generated by mini-batch discrimination have a relatively good quality but the other task is compromised. The Semi-supervised task is achieved with a good accuracy by using feature matching or plain loss.