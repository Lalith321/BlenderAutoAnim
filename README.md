# BlenderAutoAnim

This project is aimed to create a complete animated sequence of dance primarily using generative AI.
I have used Deep convolutional neural network that uses functional pipeline to remove the unnecessary windows to be calculated. The CNN requires an image data as input for training and the bones data that we get is in form of vector. Each of those values resemble quaternion,x,y,z. in a 3 dimensional space. Then we also have two other things to consider that is timestamp and bone number. All this make up a 5D tensor.


The complete set of movements here seem to have moved to fast and thus it appears to be a glitch and as the time passes the rig and anim forms an un natural shape/pose.
This might be due to lesser convolution samples which also corresponds to bigger windows. This might give us a largely differentiating values. Having a large distance between each keyframe of animation while the time difference is less or same tends to speedup the entire animation.

https://github.com/Lalith321/BlenderAutoAnim/assets/51789163/fdcc7363-48e5-4508-9e97-864ab56e02f8

Below is a pose from one of the videos that closely resembles a dance move that has negligible errors.

![image](https://github.com/Lalith321/BlenderAutoAnim/assets/51789163/7b383639-7b29-469a-add6-70604cb46613)

While the image here has its leg completely twisted.

![image](https://github.com/Lalith321/BlenderAutoAnim/assets/51789163/1d2b7757-3705-428a-a0f3-fbc34000a042)

I am still working on this to get desirable result.
