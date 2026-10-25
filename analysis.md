Done in 2 parts:
Trained: took 26hrs
Tested: Using Live web cam.
Mistakes: Using too many classes, so testing of each class not possible. 
Analysis: 
- Too many classes makes it difficult to analyse the results for each class, but the confusion matrix produced shows that not all classes had good outcome, some better than other
- Live Web cam testmade me realize that depending on the dataset, the same object(in my case human) depending on the distance from the camera, detects different objects.
- Moving further to closer to the camera, it identifies different objects. 

25m - Family snapshot / Pet (yeah it is identifying me as a pet, guess we have a lot in common) + hand-> 
10m - glasses + hand or palm sometimes ->
5m - teeth palm

- My laptop webcam and the lighting is bad. Depending on the occlusion, model is able to identify or mis idendify the same object in each frame even if every thing is constant.
- Quality of camera is very imp to the model working its best regardless of how it is trained. 
- Some object dataset (eg: toilet paper) had less filtered images (images that are not toilet paper were mixed in) making detection of that object(eg: toilet paper) more difficult.
- I assumed model is associating objects with how it has "learned" about them from the images, but it more so learned the features of the object. Example glasses on table (most of the images in dataset) was idenfied correctly irrespective if i was wearing them or if it was kept the table. But glasses laying sideways not exactly looking like glasses was not getting identified at all.
- It was able to identify a lot of objects through the mini tour of the house like sinks, chairs, sofas, freezers, oven, and some not so much as toilet paper, pens, jewellery(like the pendant i was wearing). 
- This was not a controlled test, more so a open learning env for me to learn about YOLO so there was a lot of reasons for the model not trained to its best, and even the dataset was not processed much so a lot of misclassified images were there.

Conclusion: A lot of room to grow and learn advanced skills in this mini project like object tracking so that the detection remains consistent for each frame rather than it detecting an object in one frame and not the other.
YOLO architecture was cool, and I didnot go into much depth about it. It was more so, learning how to train any good model with the dataset you have about detecting an obejct.
Camera, model, occlusion, dataset played a main role in this training.  


Few Links which helped in training the model and understanding the output:<Link>[https://academy.ultralytics.com/courses/train-your-first-yolo/train-your-first-model] <Link>[https://docs.ultralytics.com/guides/yolo26-training-recipe]

Configurations in YOLO26 used for training the model: <LINK>[https://docs.ultralytics.com/usage/cfg]
