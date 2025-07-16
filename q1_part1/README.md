# CV Assignment-3

## Question 1 

Modify Faster R-CNN for predicting Oriented Bounding Boxes

Question 1:

Part 1: 
hyperparameter 1 : original 
<video controls src="hyper1/objectness1.mp4" title="Title"></video>
<video controls src="hyper1/anchor1.mp4" title="Title"></video>
<video controls src="hyper1/rpn_vs_roi1.mp4" title="Title"></video>
<video controls src="hyper1/box1.mp4" title="Title"></video>



Hyperparameter 2 :

rpn_nms_thresh: 0.5 (changed from default 0.7)
Keep all other parameters the same as the original

Lower rpn_nms_thresh (0.5): This creates a more aggressive NMS process that will remove more overlapping boxes. Only proposals with IoU less than 0.5 will be kept after NMS (compared to the more lenient 0.7).

This change will show clear differences in your visualizations:

The proposal evolution will show fewer redundant boxes around the same object
<video controls src="hyper2/anchor2.mp4" title="Title"></video>
<video controls src="hyper2/rpn_vs_roi2.mp4" title="Title"></video>
<video controls src="hyper2/objectness2.mp4" title="Title"></video>
<video controls src="hyper2/box2.mp4" title="Title"></video>


Hyperparameter 3 :

rpn_pre_nms_top_n_train=1000, (from 2000) 
rpn_post_nms_top_n_train=500  (from 1000)

Lower rpn_pre_nms_top_n_train (1000): This will select fewer proposals before applying Non-Maximum Suppression, effectively making the model more selective earlier in the pipeline.
Lower rpn_post_nms_top_n_train (500): This will further reduce the number of proposals that reach the ROI Head after NMS.
<video controls src="hyper3/objectness3.mp4" title="Title"></video>
<video controls src="hyper3/anchor3.mp4" title="Title"></video>
<video controls src="hyper3/rpn_vs_roi3.mp4" title="Title"></video>
<video controls src="hyper3/box3.mp4" title="Title"></video>
