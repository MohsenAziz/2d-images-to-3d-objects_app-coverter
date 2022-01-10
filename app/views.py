# Important imports
from app import app
from flask import request, render_template, url_for
import cv2
import numpy as np
from PIL import Image
import string
import random
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
from PIL import Image

import neural_renderer as nr
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BitMasks
from detectron2.utils.visualizer import Visualizer as PointRendVisualizer

from phosa.bodymocap import (
    get_bodymocap_predictor,
    process_mocap_predictions,
    visualize_orthographic,
)
from phosa.global_opt import optimize_human_object, visualize_human_object
from phosa.pointrend import get_pointrend_predictor
from phosa.pose_optimization import find_optimal_poses
from phosa.utils import bbox_xy_to_wh, center_vertices

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'



# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():

	# Execute if request is get
	if request.method == "GET":
		full_filename1 =  'images/white_bg.jpg'
		full_filename2 =  'images/white_bg.jpg'
		full_filename3 =  'images/white_bg.jpg'
		full_filename4 =  'images/white_bg.jpg'
		items=[full_filename1,full_filename2,full_filename3,full_filename4]

		return render_template("index.html",links=items)
    

	# Execute if reuqest is post
	if request.method == "POST":
		s=request.form['class_name']
		image_upload = request.files['image_upload']
		imagename = image_upload.filename

		# generating unique name to save image
		#letters = string.ascii_lowercase
		#name = ''.join(random.choice(letters) for i in range(10)) + '.png'

		name1='Instance_segmentation_image.png'
		name2='human_mesh_estimation.png'
		name3='final_mesh_front_view.png'
		name4='final_mesh_top_view.png'
		full_filename1=  'uploads/' + name1
		full_filename2 = 'uploads/' + name2
		full_filename3 = 'uploads/' + name3
		full_filename4 = 'uploads/' + name4

		#to do


		image = Image.open(image_upload).convert("RGB")
		w, h = image.size
		IMAGE_SIZE = 640
		r = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
		w = int(r * w)
		h = int(r * h)
		image = np.array(image.resize((w, h)))

        


		# instances segmentation

		segmenter = get_pointrend_predictor()
		instances = segmenter(image)["instances"]
		vis = PointRendVisualizer(image, metadata=MetadataCatalog.get("coco_2017_val"))
		instance_segmentation_image = Image.fromarray(vis.draw_instance_predictions(instances.to("cpu")).get_image())
		instance_segmentation_image=instance_segmentation_image.resize((640, 640))

		# human mesh estimation

		is_person = (instances.pred_classes == 0)
		bboxes_person = (instances[is_person].pred_boxes.tensor.cpu().numpy())
		masks_person = instances[is_person].pred_masks
		human_predictor = get_bodymocap_predictor()
		# Expects bgr image
		mocap_predictions = human_predictor.regress(
			img_original=image[..., ::-1],
			body_bbox_list=bbox_xy_to_wh(bboxes_person),
		)
		human_parameters = process_mocap_predictions(
			mocap_predictions=mocap_predictions,
			bboxes=bboxes_person,
			masks=masks_person,
			image_size=max(image.shape)
		)
		vis_image = visualize_orthographic(image, human_parameters)
		# If you don't see any mesh outputs, you may need to build NMR from source
		human_mesh_estimation = Image.fromarray(vis_image)
		human_mesh_estimation=human_mesh_estimation.resize((640, 640))
        
        
        #predict image classes
        
		OBJ_INFO={'bicycle':["models/meshes/bicycle_01.obj"],
		'motorcycle':['models/meshes/motorcycle_01.obj'],
		'tennis':['models/meshes/tennis_01.obj'],
		'laptop':['models/meshes/laptop_01.obj'],
		'skateboard':['models/meshes/skateboard_01.obj'],
		'surfboard':['models/meshes/surfboard_01.obj'],
		'bench':['models/meshes/bench_01.obj'],
		'bat':['models/meshes/bat_01.obj']      }
    

		# Object Pose Estimation

		verts, faces = nr.load_obj(OBJ_INFO[s][0])
		verts, faces = center_vertices(verts, faces)

		# Increase these parameters if fit looks bad.
		num_iterations = 2
		num_initializations = 2000

		# Reduce batch size if your GPU runs out of memory.
		batch_size = 500

		object_parameters = find_optimal_poses(
			instances, verts, faces, class_name=s, visualize=True, image=image,
			num_iterations=num_iterations, num_initializations=num_initializations,
			batch_size=batch_size,
		)




		# Joint Human and Object Optimization
		#increse np. of iterations to enhance model

		model = optimize_human_object(
			person_parameters=human_parameters,
			object_parameters=object_parameters,
			class_name=s,
			num_iterations=25,
			lr=2e-3,
		)

		# final_output
		rend, top = visualize_human_object(model, image)
		final_image_front = Image.fromarray(rend)
		final_image_front=final_image_front.resize((640, 640))
		final_image_top = Image.fromarray(top)
		final_image_top=final_image_top.resize((640, 640))







		instance_segmentation_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name1))
		human_mesh_estimation.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name2))
		final_image_front.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name3))
		final_image_top.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name4))

		# Returning template, filename, extracted text
        
		items=[full_filename1,full_filename2,full_filename3,full_filename4]
# 		first_name='final_mesh_front_view'
# 		second_name='final_mesh_top_view'
# 		pred=[first_name,second_name]
		return render_template('index.html', links=items,pred='Final_mesh_output')

# Main function
if __name__ == '__main__':
    app.run(debug=True)
