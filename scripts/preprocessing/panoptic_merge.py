import json
import sys


# Helper function to reduce overall size of the merged panoptic label file by removing redundant information
def minimize_annos(annos):
	vid_data_to_delete = ['video_id']
	img_data_to_delete = ['id', 'width', 'height', 'scene_id', 'depth_file_name']
	for video in annos['videos']:
		for img in video['images']:
			for data in img_data_to_delete:
				del img[data]
		for data in vid_data_to_delete:
			del video[data]

	vid_anno_data_to_delete = ['video_id']
	img_anno_data_to_delete = ['image_id', 'segments_info']
	for video_anno in annos['annotations']:
		video_inst = [inst for inst in annos['instances'] if inst['video_name']==video_anno['video_name']]
		# Since MVPd videos are guaranteed to keep consistent instance id's across sequnce, 
        # we are guaranteed that an instance id is unique to a specific object across the sequence
        # hence, merging frames instance id to category id is valid
		instance_id_map = {}
		for img_anno in video_anno['annotations']:
			for seg in img_anno['segments_info']:
				seg_inst = [inst for inst in video_inst if inst['id']==seg['instance_id']]
				assert len(seg_inst)==1
				seg_inst =  next(iter(seg_inst))
				assert seg_inst['category_id']==seg['category_id']

				seg_id = seg['id']
				seg_category_id = seg['category_id']
				seg_scene_name = seg_inst['scene_name']
				seg_color = seg_inst['color']

				if seg_id not in instance_id_map:
					instance_id_map[seg_id] = {
													'category_id': seg_category_id,
												  	'scene_name': seg_scene_name,
													'color': seg_color,
												}
				else:
					assert seg_category_id == instance_id_map[seg_id]['category_id']
					assert seg_scene_name == instance_id_map[seg_id]['scene_name']
					assert seg_color == instance_id_map[seg_id]['color']
					
			for data in img_anno_data_to_delete:
				del img_anno[data]

		for data in vid_anno_data_to_delete:
			del video_anno[data]

		video_anno['instance_id_map'] = instance_id_map
	
	return annos


if __name__=="__main__":
	assert len(sys.argv)==3 or len(sys.argv)==4
	src = sys.argv[1]
	dst = sys.argv[2]

	minimize = False
	if len(sys.argv)==4:
		assert sys.argv[3]=='--minimize', 'Usage: python panoptic_merge.py <src> <dst> [--minimize: minimize merged file]'
		minimize = True

	scene_annos = json.load(open(src, 'r'))
	if minimize:
		# To save disk-space, remove unused metadata
		scene_annos = minimize_annos(scene_annos)

	all_annos = json.load(open(dst, 'r'))
	all_annos['videos'] += scene_annos['videos']
	all_annos['annotations'] += scene_annos['annotations']
	all_annos['categories'] = scene_annos['categories']
	if not minimize:
		all_annos['instances'] += scene_annos['instances']

	# Update category information to account for instance-level annotations in HM3D dataset
	stuff_category_ids = [1, 2, 17]
	stuff_category_names = ['wall', 'floor', 'ceiling']
	for cat in all_annos['categories']:
		if cat['id'] in stuff_category_ids:
			assert cat['name']==stuff_category_names[stuff_category_ids.index(cat['id'])]
			cat['isthing'] = 0


	json.dump(all_annos, open(dst, 'w'))