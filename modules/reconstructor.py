#%% librari import
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation, AutoModelForDepthEstimation, AutoImageProcessor
from transformers import DPTForDepthEstimation, DPTImageProcessor
#%% model
model = GLPNForDepthEstimation.from_pretrained('vinvino02/glpn-nyu')
feature_extractor = GLPNImageProcessor.from_pretrained('vinvino02/glpn-nyu')

#model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
#feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-large")


#model = AutoModelForDepthEstimation.from_pretrained("isl-org/ZoeDepth")
#processor = AutoImageProcessor.from_pretrained("isl-org/ZoeDepth")
#from leres import LeReS

#model = LeReS(pretrained=True)
#depth_map = model.estimate_depth(image)

#%% load image and resize
#image = Image.open('../images/Salon/Salon3.jpeg')
image = Image.open('../images/Cafeteria.jpeg')
plt.imshow(image)
new_height = 480 if image.height >480 else image.height
new_height -= (new_height % 32) # investigar que hace esta operacion
new_width = int(new_height * image.width / image.height)
diff = new_width % 32

new_width = new_width - diff if diff < 16 else  new_width + 32- diff
new_size = (new_width, new_height)
image = image.resize(new_size)

#%% prepare de image for model
inputs = feature_extractor(images = image, return_tensors = 'pt')

#%% prediction model
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
print(predicted_depth.shape)
#%% post processing
pad = 16
#import torch.nn.functional as F

#predicted_depth_resized = F.interpolate(predicted_depth.unsqueeze(0), size=(image.height, image.width), mode='bilinear', align_corners=False)


output = predicted_depth.squeeze().cpu().numpy() * 1000.0
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))
#visualize prediction
fix,ax = plt.subplots(1,2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(5)


#%% importing libraries
import numpy as np
import open3d as o3d

#%% depth for open3d
width, height = image.size
depth_image = (output * 255 / np.max(output)).astype(np.uint8)
import cv2
depth_image = cv2.GaussianBlur(depth_image, (5,5), 0)
image = np.array(image)


fix,ax = plt.subplots(1,2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(5)

#create rgbd image
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False) #estaba en false

#%% creating a camera

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

#%% creating o3d point cloud
def visualizer(pcd, point_size=1, w=1280, h=720, mesh_back_face = True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.mesh_show_back_face=mesh_back_face
    #vis.update_geometry(render_opt)
    vis.run()
    vis.destroy_window()

def visualizer_mesh(pcd, point_size=1, w=1280, h=720, mesh_back_face = True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    #render_opt.point_size = point_size
    render_opt.mesh_show_back_face=mesh_back_face
    #vis.update_geometry(render_opt)
    vis.run()
    vis.destroy_window()

pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
#o3d.visualization.draw_geometries([pcd_raw])
#visualizer(pcd_raw)

print("Number of points: {}".format(pcd_raw))
#%% post processing the 3d point cloud
#outliers removal

pcl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.5)#tenia 10 y 6
pcd = pcd_raw.select_by_index(ind)

#estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()
#o3d.visualization.draw_geometries([pcd])
rotation = pcd.get_rotation_matrix_from_xyz((np.pi, 0,0))
pcd.rotate(rotation, center=(0,0,0))

'''
plane_model, inliners = pcd.segment_plane(distance_threshold=0.000001, ransac_n=5, num_iterations=10000)
suelo = pcd.select_by_index(inliners)
resto = pcd.select_by_index(inliners, invert=True)
#colorear
suelo.paint_uniform_color([1,0,0])
resto.paint_uniform_color([0,1,0])

o3d.visualization.draw_geometries([suelo, resto])
alpha = 0.03 #ajustar segun la densidad de puntos
#mesh_suelo= o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(suelo,o3d.utility.DoubleVector([2.0]))

#o3d.visualization.draw_geometries([mesh_suelo])
'''

visualizer(pcd)

print("Number of points: {}".format(pcd))
#%% surface reconstruction


mesh,densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=16, n_threads=4)
#mesh = mesh.filter_smooth_taubin(number_of_iterations=5) #suavizado con taubin
#mesh = mesh.fill_holes()
#filtrar areas de baja densidad
density_treshold = np.percentile(densities, 5) #elimina el 5%
vertices_keep = densities > density_treshold
mesh.remove_vertices_by_mask(~vertices_keep)

#rotate the mesh
#rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0,0))
#mesh.rotate(rotation, center=(0,0,0))

#o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
mesh.compute_vertex_normals(normalized=True)
 #rellenar agujeros en la malla
visualizer_mesh(mesh)

#mesh_uniform = mesh.paint_uniform_color([0.9, 0.8, 0.9])
#mesh_uniform.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh_uniform], mesh_show_back_face=True)

#%% 3d mesh export
import datetime
ruta = datetime.datetime.now()
ruta = "MESH" + ruta.strftime("%Y%m%d%H%M%S")
print(ruta)
o3d.io.write_point_cloud("../PointClouds/"+ruta+'.ply', pcd)
o3d.io.write_triangle_mesh('../meshes/'+ruta+'.obj', mesh)

