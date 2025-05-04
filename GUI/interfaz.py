'''
@TODO quitar la visualizacion de la PCD y MESH, colocar un boton para abrirlas en una ventana
separada para mejorar el rendimiento y visualizacion del usuario.
'''

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from datetime import datetime
import open3d as o3d
import numpy as np
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import cv2
import sys
#global variables
G_pcd = None
G_mesh = None
G_bandera = False
G_nombre_archivo = None
# Función para generar un nombre único basado en la fecha y hora
def generar_nombre_archivo():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Función para abrir el explorador de archivos y cargar la imagen
def abrir_imagen():
    archivo = filedialog.askopenfilename(filetypes=[("Imagenes", "*.jpg;*.jpeg;*.png")])
    if archivo:
        # Guardar la imagen en la carpeta "Images"
        if not os.path.exists('Images'):
            os.makedirs('Images')
        if not os.path.exists('Depth_Images'):
            os.makedirs('Depth_Images')
        if not os.path.exists('Meshes'):
            os.makedirs('Meshes')
        if not os.path.exists('PCDs'):
            os.makedirs('PCDs')

        global G_nombre_archivo
        G_nombre_archivo = generar_nombre_archivo()
        image = Image.open(archivo)
        image.save(f"Images/{G_nombre_archivo}.png")

        # Mostrar la imagen en el panel correspondiente
        mostrar_imagen(image,tipo=1)

        '''############STARTS THE CODE OF RECONSTRUCTION##############'''
        model = GLPNForDepthEstimation.from_pretrained('vinvino02/glpn-nyu')
        feature_extractor = GLPNImageProcessor.from_pretrained('vinvino02/glpn-nyu')
        new_height = 480 if image.height > 480 else image.height
        new_height -= (new_height % 32)  # investigar que hace esta operacion
        new_width = int(new_height * image.width / image.height)
        diff = new_width % 32

        new_width = new_width - diff if diff < 16 else new_width + 32 - diff
        new_size = (new_width, new_height)
        image = image.resize(new_size)
        inputs = feature_extractor(images=image, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        #print(predicted_depth.shape)
        pad = 16
        output = predicted_depth.squeeze().cpu().numpy() * 1000.0
        output = output[pad:-pad, pad:-pad]
        image = image.crop((pad, pad, image.width - pad, image.height - pad))

        width, height = image.size
        depth_image = (output * 255 / np.max(output)).astype(np.uint8)

        depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
        image = np.array(image)
        mostrar_imagen(depth_image, tipo=2)

        depth_o3d = o3d.geometry.Image(depth_image)
        image_o3d = o3d.geometry.Image(image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                        convert_rgb_to_intensity=False)  # estaba en false

        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)
        pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

        pcl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.5)  # tenia 10 y 6
        pcd = pcd_raw.select_by_index(ind)
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction()
        # o3d.visualization.draw_geometries([pcd])
        rotation = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd.rotate(rotation, center=(0, 0, 0))

        #mostrar_imagen(pcd, tipo=3)
        global G_pcd
        G_pcd = pcd

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=16, n_threads=4)
        density_treshold = np.percentile(densities, 5)  # elimina el 5%
        vertices_keep = densities > density_treshold
        mesh.remove_vertices_by_mask(~vertices_keep)
        mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
        mesh.compute_vertex_normals(normalized=True)
        global G_mesh
        global G_bandera
        G_mesh = mesh
        G_bandera = True
        #mostrar_imagen(mesh, tipo=4)

# Función para mostrar la imagen cargada
def mostrar_imagen(imagen, tipo):
    if tipo==1:
        imagen = imagen.resize((200, 180))  # Redimensionar la imagen para el panel
        img_tk = ImageTk.PhotoImage(imagen)
        panel_imagen.config(image=img_tk)
        panel_imagen.image = img_tk
    elif tipo==2:
        #plt.imshow(imagen)
        #plt.show()
        img_color = cv2.applyColorMap(imagen, cv2.COLORMAP_JET)
        imagen = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        imagen.save(f"Depth_Images/{G_nombre_archivo}.png")
        imagen = imagen.resize((200, 180))  # Redimensionar la imagen para el panel
        #print(type(imagen))
        img_tk = ImageTk.PhotoImage(imagen)
        panel_imagen_profundidad.config(image=img_tk)
        panel_imagen_profundidad.image = img_tk




# Función para generar la nube de puntos (ejemplo básico)
def generar_nube_de_puntos():

    if G_bandera == True:
        # Mostrar la nube de puntos
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=600, height=480)
        vis.add_geometry(G_pcd)
        render_opt = vis.get_render_option()
        render_opt.point_size = 1
        render_opt.mesh_show_back_face = True
        # vis.update_geometry(render_opt)
        vis.run()
        vis.destroy_window()
        o3d.io.write_point_cloud(f'PCDs/{G_nombre_archivo}.ply',G_pcd)
        #o3d.visualization.draw_geometries([G_pcd], window_name="Nube de Puntos", width=600, height=480)



# Función para generar la malla (ejemplo básico)
def generar_mesh():
    if G_bandera==True:
        # Mostrar la malla
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=600,height=480)
        vis.add_geometry(G_mesh)
        render_opt = vis.get_render_option()
        render_opt.mesh_show_back_face=True
        vis.run()
        vis.destroy_window()
        o3d.io.write_triangle_mesh(f'Meshes/{G_nombre_archivo}.obj', G_mesh)
        #o3d.visualization.draw_geometries([G_mesh], window_name="Malla", width=600, height=400)


def salir():
    sys.exit(0)

# Configuración de la ventana principal
root = tk.Tk()
root.title("Reconstructor 3D JLHC")
ico = Image.open('icon.png')
photo = ImageTk.PhotoImage(ico)
root.wm_iconphoto(False, photo)
root.geometry("800x640")
root.config(bg="#f0f4f7")
#not resizable window
root.resizable(width=False, height=False)

# Crear un marco principal
frame = tk.Frame(root, bg="#f0f4f7")
frame.pack(pady=20)


titulo = tk.Label(frame, text="Reconstrucción 3D desde imagen 2D", font=("Arial", 18), bg="#f0f4f7", pady=10)
titulo.grid(row=0, column=0, columnspan=2)


# Panel para mostrar la imagen cargada
lbl_ImgLoaded = tk.Label(frame, text="Imagen de entrada 2D", font=("Arial", 14), bg="#f0f4f7", pady=10)
lbl_ImgLoaded.grid(row=1, column=0)

panel_imagen = tk.Label(frame, bg="#e0f7fa")  # Fondo azul claro para la imagen
panel_imagen.grid(row=2, column=0)

# Panel para mostrar la imagen de profundidad
lbl_DepthImg = tk.Label(frame, text="Imagen de profundidad", font=("Arial", 14), bg="#f0f4f7", pady=10)
lbl_DepthImg.grid(row=1, column=1)
panel_imagen_profundidad = tk.Label(frame, bg="#e0f7fa")  # Fondo amarillo para la imagen de profundidad
panel_imagen_profundidad.grid(row=2, column=1)

# Panel para mostrar la nube de puntos (utilizando Open3D)
lbl_PCD = tk.Button(frame, text="Nube de puntos", font=("Arial", 12, 'bold'), bg="#42f5ce", pady=5, padx=5, width=15, command=generar_nube_de_puntos)
lbl_PCD.grid(row=3, column=0)

# Panel para mostrar la malla (utilizando Open3D)
lbl_Mesh = tk.Button(frame, text="Malla", font=("Arial", 12, 'bold'), bg="#42f5ce", pady=5, padx=5, width=15,command=generar_mesh)
lbl_Mesh.grid(row=3, column=1)


# Botón para cargar la imagen
boton_abrir = tk.Button(frame, text="Abrir Imagen", command=abrir_imagen, font=("Arial", 12, 'bold'), bg="#4CAF50", fg="white",
                        padx=5, pady=5, width=15)
boton_abrir.grid(row=4, column=0)

btn_save = tk.Button(frame, text="Salir", command=salir, font=("Arial", 12, 'bold'), bg="#F5CF45", fg="white",
                        padx=5, pady=5, width=15)
btn_save.grid(row=4, column=1)


#labels info
info1 = tk.Label(frame, text="Proyecto final - Computer Vision", font=("Arial", 10), bg="#f0f4f7", pady=1)
info1.grid(row=5, column=0, columnspan=2)
info2 = tk.Label(frame, text="Universidad de Guanajuato - CIS", font=("Arial", 10), bg="#f0f4f7", pady=1)
info2.grid(row=6, column=0, columnspan=2)
info3 = tk.Label(frame, text="José Luis Hernández Camacho", font=("Arial", 8), bg="#f0f4f7", pady=1)
info3.grid(row=7, column=0, columnspan=2)


# Ejecutar la aplicación
root.mainloop()
