from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

global cameraUp
global cameraRight
global cameraPan
global cameraPos
global cameraOrigin
global cameraDirection
global world_up

cameraUp = glm.vec3(0,1,0)
cameraRight = glm.vec3(0,0,0)
cameraDirection = glm.vec3(0,0,0)
cameraPan = glm.vec3(0,0,0)
cameraPos = glm.vec3(0,0,0)
cameraOrigin = glm.vec3(0,0,0)
world_up = glm.vec3(0,1,0)

lastX = 0
lastY = 0

direction_x = 0.5000000000000001 * 50
direction_y = 0.7071067811865475 * 50
direction_z = 0.5 * 50
distance = 0

yaw = 45
token = 1
pitch = 45
zoom = 0.0
chk = 1

vertex_pos = []
vertex_norm = []
face_info = []
chk_drop = False
chk_animaiting = False

face_total = 0
tri_ver = 0
quad_ver = 0
multi_ver = 0
triangle = []
temp_triangle = 0
wire = 1

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(transpose(inverse(M))) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 color;

void main()
{
    // light and material properties
    vec3 light_pos1 = vec3(3,2,4);
    vec3 light_color1 = vec3(1,1,1);
    
    vec3 light_pos2 = vec3(-3,2,-1);
    vec3 light_color2 = vec3(1,1,1);
    
    vec3 material_color = color;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient1 = 0.1*light_color1;
    vec3 light_diffuse1 = light_color1;
    vec3 light_specular1 = light_color1;
    
    vec3 light_ambient2 = 0.1*light_color2;
    vec3 light_diffuse2 = light_color2;
    vec3 light_specular2 = light_color2;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular1 = light_color1;  // for non-metal material
    vec3 material_specular2 = light_color2;  // for non-metal material

    // ambient
    vec3 ambient1 = light_ambient1 * material_ambient;
    vec3 ambient2 = light_ambient2 * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir1 = normalize(light_pos1 - surface_pos);
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    // diffuse
    float diff1 = max(dot(normal, light_dir1), 0);
    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse1 = diff1 * light_diffuse1 * material_diffuse;
    vec3 diffuse2 = diff2 * light_diffuse2 * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    
    vec3 reflect_dir1 = reflect(-light_dir1, normal);
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec1 = pow( max(dot(view_dir, reflect_dir1), 0.0), material_shininess);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    
    vec3 specular1 = spec1 * light_specular1 * material_specular1;
    vec3 specular2 = spec2 * light_specular2 * material_specular2;

    vec3 color = ambient1 + ambient2 + diffuse1 + diffuse2 + specular1 + specular2;
    FragColor = vec4(color, 1.);
}
'''

class Node:
    def __init__(self, parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color


def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------

    # vertex shader
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)  # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source)  # provide shader source codeㅅ
    glCompileShader(vertex_shader)  # compile the shader object

    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())

    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)  # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source)  # provide shader source code
    glCompileShader(fragment_shader)  # compile the shader object

    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()  # create an empty program object
    glAttachShader(shader_program, vertex_shader)  # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)  # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program  # return the shader program

def prepare_vao_pikachu():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "obj_files", "Pikachu.obj")

    with open(file_path, "r") as file:
        parsing(file)

    triangle.append(temp_triangle)

    obj_vao_vertices = []
    # prepare vertex data (in main memory)
    for i in face_info:
        for j in i:
            obj_vao_vertices.append(vertex_pos[j[0]])
            obj_vao_vertices.append(vertex_norm[j[2]])
    obj_vao_vertices = np.array(obj_vao_vertices, dtype=np.float32)
    obj_vao_vertices = glm.array(obj_vao_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, obj_vao_vertices.nbytes, obj_vao_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_umbreon():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "obj_files", "Umbreon.obj")

    with open(file_path, "r") as file:
        parsing(file)

    triangle.append(temp_triangle)

    obj_vao_vertices = []
    # prepare vertex data (in main memory)
    for i in face_info:
        for j in i:
            obj_vao_vertices.append(vertex_pos[j[0]])
            obj_vao_vertices.append(vertex_norm[j[2]])
    obj_vao_vertices = np.array(obj_vao_vertices, dtype=np.float32)
    obj_vao_vertices = glm.array(obj_vao_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, obj_vao_vertices.nbytes, obj_vao_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_tree():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "obj_files", "Tree.obj")

    with open(file_path, "r") as file:
        parsing(file)

    triangle.append(temp_triangle)

    obj_vao_vertices = []
    # prepare vertex data (in main memory)
    for i in face_info:
        for j in i:
            obj_vao_vertices.append(vertex_pos[j[0]])
            obj_vao_vertices.append(vertex_norm[j[2]])
    obj_vao_vertices = np.array(obj_vao_vertices, dtype=np.float32)
    obj_vao_vertices = glm.array(obj_vao_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, obj_vao_vertices.nbytes, obj_vao_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_table():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "obj_files", "Picnic_table.obj")

    with open(file_path, "r") as file:
        parsing(file)

    triangle.append(temp_triangle)

    obj_vao_vertices = []
    # prepare vertex data (in main memory)
    for i in face_info:
        for j in i:
            obj_vao_vertices.append(vertex_pos[j[0]])
            obj_vao_vertices.append(vertex_norm[j[2]])
    obj_vao_vertices = np.array(obj_vao_vertices, dtype=np.float32)
    obj_vao_vertices = glm.array(obj_vao_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, obj_vao_vertices.nbytes, obj_vao_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_maple():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "obj_files", "Maple_Leaf.obj")

    with open(file_path, "r") as file:
        parsing(file)

    triangle.append(temp_triangle)

    obj_vao_vertices = []
    # prepare vertex data (in main memory)
    for i in face_info:
        for j in i:
            obj_vao_vertices.append(vertex_pos[j[0]])
            obj_vao_vertices.append(vertex_norm[j[2]])
    obj_vao_vertices = np.array(obj_vao_vertices, dtype=np.float32)
    obj_vao_vertices = glm.array(obj_vao_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, obj_vao_vertices.nbytes, obj_vao_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_fist():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "obj_files", "Fist.obj")

    with open(file_path, "r") as file:
        parsing(file)

    triangle.append(temp_triangle)

    obj_vao_vertices = []
    # prepare vertex data (in main memory)
    for i in face_info:
        for j in i:
            obj_vao_vertices.append(vertex_pos[j[0]])
            obj_vao_vertices.append(vertex_norm[j[2]])
    obj_vao_vertices = np.array(obj_vao_vertices, dtype=np.float32)
    obj_vao_vertices = glm.array(obj_vao_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, obj_vao_vertices.nbytes, obj_vao_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_foot():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "obj_files", "Foot.obj")

    with open(file_path, "r") as file:
        parsing(file)

    triangle.append(temp_triangle)

    obj_vao_vertices = []
    # prepare vertex data (in main memory)
    for i in face_info:
        for j in i:
            obj_vao_vertices.append(vertex_pos[j[0]])
            obj_vao_vertices.append(vertex_norm[j[2]])
    obj_vao_vertices = np.array(obj_vao_vertices, dtype=np.float32)
    obj_vao_vertices = glm.array(obj_vao_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, obj_vao_vertices.nbytes, obj_vao_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_obj():
    obj_vao_vertices = []
    # prepare vertex data (in main memory)
    for i in face_info:
        for j in i:
            obj_vao_vertices.append(vertex_pos[j[0]])
            obj_vao_vertices.append(vertex_norm[j[2]])
    obj_vao_vertices = np.array(obj_vao_vertices, dtype=np.float32)
    obj_vao_vertices = glm.array(obj_vao_vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, obj_vao_vertices.nbytes, obj_vao_vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    vertices = []

    list1 = [0.0,50.0,1.0,1.0,1.0]
    list2 = [0.0,-50.0,1.0,1.0,1.0]
    list3 = [50.0,0.0]
    list4 = [1.0,1.0,1.0,-50.0,0.0]
    list5 = [1.0,1.0,1.0]

    for i in range(-50,50):
        vertices.append(i)
        for j in list1:
            vertices.append(j)

        vertices.append(i)
        for j in list2:
            vertices.append(j)

    for i in range(-50,50):
        for j in list3:
            vertices.append(j)

        vertices.append(i)
        for j in list4:
            vertices.append(j)

        vertices.append(i)
        for j in list5:
            vertices.append(j)

    vertices = np.array(vertices, dtype=np.float32)
    vertices = glm.array(vertices)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def cursor_callback_orbit(window, xpos, ypos):
    global lastX, lastY, yaw, pitch, direction_x, direction_y, direction_z, world_up, distance, token

    # store offset
    xoffset = xpos - lastX
    yoffset = lastY - ypos

    # store last pos to calculate offset
    lastX = xpos
    lastY = ypos

    # control speed of orbit
    sensitivity = 0.3
    xoffset = xoffset * sensitivity
    yoffset = yoffset * sensitivity

    # control azimuth and elevation
    pitch = pitch + yoffset

    # exception handling
    if np.cos(np.radians(pitch)) > 0:
        world_up = glm.vec3(0,1,0)
        yaw = yaw + xoffset
    else:
        world_up = glm.vec3(0,-1,0)
        yaw = yaw - xoffset

    # calculate camera pos
    direction_x = distance * np.cos(np.radians(pitch)) * np.cos(np.radians(yaw))
    direction_y = distance * np.sin(np.radians(pitch))
    direction_z = distance * np.cos(np.radians(pitch)) * np.sin(np.radians(yaw))

def cursor_callback_pan(window, xpos, ypos):
    global lastX, lastY, cameraRight, cameraUp, cameraPan

    # store offset
    xoffset = -(xpos - lastX)
    yoffset = -(lastY - ypos)

    # store last pos to calculate offset
    lastX = xpos
    lastY = ypos

    # control speed of orbit
    sensitivity = 0.01
    xoffset = xoffset * sensitivity
    yoffset = yoffset * sensitivity

    # calculate camerapan
    cameraPan = cameraPan + (cameraRight * xoffset) + (cameraUp * yoffset)

def cursor_callback_wait(window, xpos, ypos):
    return None

def scroll_callback_zoom(window, xoffset, yoffset):
    global zoom, direction_x, direction_y, direction_z, distance, cameraOrigin, cameraPan, cameraPos, cameraDirection

    zoom = -yoffset * 0.01

    # store past pos to control max zoom-in
    past_x = direction_x
    past_y = direction_y
    past_z = direction_z

    # calculate zoom
    direction_x = direction_x + zoom * cameraDirection.x
    direction_y = direction_y + zoom * cameraDirection.y
    direction_z = direction_z + zoom * cameraDirection.z

    # control max zoom-in
    cameraPos = glm.vec3(direction_x,direction_y,direction_z)
    distance = glm.distance(cameraPos + cameraPan, cameraOrigin + cameraPan)

    if distance < 0.2:
        direction_x = past_x
        direction_y = past_y
        direction_z = past_z

def button_callback(window, button, action, mod):
    global lastX,lastY
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            lastX,lastY = glfwGetCursorPos(window)
            glfwSetCursorPosCallback(window, cursor_callback_orbit)
        elif action == GLFW_RELEASE:
            glfwSetCursorPosCallback(window, cursor_callback_wait)
    elif button == GLFW_MOUSE_BUTTON_RIGHT:
        if action==GLFW_PRESS:
            lastX, lastY = glfwGetCursorPos(window)
            glfwSetCursorPosCallback(window, cursor_callback_pan)
        elif action == GLFW_RELEASE:
            glfwSetCursorPosCallback(window, cursor_callback_wait)

def key_callback(window, key, scancode, action, mods):
    global chk,chk_drop,chk_animaiting,wire
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    elif key == GLFW_KEY_V:
        if action == GLFW_RELEASE:
            chk = -chk
    elif key == GLFW_KEY_H and action == GLFW_PRESS:
        chk_drop = False
        chk_animaiting = True
    elif key == GLFW_KEY_Z:
        if action == GLFW_RELEASE:
            wire = -wire

def parsing(obj_file):
    global vertex_pos, vertex_norm, face_info, chk_drop, face_total, tri_ver, quad_ver, multi_ver, temp_triangle

    # information to print on terminal
    face_total = 0
    tri_ver = 0
    quad_ver = 0
    multi_ver = 0

    # numbers of triangle to draw
    temp_triangle = 0

    # initialize list of OBJ files
    vertex_pos = []
    vertex_norm = []
    face_info = []

    # parsing
    while True:
        # read line by line
        line = obj_file.readline()

        # if read all file, stop
        if not line:
            break

        # parsing line
        obj_info = line.split()

        # in obj file, if white space line exist, exception handling
        A = len(obj_info)
        if A == 0:
            continue

        # read information
        if obj_info[0] == 'v':
            x = float(obj_info[1])
            y = float(obj_info[2])
            z = float(obj_info[3])
            vertex_pos.append([x,y,z])

        elif obj_info[0] == 'vn':
            x = float(obj_info[1])
            y = float(obj_info[2])
            z = float(obj_info[3])
            vertex_norm.append([x,y,z])

        elif obj_info[0] == 'f':
            face_total = face_total + 1

            # write information about face to write on terminal
            face_len = len(obj_info) - 1
            if face_len == 3:
                tri_ver = tri_ver + 1
            elif face_len == 4:
                quad_ver = quad_ver + 1
            else:
                multi_ver = multi_ver + 1

            temp = obj_info[1].split('/')

            # save the first vertex
            v = int(temp[0]) - 1
            # vt = int(temp[1]) - 1
            vn = int(temp[2]) - 1
            vertex_zero = [v,0,vn]

            vertex_temp = []

            # save the other vertices
            for i in obj_info[2:]:
                temp = i.split('/')

                v = int(temp[0]) - 1
                # vt = int(temp[1]) - 1
                vn = int(temp[2]) - 1
                vertex_temp.append([v,0,vn])

            length = len(vertex_temp)

            # mapping vertices. [0] + [1,2], [2,3], [3,4] ...
            for i in range(0,length-1):
                vertex_return = []
                vertex_return.append(vertex_zero)
                vertex_return.append(vertex_temp[i])
                vertex_return.append(vertex_temp[i+1])
                face_info.append(vertex_return)
                temp_triangle = temp_triangle + 1

def drop_callback(window, path):
    global vertex_pos, vertex_norm, face_info, chk_drop, chk_animaiting

    chk_animaiting = False
    chk_drop = True
    file = open(path[0])

    parsing(file)

    file_name = path[0].split("/")[-1]

    print(path)
    print("Obj file name : ", file_name)
    print("Total faces : ", face_total)
    print("Total 3 vertices faces : ", tri_ver)
    print("Total 4 vertices faces : ", quad_ver)
    print("Total more than 4 vertices faces : ", multi_ver)

def draw_node(vao, node, VP, MVP_loc, color_loc):
    global temp_triangle
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, temp_triangle * 3)

def main():
    global cameraPos, cameraOrigin, cameraDirection, cameraRight, cameraUp, cameraPan, direction_x, direction_y, direction_z, distance, temp_triangle

    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)  # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)  # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2019030991_Project#2', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    color_loc = glGetUniformLocation(shader_program, 'color')

    # keycallback function
    glfwSetKeyCallback(window, key_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback_zoom)
    glfwSetDropCallback(window, drop_callback)

    # prepare node
    tree = Node(None, glm.scale((0.5,0.5,0.5)), glm.vec3(0,1,0))

    table = Node(tree, glm.scale((5.0,5.0,5.0)), glm.vec3(1,1,1))
    maple = Node(tree, glm.mat4(), glm.vec3(0.8,0.3,0.2))

    pikachu = Node(table, glm.mat4(), glm.vec3(1,1,0))
    umbreon = Node(table, glm.mat4(), glm.vec3(0.2,0.2,0.2))
    fist = Node(maple, glm.mat4(), glm.vec3(0.6,0.4,0.3))
    foot = Node(maple, glm.scale((0.3,0.3,0.3)), glm.vec3(0.3,0.5,0.8))

    # prepare vao
    vao_frame = prepare_vao_grid()

    # prepare node vao
    vao_tree = prepare_vao_tree()

    vao_table = prepare_vao_table()
    vao_maple = prepare_vao_maple()

    vao_fist = prepare_vao_fist()
    vao_pikachu = prepare_vao_pikachu()
    vao_umbreon = prepare_vao_umbreon()
    vao_foot = prepare_vao_foot()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # wireframe mode
        if wire == -1:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # solid mode (default)
        if wire == 1:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # projection matrix

        # use orthogonal projection
        if chk == -1:
            P = glm.ortho(-0.5,0.5,-0.5,0.5,-10,10)

        # use perspective projection (default)
        if chk == 1:
            P = glm.perspective(45, 800.0 / 800.0, 0.1, 100.0)


        # camera position / camera vector
        cameraOrigin = glm.vec3(0, 0, 0)
        cameraPos = glm.vec3(direction_x,direction_y,direction_z)
        cameraRight = glm.normalize(glm.cross(world_up, cameraPos))
        cameraUp = glm.cross(cameraPos,cameraRight)
        cameraDirection = glm.normalize(cameraPos - cameraOrigin)

        # camera distance
        distance = glm.distance(cameraPos + cameraPan, cameraOrigin + cameraPan)

        # view matrix
        V = glm.lookAt(cameraPos + cameraPan, cameraOrigin + cameraPan, cameraUp)

        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P * V * I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        M = glm.translate(glm.vec3(0,0,0))

        # current frame: P*V*M
        MVP = P * V * M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw grid on xz plane
        glUniform3f(color_loc,1.0,1.0,1.0)

        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 10000)

        # check is Drop
        if chk_drop == True:
            vao_obj = prepare_vao_obj()

            # choose color
            glUniform3f(color_loc, 0.0, 1.0, 0.0)

            # draw dropped obj file
            glBindVertexArray(vao_obj)
            glDrawArrays(GL_TRIANGLES, 0, temp_triangle * 3)

        if chk_animaiting == True:

            t = glfwGetTime()

            # Grandparent node - Tree. Turn around
            tree.set_transform(glm.rotate(t, (0,1,0)))

            # Parent node - Table and Maple. Both go up and down.
            table.set_transform(glm.translate((15.0,glm.sin(t)*5,15.0)))
            maple.set_transform(glm.translate((0,20 + glm.sin(t),0)))

            # Child node - pikachu and umbreon is child of Table. Both turn around the table.
            # Child node - fist and foot is child of Maple. Both turn around the maple. Foot go up and down too.
            pikachu.set_transform(glm.rotate(t,(0,1,0)) * glm.translate((-17.0,0,0)))
            umbreon.set_transform(glm.rotate(t, (0, 1, 0)) * glm.translate((-15.0, 0, 0)))
            fist.set_transform(glm.rotate(t, (0, 1, 0)) * glm.translate((-15.0, 0, 0)))
            foot.set_transform(glm.rotate(t, (0, 1, 0)) * glm.translate((15.0, 0 + 5*glm.sin(t), 0)))

            # update information about transform
            tree.update_tree_global_transform()

            # draw every node
            temp_triangle = triangle[0]
            draw_node(vao_tree, tree, P * V, MVP_loc, color_loc)

            temp_triangle = triangle[1]
            draw_node(vao_table, table, P * V, MVP_loc, color_loc)
            temp_triangle = triangle[2]
            draw_node(vao_maple, maple, P * V, MVP_loc, color_loc)

            temp_triangle = triangle[3]
            draw_node(vao_fist, fist, P * V, MVP_loc, color_loc)
            temp_triangle = triangle[4]
            draw_node(vao_pikachu, pikachu, P * V, MVP_loc, color_loc)
            temp_triangle = triangle[5]
            draw_node(vao_umbreon, umbreon, P * V, MVP_loc, color_loc)
            temp_triangle = triangle[6]
            draw_node(vao_foot, foot, P * V, MVP_loc, color_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()