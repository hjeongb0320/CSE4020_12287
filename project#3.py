from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

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

direction_x = 0.5000000000000001 * 10
direction_y = 0.7071067811865475 * 10
direction_z = 0.5 * 10
distance = 0

yaw = 45
token = 1
pitch = 45
zoom = 0.0
chk = 1

chk_drop = False
nodes = []
channel = []
tree = []
stack = 0
line_rendering = True
box_rendering = False
animate = False
now_frame = 0
frames = 0
frame_time = 0
motion_list = []

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

    vec3 material_color = color;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient1 = 0.5*light_color1;
    vec3 light_diffuse1 = light_color1;
    vec3 light_specular1 = light_color1;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular1 = light_color1;  // for non-metal material

    // ambient
    vec3 ambient1 = light_ambient1 * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir1 = normalize(light_pos1 - surface_pos);

    // diffuse
    float diff1 = max(dot(normal, light_dir1), 0);
    vec3 diffuse1 = diff1 * light_diffuse1 * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);

    vec3 reflect_dir1 = reflect(-light_dir1, normal);
    float spec1 = pow( max(dot(view_dir, reflect_dir1), 0.0), material_shininess);

    vec3 specular1 = spec1 * light_specular1 * material_specular1;

    vec3 color = ambient1 + diffuse1 + specular1;
    FragColor = vec4(color, 1.);
}
'''

class Node:
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

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

def prepare_vao_box():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
                         # position      normal
                         -1, 1, 1, 0, 0, 1,  # v0
                         1, -1, 1, 0, 0, 1,  # v2
                         1, 1, 1, 0, 0, 1,  # v1

                         -1, 1, 1, 0, 0, 1,  # v0
                         -1, -1, 1, 0, 0, 1,  # v3
                         1, -1, 1, 0, 0, 1,  # v2

                         -1, 1, -1, 0, 0, -1,  # v4
                         1, 1, -1, 0, 0, -1,  # v5
                         1, -1, -1, 0, 0, -1,  # v6

                         -1, 1, -1, 0, 0, -1,  # v4
                         1, -1, -1, 0, 0, -1,  # v6
                         -1, -1, -1, 0, 0, -1,  # v7

                         -1, 1, 1, 0, 1, 0,  # v0
                         1, 1, 1, 0, 1, 0,  # v1
                         1, 1, -1, 0, 1, 0,  # v5

                         -1, 1, 1, 0, 1, 0,  # v0
                         1, 1, -1, 0, 1, 0,  # v5
                         -1, 1, -1, 0, 1, 0,  # v4

                         -1, -1, 1, 0, -1, 0,  # v3
                         1, -1, -1, 0, -1, 0,  # v6
                         1, -1, 1, 0, -1, 0,  # v2

                         -1, -1, 1, 0, -1, 0,  # v3
                         -1, -1, -1, 0, -1, 0,  # v7
                         1, -1, -1, 0, -1, 0,  # v6

                         1, 1, 1, 1, 0, 0,  # v1
                         1, -1, 1, 1, 0, 0,  # v2
                         1, -1, -1, 1, 0, 0,  # v6

                         1, 1, 1, 1, 0, 0,  # v1
                         1, -1, -1, 1, 0, 0,  # v6
                         1, 1, -1, 1, 0, 0,  # v5

                         -1, 1, 1, -1, 0, 0,  # v0
                         -1, -1, -1, -1, 0, 0,  # v7
                         -1, -1, 1, -1, 0, 0,  # v3

                         -1, 1, 1, -1, 0, 0,  # v0
                         -1, 1, -1, -1, 0, 0,  # v4
                         -1, -1, -1, -1, 0, 0,  # v7
                         )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)  # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)  # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr,
                 GL_STATIC_DRAW)  # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32),
                          ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_line(now_node):
    # prepare vertex data (in main memory)
    # 6 vertices for 2 triangles

    sz = 1

    if now_node.parent == None:
        return None
    now_point = now_node.get_global_transform() * glm.vec4(0, 0, 0, 1) * sz
    parent_point = now_node.parent.get_global_transform() * glm.vec4(0, 0, 0, 1) * sz

    vertices = glm.array([now_point.xyz, parent_point.xyz])

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)  # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)  # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr,
                 GL_STATIC_DRAW)  # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

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
    global chk, line_rendering, box_rendering, animate
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    elif key == GLFW_KEY_V:
        if action == GLFW_RELEASE:
            chk = -chk
    elif key == GLFW_KEY_1:
        line_rendering = True
        box_rendering = False
    elif key == GLFW_KEY_2:
        line_rendering = False
        box_rendering = True
    elif key == GLFW_KEY_SPACE:
        animate = True

def drop_callback(window, path):
    global chk_drop, frames, frame_time

    chk_drop = True
    file = open(path[0])

    parsing(file)
    treeing()

    file_name = path[0].split("/")[-1]

    print("Bvh file name : ", file_name)
    print("Number of frames : ", frames)
    print("FPS : ", 1/frame_time)
    print("Number of joints : ", len(nodes))
    print("List of all joint names : ",)
    for i in nodes:
        print(i[0])

def parsing(bvh_file):
    global nodes, channel, stack, frames, frame_time, motion_list

    stack_list = []
    nodes = []
    channel = []
    motion_list = []
    stack = 0

    level = 0

    while True:
        # read line by line
        line = bvh_file.readline()

        bvh_info = line.split()

        # if read all file, stop
        if not line:
            break

        if bvh_info[0] == 'HIERARCHY':
            level = 1
            continue

        if bvh_info[0] == 'MOTION' :
            level = 2
            continue

        if level == 1 :
            if bvh_info[0] == 'ROOT' or bvh_info[0] == 'JOINT' or bvh_info[0] == 'End' :

                if bvh_info[0] == 'End':
                    channel.append(['block'])
                stack_list.append(len(nodes))

                temp = []
                temp.append(bvh_info[1])
                nodes.append(temp)
                continue

            elif bvh_info[0] == '{' :
                stack = stack + 1

            elif bvh_info[0] == '}' :
                now = stack_list.pop()

                if len(stack_list) == 0 :
                    parent = -1
                else :
                    parent = stack_list[len(stack_list) - 1]

                nodes[now].append(parent)
                stack = stack - 1

                if stack == 0 :
                    continue

            elif bvh_info[0] == 'OFFSET' :
                x = bvh_info[1]
                y = bvh_info[2]
                z = bvh_info[3]

                nodes[len(nodes) - 1].append(x)
                nodes[len(nodes) - 1].append(y)
                nodes[len(nodes) - 1].append(z)

            elif bvh_info[0] == 'CHANNELS' :
                channel.append(bvh_info[2:])

        if level == 2 :
            if bvh_info[0] == 'Frames:' :
                frames = int(bvh_info[1])
            elif bvh_info[0] == 'Frame' and bvh_info[1] == 'Time:' :
                frame_time = float(bvh_info[2])
            else :
                motion_list.append(bvh_info[0:])

def treeing():
    global nodes, channel, tree

    tree = []

    for i in nodes:
        node_name = i[0]

        x = float(i[1])
        y = float(i[2])
        z = float(i[3])
        parent_idx = i[4]
        if parent_idx == -1 :
            parent = None
        else :
            parent = tree[parent_idx]

        temp = Node(parent, glm.translate(glm.vec3(x,y,z)), glm.scale((1,1,1)), glm.vec3(1,1,1))

        tree.append(temp)

def draw_node_line(vao, node, VP, MVP_loc, color_loc):
    MVP = VP
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_LINES, 0, 2)

def draw_node_box(vao, node, VP, M_loc, MVP_loc, color_loc):

    sz = 1

    if node.parent == None:
        return None

    now_point = node.get_global_transform() * glm.vec4(0, 0, 0, 1) * sz
    parent_point = node.parent.get_global_transform() * glm.vec4(0, 0, 0, 1) * sz

    my_length = glm.distance(now_point.xyz,parent_point.xyz)

    vec1 = glm.vec3(0, 1, 0)
    vec2 = glm.normalize(parent_point.xyz-now_point.xyz)

    angle = glm.acos(glm.dot(vec1,vec2))
    axis = glm.normalize(glm.cross(vec1,vec2))

    rotate = glm.rotate(angle,axis)

    M = glm.translate((now_point.xyz + parent_point.xyz)/2) * rotate * glm.scale((0.02,my_length/2,0.02))
    N_MVP = VP * M
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(N_MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, 36)

def main():
    global cameraPos, cameraOrigin, cameraDirection, cameraRight, cameraUp, cameraPan, direction_x, direction_y, direction_z, distance, tree, nodes, now_frame

    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)  # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)  # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2019030991_Project#3', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    M_loc = glGetUniformLocation(shader_program, 'M')
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    color_loc = glGetUniformLocation(shader_program, 'color')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')

    # keycallback function
    glfwSetKeyCallback(window, key_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback_zoom)
    glfwSetDropCallback(window, drop_callback)

    # prepare vao
    vao_frame = prepare_vao_grid()

    time = 0
    t2 = 0

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix

        # use orthogonal projection
        if chk == -1:
            P = glm.ortho(-0.5,0.5,-0.5,0.5,-10,10)

        # use perspective projection
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
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        # draw grid on xz plane
        glUniform3f(color_loc,1,1,1)

        # draw grid on xz plane
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 10000)

        t = glfwGetTime()
        time = time + (t - t2)
        t2 = t

        # check is Drop
        if chk_drop == True:
            k = 0

            # draw nodes

            if animate == True:

                if time > frame_time:
                    time = 0
                    now_frame = (now_frame + 1) % int(frames)

                    motion_idx = 0
                    chan_idx = 0

                    for i in tree:
                        if i == tree[0]:
                            mat = glm.mat4()

                            for j in channel[chan_idx]:
                                x_r = 0
                                y_r = 0
                                z_r = 0
                                x_p = 0
                                y_p = 0
                                z_p = 0
                                j = j.upper()
                                if j == 'XPOSITION':
                                    x_p = float(motion_list[now_frame][motion_idx])
                                elif j == 'YPOSITION':
                                    y_p = float(motion_list[now_frame][motion_idx])
                                elif j == 'ZPOSITION':
                                    z_p = float(motion_list[now_frame][motion_idx])
                                elif j == 'XROTATION':
                                    x_r = glm.radians(float(motion_list[now_frame][motion_idx]))
                                elif j == 'YROTATION':
                                    y_r = glm.radians(float(motion_list[now_frame][motion_idx]))
                                elif j == 'ZROTATION':
                                    z_r = glm.radians(float(motion_list[now_frame][motion_idx]))
                                motion_idx = motion_idx + 1
                                mat = mat * glm.translate((x_p, y_p, z_p)) * glm.rotate(x_r, (1, 0, 0)) * glm.rotate(y_r, (0, 1, 0)) * glm.rotate(z_r, (0, 0, 1))

                            chan_idx = chan_idx + 1
                            i.set_joint_transform(mat)

                        elif channel[chan_idx][0] == 'block':
                            chan_idx = chan_idx + 1
                            continue
                        else:
                            mat = glm.mat4()
                            for j in channel[chan_idx]:
                                x_r = 0
                                y_r = 0
                                z_r = 0
                                x_p = 0
                                y_p = 0
                                z_p = 0
                                j = j.upper()
                                if j == 'XROTATION':
                                    x_r = glm.radians(float(motion_list[now_frame][motion_idx]))
                                elif j == 'YROTATION':
                                    y_r = glm.radians(float(motion_list[now_frame][motion_idx]))
                                elif j == 'ZROTATION':
                                    z_r = glm.radians(float(motion_list[now_frame][motion_idx]))
                                motion_idx = motion_idx + 1
                                mat = mat * glm.translate((x_p, y_p, z_p)) * glm.rotate(x_r, (1, 0, 0)) * glm.rotate(y_r, (0, 1, 0)) * glm.rotate(z_r, (0, 0, 1))

                            chan_idx = chan_idx + 1
                            i.set_joint_transform(mat)

            # recursively update global transformations of all nodes
            tree[0].update_tree_global_transform()

            if line_rendering == True:

                for i in tree:
                    vao_line = prepare_vao_line(i)
                    if vao_line == None:
                        k = k + 1
                        continue
                    draw_node_line(vao_line, i, P * V, MVP_loc, color_loc)
                    k = k + 1

            if box_rendering == True:

                    for i in tree:
                        vao_box = prepare_vao_box()
                        draw_node_box(vao_box, i, P * V, M_loc, MVP_loc, color_loc)
                        k = k + 1


        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
