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

direction_x = 0.5000000000000001
direction_y = 0.7071067811865475
direction_z = 0.5
distance = 0

yaw = 45
token = 1
pitch = 45
zoom = 0.0
chk = 1

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''


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

def prepare_vao_triangle():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.1, 0.0, 0.0,  1.0, 1.0, 1.0, # v0
         0.0, 0.1, 0.0,  1.0, 1.0, 1.0, # v1
         0.0, 0.0, 0.1,  1.0, 1.0, 1.0, # v2
    )

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

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
            # position        # color
        -0.1, 0.0, 0.3, 1.0, 1.0, 1.0,  #1
        -0.1, 0.0, -0.3, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.3, 1.0, 1.0, 1.0,  #2
        0.0, 0.0, -0.3, 1.0, 1.0, 1.0,
        0.1, 0.0, 0.3, 1.0, 1.0, 1.0,  #3
        0.1, 0.0, -0.3, 1.0, 1.0, 1.0,
        -0.2, 0.0, 0.3, 1.0, 1.0, 1.0,  #4
        -0.2, 0.0, -0.3, 1.0, 1.0, 1.0,
        0.2, 0.0, 0.3, 1.0, 1.0, 1.0,  #5
        0.2, 0.0, -0.3, 1.0, 1.0, 1.0,
        0.3, 0.0, 0.3, 1.0, 1.0, 1.0,  #6
        0.3, 0.0, -0.3, 1.0, 1.0, 1.0,
        -0.3, 0.0, 0.3, 1.0, 1.0, 1.0,  #7
        -0.3, 0.0, -0.3, 1.0, 1.0, 1.0,

        0.3, 0.0, 0.0, 1.0, 1.0, 1.0,  #1
        -0.3, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.3, 0.0, -0.1, 1.0, 1.0, 1.0,  #2
        -0.3, 0.0, -0.1, 1.0, 1.0, 1.0,
        0.3, 0.0, 0.1, 1.0, 1.0, 1.0,  #3
        -0.3, 0.0, 0.1, 1.0, 1.0, 1.0,
        0.3, 0.0, 0.2, 1.0, 1.0, 1.0,  #4
        -0.3, 0.0, 0.2, 1.0, 1.0, 1.0,
        0.3, 0.0, -0.2, 1.0, 1.0, 1.0,  #5
        -0.3, 0.0, -0.2, 1.0, 1.0, 1.0,
        0.3, 0.0, 0.3, 1.0, 1.0, 1.0,  #6
        -0.3, 0.0, 0.3, 1.0, 1.0, 1.0,
        0.3, 0.0, -0.3, 1.0, 1.0, 1.0,  #7
        -0.3, 0.0, -0.3, 1.0, 1.0, 1.0,
                         )

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
    global chk
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    elif key == GLFW_KEY_V:
        if action == GLFW_RELEASE:
            chk = -chk

def main():
    global cameraPos, cameraOrigin, cameraDirection, cameraRight, cameraUp, cameraPan, direction_x, direction_y, direction_z, distance

    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)  # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)  # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2019030991_Project#1', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')

    # keycallback function
    glfwSetKeyCallback(window, key_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback_zoom)

    # prepare vao
    # vao_triangle = prepare_vao_triangle()
    vao_frame = prepare_vao_grid()

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

        # draw grid on xz plane
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 28)

        # draw triangle
        # glBindVertexArray(vao_triangle)
        # glDrawArrays(GL_TRIANGLES, 0, 3)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
