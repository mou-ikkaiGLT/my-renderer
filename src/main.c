#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <math.h>

/* ── Matrix math ─────────────────────────────────────────────────────────────
   OpenGL uses column-major 4x4 matrices.
   Index layout: m[col * 4 + row]
────────────────────────────────────────────────────────────────────────────── */

typedef float mat4[16];

static void mat4_identity(mat4 m)
{
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

/* Multiply two matrices: out = a * b */
static void mat4_multiply(mat4 out, const mat4 a, const mat4 b)
{
    mat4 tmp;
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            tmp[col*4 + row] = 0.0f;
            for (int k = 0; k < 4; k++)
                tmp[col*4 + row] += a[k*4 + row] * b[col*4 + k];
        }
    for (int i = 0; i < 16; i++) out[i] = tmp[i];
}

/* Rotate around the Y axis — controls left/right (yaw) */
static void mat4_rotate_y(mat4 m, float angle)
{
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0]  =  c;
    m[2]  = -s;
    m[8]  =  s;
    m[10] =  c;
}

/* Rotate around the X axis — controls up/down (pitch) */
static void mat4_rotate_x(mat4 m, float angle)
{
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[5]  =  c;
    m[6]  =  s;
    m[9]  = -s;
    m[10] =  c;
}

/* Rotate around the Z axis — controls roll (Q/E) */
static void mat4_rotate_z(mat4 m, float angle)
{
    mat4_identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0]  =  c;
    m[1]  =  s;
    m[4]  = -s;
    m[5]  =  c;
}

/* Translate (move) by x, y, z */
static void mat4_translate(mat4 m, float x, float y, float z)
{
    mat4_identity(m);
    m[12] = x;
    m[13] = y;
    m[14] = z;
}

/* Perspective projection matrix */
static void mat4_perspective(mat4 m, float fov, float aspect, float near_z, float far_z)
{
    float f = 1.0f / tanf(fov * 0.5f);
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (far_z + near_z) / (near_z - far_z);
    m[11] = -1.0f;
    m[14] = (2.0f * far_z * near_z) / (near_z - far_z);
}

/* ── Input state ─────────────────────────────────────────────────────────────
   Stored as globals so GLFW callbacks can write to them.
────────────────────────────────────────────────────────────────────────────── */

static float yaw   = 0.0f;   /* left/right rotation in radians */
static float pitch = 0.0f;   /* up/down rotation in radians    */
static float roll  = 0.0f;   /* clockwise/counter rotation in radians */

static int    mouse_dragging = 0;
static double last_mouse_x   = 0.0;
static double last_mouse_y   = 0.0;

#define MOUSE_SENSITIVITY  0.005f   /* radians per pixel dragged  */
#define KEY_ROTATION_SPEED 1.8f     /* radians per second for WASD */

/* ── GLFW callbacks ──────────────────────────────────────────────────────────*/

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

/* Track when the left mouse button is pressed or released */
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mouse_dragging = 1;
            glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
        } else if (action == GLFW_RELEASE) {
            mouse_dragging = 0;
        }
    }
}

/* While dragging, accumulate rotation from mouse movement */
void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
    (void)window;
    if (!mouse_dragging) return;

    float dx = (float)(xpos - last_mouse_x);
    float dy = (float)(ypos - last_mouse_y);
    last_mouse_x = xpos;
    last_mouse_y = ypos;

    yaw   += dx * MOUSE_SENSITIVITY;
    pitch += dy * MOUSE_SENSITIVITY;
}

/* ── Shader helper ───────────────────────────────────────────────────────────*/

const char *vertexShaderSource =
    "#version 410 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 aColor;\n"
    "layout (location = 2) in vec3 aNormal;\n"
    "out vec3 fragColor;\n"
    "out vec3 fragNormal;\n"
    "out vec3 fragPos;\n"
    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "void main() {\n"
    "    vec4 worldPos  = model * vec4(aPos, 1.0);\n"
    "    gl_Position    = projection * view * worldPos;\n"
    "    fragPos        = vec3(worldPos);\n"             /* world-space position for specular */
    "    fragNormal     = mat3(model) * aNormal;\n"
    "    fragColor      = aColor;\n"
    "}\0";

const char *fragmentShaderSource =
    "#version 410 core\n"
    "in vec3 fragColor;\n"
    "in vec3 fragNormal;\n"
    "in vec3 fragPos;\n"
    "out vec4 FragColor;\n"
    "uniform vec3 cameraPos;\n"
    "uniform int  lightingEnabled;\n"
    "void main() {\n"
    "    if (lightingEnabled == 0) {\n"
    "        FragColor = vec4(fragColor, 1.0);\n"        /* flat, no lighting */
    "        return;\n"
    "    }\n"
    "    vec3  lightDir    = normalize(vec3(1.0, 2.0, 1.5));\n"
    "    vec3  norm        = normalize(fragNormal);\n"
    "    vec3  viewDir     = normalize(cameraPos - fragPos);\n"
    "    vec3  reflectDir  = reflect(-lightDir, norm);\n"
    "    float ambient     = 0.15;\n"
    "    float diff        = max(dot(norm, lightDir), 0.0);\n"
    "    float spec        = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);\n"
    "    vec3  result      = (ambient + diff) * fragColor\n"
    "                      + 0.5 * spec * vec3(1.0);\n"  /* white specular highlight */
    "    FragColor = vec4(result, 1.0);\n"
    "}\0";

unsigned int compile_shader(unsigned int type, const char *source)
{
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char log[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
    }

    return shader;
}

int main(void)
{
    /* ── Init GLFW ─────────────────────────────────────────────────────────── */
    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "Pyramid", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to init GLAD\n");
        return -1;
    }

    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("Controls: WASD to rotate | click and drag to rotate\n");

    glEnable(GL_DEPTH_TEST);

    /* ── Pyramid vertex data ───────────────────────────────────────────────────
       Each vertex: x, y, z,  r, g, b,  nx, ny, nz  (9 floats)
       Normals are computed as the cross product of two face edges,
       then normalized. Each face shares one normal across all 3 vertices.

       Front  normal: ( 0.0000,  0.4472,  0.8944)
       Right  normal: ( 0.8944,  0.4472,  0.0000)
       Back   normal: ( 0.0000,  0.4472, -0.8944)
       Left   normal: (-0.8944,  0.4472,  0.0000)
       Base   normal: ( 0.0000, -1.0000,  0.0000)
    ────────────────────────────────────────────────────────────────────────── */
    float vertices[] = {
        /* Front face — orange */
         0.0f,  0.7f,  0.0f,   1.0f, 0.5f, 0.0f,   0.0f,    0.4472f,  0.8944f,
        -0.5f, -0.3f,  0.5f,   1.0f, 0.5f, 0.0f,   0.0f,    0.4472f,  0.8944f,
         0.5f, -0.3f,  0.5f,   1.0f, 0.5f, 0.0f,   0.0f,    0.4472f,  0.8944f,

        /* Right face — red */
         0.0f,  0.7f,  0.0f,   0.8f, 0.1f, 0.1f,   0.8944f, 0.4472f,  0.0f,
         0.5f, -0.3f,  0.5f,   0.8f, 0.1f, 0.1f,   0.8944f, 0.4472f,  0.0f,
         0.5f, -0.3f, -0.5f,   0.8f, 0.1f, 0.1f,   0.8944f, 0.4472f,  0.0f,

        /* Back face — green */
         0.0f,  0.7f,  0.0f,   0.1f, 0.7f, 0.2f,   0.0f,    0.4472f, -0.8944f,
         0.5f, -0.3f, -0.5f,   0.1f, 0.7f, 0.2f,   0.0f,    0.4472f, -0.8944f,
        -0.5f, -0.3f, -0.5f,   0.1f, 0.7f, 0.2f,   0.0f,    0.4472f, -0.8944f,

        /* Left face — blue */
         0.0f,  0.7f,  0.0f,   0.2f, 0.3f, 0.9f,  -0.8944f, 0.4472f,  0.0f,
        -0.5f, -0.3f, -0.5f,   0.2f, 0.3f, 0.9f,  -0.8944f, 0.4472f,  0.0f,
        -0.5f, -0.3f,  0.5f,   0.2f, 0.3f, 0.9f,  -0.8944f, 0.4472f,  0.0f,

        /* Base triangle 1 — dark grey */
        -0.5f, -0.3f,  0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
         0.5f, -0.3f,  0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
         0.5f, -0.3f, -0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,

        /* Base triangle 2 — dark grey */
        -0.5f, -0.3f,  0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
         0.5f, -0.3f, -0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
        -0.5f, -0.3f, -0.5f,   0.3f, 0.3f, 0.3f,   0.0f,   -1.0f,    0.0f,
    };

    /* ── VAO / VBO ─────────────────────────────────────────────────────────── */
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    /* stride is now 9 floats: position(3) + color(3) + normal(3) */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    /* ── Build shader program ──────────────────────────────────────────────── */
    unsigned int vertexShader   = compile_shader(GL_VERTEX_SHADER,   vertexShaderSource);
    unsigned int fragmentShader = compile_shader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    char log[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, log);
        fprintf(stderr, "Shader link error: %s\n", log);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    int modelLoc          = glGetUniformLocation(shaderProgram, "model");
    int viewLoc           = glGetUniformLocation(shaderProgram, "view");
    int projectionLoc     = glGetUniformLocation(shaderProgram, "projection");
    int cameraPosLoc      = glGetUniformLocation(shaderProgram, "cameraPos");
    int lightingEnabledLoc= glGetUniformLocation(shaderProgram, "lightingEnabled");

    mat4 projection;
    mat4_perspective(projection, 45.0f * (3.14159265f / 180.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    mat4 view;
    mat4_translate(view, 0.0f, 0.0f, -2.5f);

    /* Camera sits at z=2.5 in world space (inverse of the view translation) */
    glUseProgram(shaderProgram);
    glUniform3f(cameraPosLoc, 0.0f, 0.0f, 2.5f);

    /* ── Render loop ───────────────────────────────────────────────────────── */
    float last_time    = (float)glfwGetTime();
    int   lighting_on  = 1;
    int   l_key_prev   = GLFW_RELEASE;

    while (!glfwWindowShouldClose(window)) {
        /* Delta time — keeps rotation speed consistent regardless of frame rate */
        float current_time = (float)glfwGetTime();
        float dt = current_time - last_time;
        last_time = current_time;

        /* ── Keyboard input ────────────────────────────────────────────────── */
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, 1);

        /* L — toggle shading. Edge-detect so one press = one toggle */
        int l_key_curr = glfwGetKey(window, GLFW_KEY_L);
        if (l_key_curr == GLFW_PRESS && l_key_prev == GLFW_RELEASE) {
            lighting_on = !lighting_on;
            glfwSetWindowTitle(window, lighting_on ? "Pyramid  [L] shading: ON"
                                                   : "Pyramid  [L] shading: OFF");
        }
        l_key_prev = l_key_curr;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            yaw -= KEY_ROTATION_SPEED * dt;

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            yaw += KEY_ROTATION_SPEED * dt;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            pitch -= KEY_ROTATION_SPEED * dt;

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            pitch += KEY_ROTATION_SPEED * dt;

        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            roll -= KEY_ROTATION_SPEED * dt;

        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            roll += KEY_ROTATION_SPEED * dt;

        /* ── Build model matrix ────────────────────────────────────────────────
           Apply pitch (X rotation) first, then yaw (Y rotation) on top.
           In matrix math, rightmost is applied first:
             model = rotate_y(yaw) * rotate_x(pitch)
        ────────────────────────────────────────────────────────────────────── */
        mat4 rot_x, rot_y, rot_z, tmp, model;
        mat4_rotate_x(rot_x, pitch);
        mat4_rotate_y(rot_y, yaw);
        mat4_rotate_z(rot_z, roll);
        mat4_multiply(tmp,   rot_y, rot_x);
        mat4_multiply(model, tmp,   rot_z);

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glUniformMatrix4fv(modelLoc,      1, GL_FALSE, model);
        glUniformMatrix4fv(viewLoc,       1, GL_FALSE, view);
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projection);
        glUniform1i(lightingEnabledLoc, lighting_on);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 18);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    /* ── Cleanup ───────────────────────────────────────────────────────────── */
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
